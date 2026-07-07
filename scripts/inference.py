from __future__ import annotations

import os
import json
import argparse
import time
from datetime import timedelta
import sys
import glob
import logging
from contextlib import nullcontext
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision.ops import nms
from torchvision import transforms

import geopandas as gpd
from shapely.geometry import box as shapely_box
import rasterio
from rasterio import features as rio_features

# add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
from src.utils.tiling import make_grid_windows, adjust_boxes_to_global
from src.utils.fast_mask import get_mask_clipped, filter_windows_by_mask_raster
from src.utils.make_vrt import write_mosaic_vrt

# ---------------------------
# configuration
# ---------------------------

MASK_DOWNSAMPLE = 16  # downsample AOI mask for faster loading

# ----------------------------
# helpers: logging + normalizer
# ----------------------------

def configure_root_logging():
        root = logging.getLogger()

        # avoid adding handlers multiple times
        if root.handlers:
            return

        root.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)

        root.addHandler(handler)



def setup_logging(log_path: str, rank: int):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger(f"inference_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # prevent double logging
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(
        log_path.replace(".log", f"_rank{rank}.log"),
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger



def build_normalizer(norm_type: str):
    if norm_type.lower() == "imagenet":
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    return None

# -------------------------
# vrt maker
# -------------------------

def resolve_raster_input(path_or_dir):
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*.tif")))
        if not files:
            raise FileNotFoundError(f"no .tif files found in {path_or_dir}")
        parent = os.path.dirname(os.path.abspath(path_or_dir))
        parent_name = os.path.basename(parent)
        out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
        if not os.path.exists(out_vrt):
            write_mosaic_vrt(out_vrt, files)
        return out_vrt

    if any(ch in path_or_dir for ch in ["*", "?", "["]):
        files = sorted(glob.glob(path_or_dir))
        if not files:
            raise FileNotFoundError(f"glob matched no .tif files: {path_or_dir}")
        tile_dir = os.path.dirname(os.path.abspath(path_or_dir))
        parent = os.path.dirname(tile_dir)
        parent_name = os.path.basename(parent)
        out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
        if not os.path.exists(out_vrt):
            write_mosaic_vrt(out_vrt, files)
        return out_vrt

    return path_or_dir

# -------------------------
# ddp helpers
# -------------------------

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def init_distributed_if_needed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(hours=12),
        )
        if torch.cuda.is_available():
            _ = torch.zeros(1, device=f"cuda:{local_rank}")
        return local_rank
    return int(os.environ.get("LOCAL_RANK", 0))

# ----------------
# checkpoint meta
# ----------------

def load_classes_and_task_from_checkpoint(ckpt_path: str):
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        classes = data.get("classes", None)
        task = data.get("task", "detection")
        if classes is not None:
            return classes, task

    json_path = os.path.join(os.path.dirname(ckpt_path), "classes.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)["classes"], "detection"

    raise ValueError("could not determine classes from checkpoint")

# ------------
# arg parsing
# ------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detection", "instance_seg"], default="detection")
    parser.add_argument("--raster_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_vector", default="detections.gpkg")
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--infer_batch", type=int, default=4)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--nms_iou_thresh", type=float, default=0.5)
    parser.add_argument("--final_box_iou", type=float, default=0.5)
    parser.add_argument("--normalize", choices=["none", "imagenet"], default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask_path")
    parser.add_argument("--min_cover_frac", type=float, default=0.0)
    parser.add_argument("--class_area_csv")
    return parser.parse_args()

# -------
# main
# -------

def main():
    args = parse_args()

    # 1. configure root logging FIRST
    configure_root_logging()

    # 2. initialize distributed
    local_rank = init_distributed_if_needed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 3. setup per-rank logger AFTER rank is known
    logger = setup_logging(
        os.path.join(os.path.dirname(args.out_vector) or ".", "inference.log"),
        rank,
    )
    t0 = time.time()

    cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if is_main_process():
        args.raster_path = resolve_raster_input(args.raster_path)
    if dist.is_initialized():
        dist.barrier()
    args.raster_path = resolve_raster_input(args.raster_path)

    if is_main_process():
        logger.info("[debug] starting inference (world_size=%d)", world_size)

    classes, _ = load_classes_and_task_from_checkpoint(args.checkpoint)
    num_classes = len(classes) + 1

    device = f"cuda:{local_rank}" if args.device == "cuda" else "cpu"
    if args.task == "instance_seg":
        model = build_maskrcnn_model(num_classes, pretrained=False, device=device)
    else:
        model = build_fasterrcnn_model(num_classes, pretrained=False, device=device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # -----------------------------
    # windows
    # -----------------------------
    windows_all = make_grid_windows(
        args.raster_path,
        tile_size=args.tile_size,
        stride=args.stride,
    )
    total_before = len(windows_all)

    # -----------------------------
    # AOI mask
    # -----------------------------
    mask_raster = None
    mask_meta = None

    if args.mask_path:
        if is_main_process():
            logger.info("[debug] building AOI-clipped mask (downsample=%dx)", MASK_DOWNSAMPLE)
            mask_raster, mask_meta = get_mask_clipped(
                raster_path=args.raster_path,
                mask_path=args.mask_path,
                cache_dir=os.path.dirname(args.raster_path),
                downsample=MASK_DOWNSAMPLE,
            )

        if dist.is_initialized():
            dist.barrier()

        if not is_main_process():
            mask_raster, mask_meta = get_mask_clipped(
                raster_path=args.raster_path,
                mask_path=args.mask_path,
                cache_dir=os.path.dirname(args.raster_path),
                downsample=MASK_DOWNSAMPLE,
                load_only=True,
            )

        if is_main_process():
            logger.info("[debug] filtering windows by AOI mask")

        windows_all = filter_windows_by_mask_raster(
            mask_raster,
            windows_all,
            min_cover_frac=max(0.0, float(args.min_cover_frac)),
            row0=mask_meta["row0"],
            col0=mask_meta["col0"],
            downsample=mask_meta["downsample"],
        )

        if is_main_process():
            logger.info(
                "[debug] mask filtered windows=%d (from %d)",
                len(windows_all),
                total_before,
            )

    if is_main_process():
        logger.info(
            "[debug] total windows=%d, per-rank≈%d",
            len(windows_all),
            (len(windows_all) + world_size - 1) // world_size,
        )

    windows = windows_all[rank::world_size]


    # buffers (boxes/scores/labels on cpu lists; masks kept as cpu uint8 per detection)
    all_boxes, all_scores, all_labels = [], [], []
    tile_transforms: List = []  # align 1:1 with detections
    masks_flat: List[torch.Tensor] = []  # each is uint8 cpu (h,w)

    # sliding-window loop
    with rasterio.open(args.raster_path) as src:
        normalizer = build_normalizer(args.normalize)
        use_cuda = device.startswith("cuda")
        amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
        # warn if crs is geographic since area filter would be in degrees^2
        try:
            if src.crs and src.crs.is_geographic and is_main_process():
                logger.warning("[warn] raster crs is geographic. reproject to a projected crs for meaningful size threshholding.")
        except Exception:
            pass

        batch_imgs: List[torch.Tensor] = []
        batch_transforms: List = []
        batch_windows: List = []
        batch_size = max(1, int(args.infer_batch))

        def run_batch():
            if not batch_imgs:
                return
            with torch.inference_mode():
                with amp_ctx:
                    preds_list = model(batch_imgs)
                for preds, win, tform in zip(preds_list, batch_windows, batch_transforms):
                    boxes = preds["boxes"]; scores = preds["scores"]; labels = preds["labels"]
                    keep = scores >= args.score_thresh
                    boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
                    boxes_global = adjust_boxes_to_global(boxes, win)
                    # append cpu tensors for boxes/scores/labels
                    all_boxes.append(boxes_global.to("cpu"))
                    all_scores.append(scores.to("cpu"))
                    all_labels.append(labels.to("cpu"))
                    # append transforms and masks per detection (uint8 cpu to save ram)
                    if args.task == 'instance_seg' and 'masks' in preds:
                        m = preds['masks'][keep].squeeze(1)  # (n,h,w) float on device
                        if m.ndim == 2:  # single mask edge case
                            m = m.unsqueeze(0)
                        if m.shape[0] > 0:
                            m_cpu = (m.to("cpu") > 0.5).to(torch.uint8)  # threshold + pack to uint8
                            for k in range(m_cpu.shape[0]):
                                masks_flat.append(m_cpu[k])
                                tile_transforms.append(tform)
                    else:
                        # detection-only: add tform per detection count
                        for _ in range(boxes.shape[0]):
                            tile_transforms.append(tform)
            batch_imgs.clear(); batch_transforms.clear(); batch_windows.clear()

        for i, window in enumerate(windows, start=1):
            if is_main_process() and (i % 50 == 1 or i == len(windows)):
                logger.info("[debug] rank %d processing window %d/%d", rank, i, len(windows))
            img = src.read(window=window)
            tform = src.window_transform(window)
            if img.shape[0] >= 3:
                img = img[:3]
            else:
                repeats = (3 + img.shape[0] - 1) // img.shape[0]
                img = np.concatenate([img] * repeats, axis=0)[:3]
            h, w = img.shape[-2:]
            if h < 32 or w < 32:
                continue
            if not np.isfinite(img).all():
                continue
            scale = 65535.0 if src.dtypes[0] in ("uint16", "int16") else 255.0
            tensor = torch.from_numpy(img).float().div_(scale).clamp_(0.0, 1.0)
            if normalizer is not None:
                tensor = normalizer(tensor)
            if use_cuda:
                tensor = tensor.pin_memory()
            tensor = tensor.to(device, non_blocking=True)
            batch_imgs.append(tensor)
            batch_transforms.append(tform)
            batch_windows.append(window)
            if len(batch_imgs) >= batch_size:
                run_batch()
        run_batch()

    # concat boxes/scores/labels and run per-rank hard nms on device
    if is_main_process():
        logger.info("[debug] concatenating boxes/scores/labels and running per-rank hard nms")
    device_for_nms = device if device.startswith('cuda') else 'cpu'
    boxes = torch.cat(all_boxes, dim=0).to(device_for_nms) if all_boxes else torch.empty((0, 4), device=device_for_nms)
    scores = torch.cat(all_scores, dim=0).to(device_for_nms) if all_scores else torch.empty((0,), device=device_for_nms)
    labels = torch.cat(all_labels, dim=0).to(device_for_nms) if all_labels else torch.empty((0,), dtype=torch.int64, device=device_for_nms)

    keep_indices_all: List[int] = []
    if len(boxes) > 0:
        for cls_idx in labels.unique():
            cls_mask = labels == cls_idx
            if cls_mask.any():
                kept = nms(boxes[cls_mask], scores[cls_mask], args.nms_iou_thresh)
                base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
                keep_indices_all.extend(base_idx[kept].tolist())
        if keep_indices_all:
            keep_t_gpu = torch.tensor(keep_indices_all, dtype=torch.long, device=device_for_nms)
            boxes = boxes[keep_t_gpu]
            scores = scores[keep_t_gpu]
            labels = labels[keep_t_gpu]
        else:
            boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]

    # move kept boxes/scores/labels to cpu; select corresponding masks/transforms on cpu
    boxes = boxes.to("cpu"); scores = scores.to("cpu"); labels = labels.to("cpu")

    # align transforms/masks with kept detections
    if keep_indices_all:
        keep_indices_cpu = torch.tensor(keep_indices_all, dtype=torch.long).tolist()
        if args.task == 'instance_seg':
            kept_masks_flat = [masks_flat[i] for i in keep_indices_cpu]
        else:
            kept_masks_flat = []
        kept_tile_transforms = [tile_transforms[i] for i in keep_indices_cpu]
    else:
        kept_masks_flat = []
        kept_tile_transforms = []

    # fast box-level mask enforcement (pre-vectorization) to drop obviously out-of-AOI detections
    if args.mask_path and len(boxes) > 0 and mask_raster is not None:
        with rasterio.open(args.raster_path) as src:
            height, width = src.height, src.width
        keep_box = []
        min_frac = max(0.0, float(args.min_cover_frac))

        for idx, b in enumerate(boxes.tolist()):
            xmin, ymin, xmax, ymax = b

            # boxes are in global pixel coords already; clamp to raster bounds
            c0 = int(max(0, min(width, min(xmin, xmax))))
            c1 = int(max(0, min(width, max(xmin, xmax))))
            r0 = int(max(0, min(height, min(ymin, ymax))))
            r1 = int(max(0, min(height, max(ymin, ymax))))

            if r1 <= r0 or c1 <= c0:
                continue

            # -------------------------------------------------
            # convert full-res pixel coords → AOI mask coords
            # -------------------------------------------------
            mr0 = (r0 - mask_meta["row0"]) // mask_meta["downsample"]
            mr1 = (r1 - mask_meta["row0"]) // mask_meta["downsample"]
            mc0 = (c0 - mask_meta["col0"]) // mask_meta["downsample"]
            mc1 = (c1 - mask_meta["col0"]) // mask_meta["downsample"]

            # clip to mask raster bounds
            hm, wm = mask_raster.shape
            mr0 = max(0, min(hm, mr0))
            mr1 = max(0, min(hm, mr1))
            mc0 = max(0, min(wm, mc0))
            mc1 = max(0, min(wm, mc1))

            if mr1 <= mr0 or mc1 <= mc0:
                continue

            chip = mask_raster[mr0:mr1, mc0:mc1]
            if chip.size == 0:
                continue

            cover = float(chip.sum()) / float(chip.size)
            if cover >= min_frac and chip.sum() > 0:
                keep_box.append(idx)

        if keep_box:
            keep_t = torch.tensor(keep_box, dtype=torch.long)
            boxes = boxes.index_select(0, keep_t)
            scores = scores.index_select(0, keep_t)
            labels = labels.index_select(0, keep_t)
            if args.task == 'instance_seg':
                kept_masks_flat = [kept_masks_flat[i] for i in keep_box]
            kept_tile_transforms = [kept_tile_transforms[i] for i in keep_box]
        else:
            boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]
            kept_masks_flat = []; kept_tile_transforms = []

    # vectorize to polygons (per-rank) — still needed to emit polygons, but we keep it lightweight
    if is_main_process():
        logger.info("[debug] vectorizing features to polygons")
    geoms, class_names, score_vals = [], [], []
    with rasterio.open(args.raster_path) as src:
        crs = src.crs
        global_transform = src.transform
    if args.task == 'instance_seg' and boxes.numel() > 0 and len(kept_masks_flat) > 0:
        for m_u8, tform, s, l in zip(kept_masks_flat, kept_tile_transforms, scores.tolist(), labels.tolist()):
            m_np = m_u8.numpy().astype('uint8')
            for geom, val in rio_features.shapes(m_np, transform=tform):
                if int(val) == 0:
                    continue
                from shapely.geometry import shape as shapely_shape
                geoms.append(shapely_shape(geom))
                class_names.append(classes[l - 1])
                score_vals.append(float(s))
    else:
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            xmin, ymin, xmax, ymax = b
            x0, y0 = global_transform * (xmin, ymin)
            x1, y1 = global_transform * (xmax, ymax)
            geom = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            geoms.append(geom)
            class_names.append(classes[l - 1])
            score_vals.append(float(s))

    gdf = gpd.GeoDataFrame({"classname": class_names, "score": score_vals}, geometry=geoms, crs=crs)

    # write per-rank partial (no heavy geopandas processing here)
    out_dir = os.path.dirname(args.out_vector) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract layer name from output filename (e.g., detections_Benton_20260707_145300 from detections_Benton_20260707_145300.gpkg)
    base, ext = os.path.splitext(os.path.basename(args.out_vector))
    layer_name = base  # Use filename without extension as layer name
    
    partial_path = f"{os.path.dirname(args.out_vector)}/{base}_rank{rank}{ext if ext.lower()=='.gpkg' else '.gpkg'}"
    gdf.to_file(partial_path, driver="GPKG", layer=layer_name)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # rank 0: merge, enforce mask via raster (fast), final box-nms dedupe, area filter, write final
    if is_main_process():
        import pandas as pd
        parts = [f"{base}_rank{r}{ext if ext.lower()=='.gpkg' else '.gpkg'}" for r in range(world_size)]
        gdfs = []
        for p in parts:
            if os.path.exists(p):
                try:
                    gdfs.append(gpd.read_file(p, layer=layer_name))
                except Exception:
                    logger.exception("[warn] failed reading %s; skipping", p)
        if len(gdfs) > 0:
            merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs if gdfs[0].crs else crs)
        else:
            merged = gpd.GeoDataFrame({"classname": [], "score": []}, geometry=[], crs=crs)


        # fast raster mask keep-only at polygon level (no costly overlay)
        if args.mask_path:
            with rasterio.open(args.raster_path) as src:
                transform = src.transform

            keep_idx = []
            min_frac = max(0.0, float(args.min_cover_frac))

            # mask dimensions (AOI-clipped, downsampled)
            hm, wm = mask_raster.shape
            row0 = mask_meta["row0"]
            col0 = mask_meta["col0"]
            ds = mask_meta["downsample"]

            for i, geom in enumerate(merged.geometry):
                if geom is None or geom.is_empty:
                    continue

                # -----------------------------------------
                # polygon bounds → pixel coords (full-res)
                # -----------------------------------------
                minx, miny, maxx, maxy = geom.bounds
                inv = ~transform

                c0, r0 = inv * (minx, miny)
                c1, r1 = inv * (maxx, maxy)

                cmin, cmax = sorted((int(c0), int(c1)))
                rmin, rmax = sorted((int(r0), int(r1)))

                if rmax <= rmin or cmax <= cmin:
                    continue

                # ------------------------------------------------
                # full-res pixel coords → AOI mask coords
                # ------------------------------------------------
                mrmin = (rmin - row0) // ds
                mrmax = (rmax - row0) // ds
                mcmin = (cmin - col0) // ds
                mcmax = (cmax - col0) // ds

                # clip to mask raster bounds
                mrmin = max(0, min(hm, mrmin))
                mrmax = max(0, min(hm, mrmax))
                mcmin = max(0, min(wm, mcmin))
                mcmax = max(0, min(wm, mcmax))

                if mrmax <= mrmin or mcmax <= mcmin:
                    continue

                chip = mask_raster[mrmin:mrmax, mcmin:mcmax]
                if chip.size == 0:
                    continue

                cover = float(chip.sum()) / float(chip.size)
                if cover >= min_frac and chip.sum() > 0:
                    keep_idx.append(i)

            merged = merged.iloc[keep_idx].copy() if keep_idx else merged.iloc[0:0].copy()
            logger.info(
                "[debug] final masking kept %d of %d features",
                len(merged),
                sum(len(x) for x in gdfs) if gdfs else 0,
            )

        # final fast dedupe: class-wise nms over polygon bounding boxes
        if len(merged) > 1:
            kept_rows = []
            for cls, group in merged.groupby("classname", group_keys=False):
                if len(group) == 1:
                    kept_rows.append(group.index[0])
                    continue
                # build boxes and scores tensors on cpu from geometry bounds (map coords)
                boxes_np = np.array([list(g.bounds) for g in group.geometry], dtype=np.float32)
                # convert to xyxy in same order (minx,miny,maxx,maxy) already
                boxes_t = torch.from_numpy(boxes_np)
                scores_t = torch.from_numpy(group["score"].astype(float).to_numpy().astype(np.float32))
                kept = nms(boxes_t, scores_t, args.final_box_iou)
                kept_rows.extend(group.index.to_numpy()[kept.numpy()].tolist())
            merged = merged.loc[kept_rows].copy()
            logger.info("[debug] final box-nms dedupe kept %d features", len(merged))

        # post-vectorization area filter
        pre_count = len(merged)
        if args.class_area_csv and os.path.exists(args.class_area_csv) and pre_count > 0:
            df = pd.read_csv(args.class_area_csv)

            # build lookup: classname -> (min_sqft, max_sqft)
            area_rules = {
                str(row["feature"]): (float(row["min"]), float(row["max"]))
                for _, row in df.iterrows()
            }

            keep_idx = []
            dropped = 0

            # compute polygon areas in CRS units (already sq ft)
            areas = merged.geometry.area.values

            for i, (cls, a) in enumerate(zip(merged["classname"], areas)):
                rule = area_rules.get(str(cls))
                if rule is None:
                    # no rule for this class → keep (change to "continue" if you prefer dropping)
                    keep_idx.append(i)
                    continue

                min_a, max_a = rule
                if min_a <= a <= max_a:
                    keep_idx.append(i)
                else:
                    dropped += 1

            merged = merged.iloc[keep_idx].copy()
        post_count = len(merged)
        logger.info("[debug] final area thresholding kept %d features (dropped %d under or over size threshhold)",
            post_count, pre_count - post_count)

        final_path = args.out_vector if args.out_vector.lower().endswith('.gpkg') else base + '.gpkg'
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        merged.to_file(final_path, driver="GPKG", layer=layer_name)
        logger.info("[done] wrote %d features → %s (elapsed %.2fs)",
                    post_count, final_path, time.time() - t0)


    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()




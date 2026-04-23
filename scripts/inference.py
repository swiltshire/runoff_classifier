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

MASK_DOWNSAMPLE = 16  # recommended for 6-inch imagery

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
    layer_name = "detections"
    base, ext = os.path.splitext(args.out_vector)
    partial_path = f"{base}_rank{rank}{ext if ext.lower()=='.gpkg' else '.gpkg'}"
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





# from __future__ import annotations
# import os
# import json
# import argparse
# import time
# from datetime import timedelta
# import sys
# import glob
# import logging
# from contextlib import nullcontext
# from typing import List

# import numpy as np
# import pandas as pd

# import torch
# import torch.distributed as dist
# import torch.backends.cudnn as cudnn
# from torchvision.ops import nms
# from torchvision import transforms

# import geopandas as gpd
# from shapely.geometry import box as shapely_box
# import rasterio
# from rasterio import features as rio_features

# # add project root to path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
# from src.utils.tiling import make_grid_windows, adjust_boxes_to_global
# from src.utils.fast_mask import get_mask, filter_windows_by_mask_raster
# from src.utils.make_vrt import write_mosaic_vrt


# MASK_DOWNSAMPLE = 16  # AOI mask downsampling factor


# # ----------------------------
# # helpers: logging + normalizer
# # ----------------------------

# def setup_logging(log_path: str, rank: int):
#     os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
#     logger = logging.getLogger(f"inference_rank{rank}")
#     logger.setLevel(logging.INFO)
#     logger.handlers.clear()
#     fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    
#     # logging: only output rank0
#     # sh = logging.StreamHandler(sys.stdout if rank == 0 else open(os.devnull, 'w'))
#     # all ranks print to console; can be very chatty
#     sh = logging.StreamHandler(sys.stdout)

#     sh.setFormatter(fmt)
#     fh = logging.FileHandler(log_path.replace(".log", f"_rank{rank}.log"), mode="w", encoding="utf-8")
#     fh.setFormatter(fmt)
#     logger.addHandler(sh)
#     logger.addHandler(fh)
#     return logger


# def build_normalizer(norm_type: str):
#     if norm_type.lower() == "imagenet":
#         return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     return None


# # -------------------------
# # vrt maker
# # -------------------------

# def resolve_raster_input(path_or_dir):
#     """
#     if path_or_dir is a directory of .tif tiles or a *.tif glob:
#     → build a vrt one level up from the tiles folder (re-use if exists).
#     otherwise:
#     → return path_or_dir unchanged.
#     """
#     if os.path.isdir(path_or_dir):
#         files = sorted(glob.glob(os.path.join(path_or_dir, "*.tif")))
#         if not files:
#             raise FileNotFoundError(f"no .tif files found in {path_or_dir}")
#         parent = os.path.dirname(os.path.abspath(path_or_dir))
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
#         if not os.path.exists(out_vrt):
#             write_mosaic_vrt(out_vrt, files)
#         return out_vrt
#     if any(ch in path_or_dir for ch in ["*", "?", "["]):
#         files = sorted(glob.glob(path_or_dir))
#         if not files:
#             raise FileNotFoundError(f"glob matched no .tif files: {path_or_dir}")
#         tile_dir = os.path.dirname(os.path.abspath(path_or_dir))
#         parent = os.path.dirname(tile_dir)
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
#         if not os.path.exists(out_vrt):
#             write_mosaic_vrt(out_vrt, files)
#         return out_vrt
#     return path_or_dir

# # -------------------------
# # ddp helpers (init + rank)
# # -------------------------

# def is_main_process() -> bool:
#     return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# def init_distributed_if_needed():
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         backend = 'nccl' if torch.cuda.is_available() else 'gloo'
#         local_rank = int(os.environ.get('LOCAL_RANK', 0))
#         if torch.cuda.is_available():
#             torch.cuda.set_device(local_rank)
#         dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
#         # touch device once so nccl learns local mapping; quiets early barrier warnings
#         if torch.cuda.is_available():
#             _ = torch.zeros(1, device=f'cuda:{local_rank}')
#         return local_rank
#     return int(os.environ.get('LOCAL_RANK', 0))

# # ----------------
# # checkpoint meta
# # ----------------

# def load_classes_and_task_from_checkpoint(ckpt_path: str):
#     data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     if isinstance(data, dict):
#         classes = data.get('classes', None)
#         task = data.get('task', 'detection')
#         if classes is not None:
#             return classes, task
#     json_path = os.path.join(os.path.dirname(ckpt_path), 'classes.json')
#     if os.path.exists(json_path):
#         with open(json_path) as f:
#             return json.load(f)['classes'], 'detection'
#     raise ValueError('could not determine classes from checkpoint or sidecar json')

# # ------------
# # arg parsing
# # ------------

# def parse_args():
#     parser = argparse.ArgumentParser(description='optimized sliding-window inference for boxes or instance masks (ddp + global nms + fast mask + final box-dedupe)')
#     parser.add_argument('--task', type=str, default='detection', choices=['detection', 'instance_seg'])
#     parser.add_argument('--raster_path', type=str, required=True)
#     parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'])
#     parser.add_argument('--checkpoint', type=str, required=True)
#     parser.add_argument('--out_vector', type=str, default='detections.gpkg')
#     parser.add_argument('--tile_size', type=int, default=1024)
#     parser.add_argument('--stride', type=int, default=512)
#     parser.add_argument('--score_thresh', type=float, default=0.5)
#     parser.add_argument('--nms_iou_thresh', type=float, default=0.5)
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--infer_batch', type=int, default=4, help='windows per forward pass per rank')
#     # global merge + soft-nms controls kept for compatibility (but we default to hard global nms)
#     parser.add_argument('--global_nms', action='store_true',
#                         help='enable global (cross-rank) class-wise nms after per-rank nms (recommended when stride < tile_size)')
#     parser.add_argument('--soft_nms', action='store_true',
#                         help='use soft-nms in the global merge step (ignored if --global_nms not set)')
#     parser.add_argument('--soft_nms_method', type=str, default='gaussian', choices=['gaussian', 'linear'],
#                         help='soft-nms decay method')
#     parser.add_argument('--soft_nms_sigma', type=float, default=0.5,
#                         help='soft-nms gaussian sigma (typical 0.5)')
#     parser.add_argument('--soft_nms_score_thresh', type=float, default=0.001,
#                         help='drop boxes whose decayed score falls below this threshold during soft-nms')
#     # mask-based window filtering + box-level enforcement
#     parser.add_argument('--mask_path', type=str, default=None,
#                         help='optional polygon mask to restrict windows and final detections to areas of interest')
#     parser.add_argument('--min_cover_frac', type=float, default=0.0,
#                         help='minimum fraction of a window/box that must intersect the mask to keep it (0..1)')
#     # final fast dedupe (box-nms on polygon bounds)
#     parser.add_argument('--final_box_iou', type=float, default=0.5,
#                         help='class-wise nms over polygon bounding boxes after merge; removes stacked duplicates efficiently')
#     # post-vectorization area filter (in crs units, which are square feet here)
#     parser.add_argument('--class_area_csv', type=str,
#                         default='root/data/feature_size_threshholds.csv',
#                         help='csv with per-class min/max feature area in square feet')
#     return parser.parse_args()

# # -------
# # main
# # -------

# def main():
#     args = parse_args()

#     # fast conv selection
#     cudnn.benchmark = True
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")

#     # ddp init (no-op for single process)
#     local_rank = init_distributed_if_needed()
#     world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
#     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

#     # resolve raster on rank 0, then barrier
#     if is_main_process():
#         args.raster_path = resolve_raster_input(args.raster_path)
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()
#     args.raster_path = resolve_raster_input(args.raster_path)

#     # logger
#     log_path = os.path.join(os.path.dirname(args.out_vector) or ".", "inference.log")
#     logger = setup_logging(log_path, rank)
#     t0 = time.time()
#     if is_main_process():
#         logger.info("[debug] starting inference (world_size=%d)", world_size)

#     # early checkpoint existence check (rank 0)
#     if is_main_process():
#         if not os.path.exists(args.checkpoint):
#             logger.error("[fatal] checkpoint not found: %s", args.checkpoint)
#             if dist.is_available() and dist.is_initialized():
#                 dist.barrier()
#             sys.exit(1)
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()

#     # model + classes
#     classes, task_from_ckpt = load_classes_and_task_from_checkpoint(args.checkpoint)
#     if args.task != task_from_ckpt:
#         logger.warning("[warn] overriding task from checkpoint (%s) with arg (%s)", task_from_ckpt, args.task)
#     num_classes = len(classes) + 1

#     device = args.device
#     if device == 'cuda' and torch.cuda.is_available():
#         device = f'cuda:{local_rank}'
#         torch.cuda.set_device(local_rank)

#     if args.task == 'instance_seg':
#         model = build_maskrcnn_model(num_classes=num_classes, pretrained=False, device=device)
#     else:
#         model = build_fasterrcnn_model(num_classes=num_classes, pretrained=False, device=device)
#     state = torch.load(args.checkpoint, map_location=device, weights_only=False)
#     model.load_state_dict(state["model_state"])
#     model.eval()
#     if is_main_process():
#         logger.info("[debug] model loaded and set to eval (device=%s)", device)

#     # build full grid once, optionally filter by rasterized mask before sharding
#     windows_all = make_grid_windows(args.raster_path, tile_size=args.tile_size, stride=args.stride)
#     total_before = len(windows_all)
    
   
#     mask_raster = None
#     if args.mask_path:
#         if is_main_process():
#             logger.info(
#                 "[debug] rasterizing polygon mask (NHD) at %dx downsample, or fetching cached",
#                 MASK_DOWNSAMPLE,
#             )
#             mask_raster = get_mask(
#                 args.raster_path,
#                 args.mask_path,
#                 cache_dir=os.path.dirname(args.raster_path),
#                 simplify_tol_px=0.5,
#                 downsample=MASK_DOWNSAMPLE,   # downsample according to specified factor (increase speed)
#             )

#     # wait for rank 0 to finish writing cache
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()

#     # all non‑zero ranks load cached mask
#     if not is_main_process():
#         mask_raster = get_mask(
#             args.raster_path,
#             args.mask_path,
#             cache_dir=os.path.dirname(args.raster_path),
#             downsample=MASK_DOWNSAMPLE,
#             load_only=True,              # (avoid rasterization)
#         )


#         if is_main_process():
#             logger.info("[debug] filtering windows by mask raster")
        
#         windows_all = filter_windows_by_mask_raster(
#             mask_raster, windows_all, min_cover_frac=max(0.0, float(args.min_cover_frac))
#         )
#         if is_main_process():
#             logger.info("[debug] mask filtered windows=%d (from %d) [min_cover_frac=%.3f]",
#                         len(windows_all), total_before, float(args.min_cover_frac))
#     if is_main_process():
#         logger.info("[debug] total windows=%d, per-rank≈%d",
#                     len(windows_all), (len(windows_all) + world_size - 1) // world_size)

#     # shard AFTER filtering
#     windows = windows_all[rank::world_size]

#     # buffers (boxes/scores/labels on cpu lists; masks kept as cpu uint8 per detection)
#     all_boxes, all_scores, all_labels = [], [], []
#     tile_transforms: List = []  # align 1:1 with detections
#     masks_flat: List[torch.Tensor] = []  # each is uint8 cpu (h,w)

#     # sliding-window loop
#     with rasterio.open(args.raster_path) as src:
#         normalizer = build_normalizer(args.normalize)
#         use_cuda = device.startswith("cuda")
#         amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
#         # warn if crs is geographic since area filter would be in degrees^2
#         try:
#             if src.crs and src.crs.is_geographic and is_main_process():
#                 logger.warning("[warn] raster crs is geographic. reproject to a projected crs for meaningful size threshholding.")
#         except Exception:
#             pass

#         batch_imgs: List[torch.Tensor] = []
#         batch_transforms: List = []
#         batch_windows: List = []
#         batch_size = max(1, int(args.infer_batch))

#         def run_batch():
#             if not batch_imgs:
#                 return
#             with torch.inference_mode():
#                 with amp_ctx:
#                     preds_list = model(batch_imgs)
#                 for preds, win, tform in zip(preds_list, batch_windows, batch_transforms):
#                     boxes = preds["boxes"]; scores = preds["scores"]; labels = preds["labels"]
#                     keep = scores >= args.score_thresh
#                     boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
#                     boxes_global = adjust_boxes_to_global(boxes, win)
#                     # append cpu tensors for boxes/scores/labels
#                     all_boxes.append(boxes_global.to("cpu"))
#                     all_scores.append(scores.to("cpu"))
#                     all_labels.append(labels.to("cpu"))
#                     # append transforms and masks per detection (uint8 cpu to save ram)
#                     if args.task == 'instance_seg' and 'masks' in preds:
#                         m = preds['masks'][keep].squeeze(1)  # (n,h,w) float on device
#                         if m.ndim == 2:  # single mask edge case
#                             m = m.unsqueeze(0)
#                         if m.shape[0] > 0:
#                             m_cpu = (m.to("cpu") > 0.5).to(torch.uint8)  # threshold + pack to uint8
#                             for k in range(m_cpu.shape[0]):
#                                 masks_flat.append(m_cpu[k])
#                                 tile_transforms.append(tform)
#                     else:
#                         # detection-only: add tform per detection count
#                         for _ in range(boxes.shape[0]):
#                             tile_transforms.append(tform)
#             batch_imgs.clear(); batch_transforms.clear(); batch_windows.clear()

#         for i, window in enumerate(windows, start=1):
#             if is_main_process() and (i % 50 == 1 or i == len(windows)):
#                 logger.info("[debug] rank %d processing window %d/%d", rank, i, len(windows))
#             img = src.read(window=window)
#             tform = src.window_transform(window)
#             if img.shape[0] >= 3:
#                 img = img[:3]
#             else:
#                 repeats = (3 + img.shape[0] - 1) // img.shape[0]
#                 img = np.concatenate([img] * repeats, axis=0)[:3]
#             h, w = img.shape[-2:]
#             if h < 32 or w < 32:
#                 continue
#             if not np.isfinite(img).all():
#                 continue
#             scale = 65535.0 if src.dtypes[0] in ("uint16", "int16") else 255.0
#             tensor = torch.from_numpy(img).float().div_(scale).clamp_(0.0, 1.0)
#             if normalizer is not None:
#                 tensor = normalizer(tensor)
#             if use_cuda:
#                 tensor = tensor.pin_memory()
#             tensor = tensor.to(device, non_blocking=True)
#             batch_imgs.append(tensor)
#             batch_transforms.append(tform)
#             batch_windows.append(window)
#             if len(batch_imgs) >= batch_size:
#                 run_batch()
#         run_batch()

#     # concat boxes/scores/labels and run per-rank hard nms on device
#     if is_main_process():
#         logger.info("[debug] concatenating boxes/scores/labels and running per-rank hard nms")
#     device_for_nms = device if device.startswith('cuda') else 'cpu'
#     boxes = torch.cat(all_boxes, dim=0).to(device_for_nms) if all_boxes else torch.empty((0, 4), device=device_for_nms)
#     scores = torch.cat(all_scores, dim=0).to(device_for_nms) if all_scores else torch.empty((0,), device=device_for_nms)
#     labels = torch.cat(all_labels, dim=0).to(device_for_nms) if all_labels else torch.empty((0,), dtype=torch.int64, device=device_for_nms)

#     keep_indices_all: List[int] = []
#     if len(boxes) > 0:
#         for cls_idx in labels.unique():
#             cls_mask = labels == cls_idx
#             if cls_mask.any():
#                 kept = nms(boxes[cls_mask], scores[cls_mask], args.nms_iou_thresh)
#                 base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
#                 keep_indices_all.extend(base_idx[kept].tolist())
#         if keep_indices_all:
#             keep_t_gpu = torch.tensor(keep_indices_all, dtype=torch.long, device=device_for_nms)
#             boxes = boxes[keep_t_gpu]
#             scores = scores[keep_t_gpu]
#             labels = labels[keep_t_gpu]
#         else:
#             boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]

#     # move kept boxes/scores/labels to cpu; select corresponding masks/transforms on cpu
#     boxes = boxes.to("cpu"); scores = scores.to("cpu"); labels = labels.to("cpu")

#     # align transforms/masks with kept detections
#     if keep_indices_all:
#         keep_indices_cpu = torch.tensor(keep_indices_all, dtype=torch.long).tolist()
#         if args.task == 'instance_seg':
#             kept_masks_flat = [masks_flat[i] for i in keep_indices_cpu]
#         else:
#             kept_masks_flat = []
#         kept_tile_transforms = [tile_transforms[i] for i in keep_indices_cpu]
#     else:
#         kept_masks_flat = []
#         kept_tile_transforms = []

#     # fast box-level mask enforcement (pre-vectorization) to drop obviously out-of-AOI detections
#     if args.mask_path and len(boxes) > 0 and mask_raster is not None:
#         with rasterio.open(args.raster_path) as src:
#             height, width = src.height, src.width
#         keep_box = []
#         min_frac = max(0.0, float(args.min_cover_frac))
#         for idx, b in enumerate(boxes.tolist()):
#             xmin, ymin, xmax, ymax = b
#             # boxes are in global pixel coords already; clamp to raster bounds
#             c0 = int(max(0, min(width, min(xmin, xmax))))
#             c1 = int(max(0, min(width, max(xmin, xmax))))
#             r0 = int(max(0, min(height, min(ymin, ymax))))
#             r1 = int(max(0, min(height, max(ymin, ymax))))
#             if r1 <= r0 or c1 <= c0:
#                 continue
            
#             # convert full‑res pixel coords → mask coords
#             mr0 = r0 // MASK_DOWNSAMPLE
#             mr1 = r1 // MASK_DOWNSAMPLE
#             mc0 = c0 // MASK_DOWNSAMPLE
#             mc1 = c1 // MASK_DOWNSAMPLE

#             chip = mask_raster[mr0:mr1, mc0:mc1]

#             if chip.size == 0:
#                         continue

#             cover = float(chip.sum()) / float(chip.size)
#             if cover >= min_frac and chip.sum() > 0:
#                 keep_box.append(idx)
#         if keep_box:
#             keep_t = torch.tensor(keep_box, dtype=torch.long)
#             boxes = boxes.index_select(0, keep_t)
#             scores = scores.index_select(0, keep_t)
#             labels = labels.index_select(0, keep_t)
#             if args.task == 'instance_seg':
#                 kept_masks_flat = [kept_masks_flat[i] for i in keep_box]
#             kept_tile_transforms = [kept_tile_transforms[i] for i in keep_box]
#         else:
#             boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]
#             kept_masks_flat = []; kept_tile_transforms = []

#     # vectorize to polygons (per-rank) — still needed to emit polygons, but we keep it lightweight
#     if is_main_process():
#         logger.info("[debug] vectorizing features to polygons")
#     geoms, class_names, score_vals = [], [], []
#     with rasterio.open(args.raster_path) as src:
#         crs = src.crs
#         global_transform = src.transform
#     if args.task == 'instance_seg' and boxes.numel() > 0 and len(kept_masks_flat) > 0:
#         for m_u8, tform, s, l in zip(kept_masks_flat, kept_tile_transforms, scores.tolist(), labels.tolist()):
#             m_np = m_u8.numpy().astype('uint8')
#             for geom, val in rio_features.shapes(m_np, transform=tform):
#                 if int(val) == 0:
#                     continue
#                 from shapely.geometry import shape as shapely_shape
#                 geoms.append(shapely_shape(geom))
#                 class_names.append(classes[l - 1])
#                 score_vals.append(float(s))
#     else:
#         for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
#             xmin, ymin, xmax, ymax = b
#             x0, y0 = global_transform * (xmin, ymin)
#             x1, y1 = global_transform * (xmax, ymax)
#             geom = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
#             geoms.append(geom)
#             class_names.append(classes[l - 1])
#             score_vals.append(float(s))

#     gdf = gpd.GeoDataFrame({"classname": class_names, "score": score_vals}, geometry=geoms, crs=crs)

#     # write per-rank partial (no heavy geopandas processing here)
#     out_dir = os.path.dirname(args.out_vector) or "."
#     os.makedirs(out_dir, exist_ok=True)
#     layer_name = "detections"
#     base, ext = os.path.splitext(args.out_vector)
#     partial_path = f"{base}_rank{rank}{ext if ext.lower()=='.gpkg' else '.gpkg'}"
#     gdf.to_file(partial_path, driver="GPKG", layer=layer_name)
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()

#     # rank 0: merge, enforce mask via raster (fast), final box-nms dedupe, area filter, write final
#     if is_main_process():
#         import pandas as pd
#         parts = [f"{base}_rank{r}{ext if ext.lower()=='.gpkg' else '.gpkg'}" for r in range(world_size)]
#         gdfs = []
#         for p in parts:
#             if os.path.exists(p):
#                 try:
#                     gdfs.append(gpd.read_file(p, layer=layer_name))
#                 except Exception:
#                     logger.exception("[warn] failed reading %s; skipping", p)
#         if len(gdfs) > 0:
#             merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs if gdfs[0].crs else crs)
#         else:
#             merged = gpd.GeoDataFrame({"classname": [], "score": []}, geometry=[], crs=crs)

#         # fast raster mask keep-only at polygon level (no costly overlay)
#         if args.mask_path:
#             with rasterio.open(args.raster_path) as src:
#                 transform = src.transform
#             keep_idx = []
#             min_frac = max(0.0, float(args.min_cover_frac))
#             for i, geom in enumerate(merged.geometry):
#                 if geom is None or geom.is_empty:
#                     continue
#                 # compute bbox of polygon in pixel coords
#                 minx, miny, maxx, maxy = geom.bounds
#                 # world->pixel
#                 inv = ~transform
#                 c0, r0 = inv * (minx, miny)
#                 c1, r1 = inv * (maxx, maxy)
#                 cmin, cmax = sorted([int(c0), int(c1)])
#                 rmin, rmax = sorted([int(r0), int(r1)])
#                 # clip to raster bounds
                
#                 h, w = mask_raster.shape

#                 mrmin = rmin // MASK_DOWNSAMPLE
#                 mrmax = rmax // MASK_DOWNSAMPLE
#                 mcmin = cmin // MASK_DOWNSAMPLE
#                 mcmax = cmax // MASK_DOWNSAMPLE

#                 mrmin = max(0, min(h, mrmin))
#                 mrmax = max(0, min(h, mrmax))
#                 mcmin = max(0, min(w, mcmin))
#                 mcmax = max(0, min(w, mcmax))

#                 if mrmax <= mrmin or mcmax <= mcmin:
#                     continue

#                 chip = mask_raster[mrmin:mrmax, mcmin:mcmax]

#                 cover = float(chip.sum()) / float(chip.size)
#                 if cover >= min_frac and chip.sum() > 0:
#                     keep_idx.append(i)
#             merged = merged.iloc[keep_idx].copy() if keep_idx else merged.iloc[0:0].copy()
#             logger.info("[debug] final masking kept %d of %d features", len(merged), sum(len(x) for x in gdfs) if gdfs else 0)

#         # final fast dedupe: class-wise nms over polygon bounding boxes
#         if len(merged) > 1:
#             kept_rows = []
#             for cls, group in merged.groupby("classname", group_keys=False):
#                 if len(group) == 1:
#                     kept_rows.append(group.index[0])
#                     continue
#                 # build boxes and scores tensors on cpu from geometry bounds (map coords)
#                 boxes_np = np.array([list(g.bounds) for g in group.geometry], dtype=np.float32)
#                 # convert to xyxy in same order (minx,miny,maxx,maxy) already
#                 boxes_t = torch.from_numpy(boxes_np)
#                 scores_t = torch.from_numpy(group["score"].astype(float).to_numpy().astype(np.float32))
#                 kept = nms(boxes_t, scores_t, args.final_box_iou)
#                 kept_rows.extend(group.index.to_numpy()[kept.numpy()].tolist())
#             merged = merged.loc[kept_rows].copy()
#             logger.info("[debug] final box-nms dedupe kept %d features", len(merged))

#         # post-vectorization area filter
#         pre_count = len(merged)
#         if args.class_area_csv and os.path.exists(args.class_area_csv) and pre_count > 0:
#             df = pd.read_csv(args.class_area_csv)

#             # build lookup: classname -> (min_sqft, max_sqft)
#             area_rules = {
#                 str(row["feature"]): (float(row["min"]), float(row["max"]))
#                 for _, row in df.iterrows()
#             }

#             keep_idx = []
#             dropped = 0

#             # compute polygon areas in CRS units (already sq ft)
#             areas = merged.geometry.area.values

#             for i, (cls, a) in enumerate(zip(merged["classname"], areas)):
#                 rule = area_rules.get(str(cls))
#                 if rule is None:
#                     # no rule for this class → keep (change to "continue" if you prefer dropping)
#                     keep_idx.append(i)
#                     continue

#                 min_a, max_a = rule
#                 if min_a <= a <= max_a:
#                     keep_idx.append(i)
#                 else:
#                     dropped += 1

#             merged = merged.iloc[keep_idx].copy()
#         post_count = len(merged)
#         logger.info("[debug] final area thresholding kept %d features (dropped %d under or over size threshhold)",
#             post_count, pre_count - post_count)

#         final_path = args.out_vector if args.out_vector.lower().endswith('.gpkg') else base + '.gpkg'
#         os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
#         merged.to_file(final_path, driver="GPKG", layer=layer_name)
#         logger.info("[done] wrote %d features → %s (elapsed %.2fs)",
#                     post_count, final_path, time.time() - t0)

#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()
#         dist.destroy_process_group()

# if __name__ == '__main__':
#     main()









# backup 4: working but final mask and dedupe took forever because they were done on all ranks

# from __future__ import annotations
# import os
# import json
# import argparse
# import time
# from datetime import timedelta
# import sys
# import glob
# import logging
# from contextlib import nullcontext

# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.backends.cudnn as cudnn
# from torchvision.ops import nms, box_iou
# from torchvision import transforms
# import geopandas as gpd
# from shapely.geometry import box as shapely_box
# import rasterio
# from rasterio import features as rio_features
# from affine import Affine  # for reconstructing tile transforms when needed

# # add project root to path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
# from src.utils.tiling import make_grid_windows, adjust_boxes_to_global #, filter_windows_by_mask
# from src.utils.fast_mask import rasterize_mask_aligned, filter_windows_by_mask_raster

# # ----------------------------
# # helpers: logging + normalizer
# # ----------------------------

# def setup_logging(log_path: str, rank: int):
#     os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
#     logger = logging.getLogger(f"inference_rank{rank}")
#     logger.setLevel(logging.INFO)
#     logger.handlers.clear()
#     fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
#     sh = logging.StreamHandler(sys.stdout if rank == 0 else open(os.devnull, 'w'))
#     sh.setFormatter(fmt)
#     fh = logging.FileHandler(log_path.replace(".log", f"_rank{rank}.log"), mode="w", encoding="utf-8")
#     fh.setFormatter(fmt)
#     logger.addHandler(sh)
#     logger.addHandler(fh)
#     return logger


# def build_normalizer(norm_type: str):
#     if norm_type.lower() == "imagenet":
#         return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     return None


# # -------------
# # soft-nms impl
# # -------------

# def soft_nms(
#     boxes: torch.Tensor,
#     scores: torch.Tensor,
#     iou_thresh: float = 0.5,
#     sigma: float = 0.5,
#     score_thresh: float = 0.001,
#     method: str = "gaussian",
# ) -> torch.Tensor:
#     """
#     soft-nms (linear or gaussian) returning kept indices (in the order selected).
#     boxes: tensor [n, 4] (xyxy in a common coordinate frame)
#     scores: tensor [n]
#     iou_thresh: used by 'linear' method as the decay onset
#     sigma: gaussian sigma (typical 0.5)
#     score_thresh: drop boxes whose decayed score falls below this
#     method: 'gaussian' or 'linear'
#     """
#     assert boxes.ndim == 2 and boxes.size(1) == 4
#     assert scores.ndim == 1 and scores.size(0) == boxes.size(0)
#     method = method.lower()
#     assert method in ("gaussian", "linear")

#     scores = scores.clone()
#     device = boxes.device

#     idxs = torch.arange(boxes.size(0), device=device)
#     keep = []

#     while idxs.numel() > 0:
#         # pick best current
#         best_local = torch.argmax(scores[idxs])
#         best = idxs[best_local]
#         keep.append(best.item())

#         if idxs.numel() == 1:
#             break
#         others = torch.cat([idxs[:best_local], idxs[best_local + 1:]])

#         # decay scores by iou with best
#         ious = box_iou(boxes[best].unsqueeze(0), boxes[others]).squeeze(0)
#         if method == "gaussian":
#             decay = torch.exp(- (ious * ious) / max(1e-8, sigma))
#             scores[others] = scores[others] * decay
#         else:
#             decay = torch.ones_like(ious, device=device)
#             mask = ious > iou_thresh
#             decay[mask] = 1.0 - ious[mask]
#             scores[others] = scores[others] * decay

#         # threshold low scores
#         remain_mask = scores[others] > score_thresh
#         idxs = others[remain_mask]

#     return torch.tensor(keep, dtype=torch.long, device="cpu")


# # -----------------------------
# # vrt resolution (cached build)
# # -----------------------------

# def build_vrt_from_tifs(files, out_vrt):
#     try:
#         from osgeo import gdal
#     except ImportError:
#         raise RuntimeError("python gdal (osgeo.gdal) is not installed.")
#     gdal.UseExceptions()
#     os.makedirs(os.path.dirname(out_vrt) or ".", exist_ok=True)
#     vrt = gdal.BuildVRT(out_vrt, [os.path.abspath(f) for f in files])
#     if vrt is None:
#         raise RuntimeError("gdal.BuildVRT failed to create the vrt.")
#     vrt.FlushCache()
#     vrt = None


# def resolve_raster_input(path_or_dir):
#     """
#     if path_or_dir is a directory of .tif tiles or a *.tif glob:
#       → build a vrt one level up from the tiles folder (re-use if exists).
#     otherwise:
#       → return path_or_dir unchanged.
#     """
#     if os.path.isdir(path_or_dir):
#         files = sorted(glob.glob(os.path.join(path_or_dir, "*.tif")))
#         if not files:
#             raise FileNotFoundError(f"no .tif files found in {path_or_dir}")
#         parent = os.path.dirname(os.path.abspath(path_or_dir))
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
#         if not os.path.exists(out_vrt):
#             build_vrt_from_tifs(files, out_vrt)
#         return out_vrt
#     if any(ch in path_or_dir for ch in ["*", "?", "["]):
#         files = sorted(glob.glob(path_or_dir))
#         if not files:
#             raise FileNotFoundError(f"glob matched no .tif files: {path_or_dir}")
#         tile_dir = os.path.dirname(os.path.abspath(path_or_dir))
#         parent = os.path.dirname(tile_dir)
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
#         if not os.path.exists(out_vrt):
#             build_vrt_from_tifs(files, out_vrt)
#         return out_vrt
#     return path_or_dir


# # -------------------------
# # ddp helpers (init + rank)
# # -------------------------

# def is_main_process() -> bool:
#     return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# def init_distributed_if_needed():
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         backend = 'nccl' if torch.cuda.is_available() else 'gloo'
#         local_rank = int(os.environ.get('LOCAL_RANK', 0))
#         if torch.cuda.is_available():
#             torch.cuda.set_device(local_rank)
#         dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
#         # touch device once so nccl learns local mapping; quiets early barrier warnings
#         if torch.cuda.is_available():
#             _ = torch.zeros(1, device=f'cuda:{local_rank}')
#         return local_rank
#     return int(os.environ.get('LOCAL_RANK', 0))


# # ----------------
# # checkpoint meta
# # ----------------

# def load_classes_and_task_from_checkpoint(ckpt_path: str):
#     data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     if isinstance(data, dict):
#         classes = data.get('classes', None)
#         task = data.get('task', 'detection')
#         if classes is not None:
#             return classes, task
#     json_path = os.path.join(os.path.dirname(ckpt_path), 'classes.json')
#     if os.path.exists(json_path):
#         with open(json_path) as f:
#             return json.load(f)['classes'], 'detection'
#     raise ValueError('could not determine classes from checkpoint or sidecar json')


# # ----------------
# # polygon-level dedupe by jaccard (iou) per class, keep highest score
# # ----------------

# def dedupe_polygons_by_iou(gdf: gpd.GeoDataFrame,
#                            class_col: str = "classname",
#                            score_col: str = "score",
#                            iou_thresh: float = 0.5) -> gpd.GeoDataFrame:
#     # require valid polygons
#     gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].copy()
#     if len(gdf) <= 1:
#         return gdf

#     kept_idx = []
#     for cls, group in gdf.groupby(class_col, group_keys=False):
#         # descending by score, greedy keep
#         group = group.sort_values(score_col, ascending=False).copy()
#         sindex = group.sindex  # rtree spatial index
#         taken = np.zeros(len(group), dtype=bool)
#         group_idx = group.index.to_numpy()

#         for i_local, idx_i in enumerate(group_idx):
#             if taken[i_local]:
#                 continue
#             gi = group.loc[idx_i].geometry
#             kept_idx.append(idx_i)

#             # candidates: bbox intersects
#             cand_pos = list(sindex.intersection(gi.bounds))
#             for j_local in cand_pos:
#                 idx_j = group_idx[j_local]
#                 if idx_j == idx_i or taken[j_local]:
#                     continue
#                 gj = group.loc[idx_j].geometry
#                 inter_area = gi.intersection(gj).area
#                 if inter_area <= 0.0:
#                     continue
#                 union_area = gi.area + gj.area - inter_area
#                 iou = inter_area / union_area if union_area > 0 else 0.0
#                 if iou >= iou_thresh:
#                     taken[j_local] = True  # drop lower-scored duplicate

#     return gdf.loc[kept_idx].copy()




# # ------------
# # arg parsing
# # ------------

# def parse_args():
#     parser = argparse.ArgumentParser(description='sliding-window inference for boxes or instance masks (ddp + soft-nms + min-area, masks cpu)')
#     parser.add_argument('--task', type=str, default='detection', choices=['detection', 'instance_seg'])
#     parser.add_argument('--raster_path', type=str, required=True)
#     parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'])
#     parser.add_argument('--checkpoint', type=str, required=True)
#     parser.add_argument('--out_vector', type=str, default='detections.gpkg')
#     parser.add_argument('--tile_size', type=int, default=1024)
#     parser.add_argument('--stride', type=int, default=512)
#     parser.add_argument('--score_thresh', type=float, default=0.5)
#     parser.add_argument('--nms_iou_thresh', type=float, default=0.5)
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--infer_batch', type=int, default=4, help='windows per forward pass per rank')

#     # global merge + soft-nms controls
#     parser.add_argument('--global_nms', action='store_true',
#                         help='enable global (cross-rank) class-wise nms after per-rank nms (recommended when stride < tile_size)')
#     parser.add_argument('--soft_nms', action='store_true',
#                         help='use soft-nms in the global merge step (ignored if --global_nms not set)')
#     parser.add_argument('--soft_nms_method', type=str, default='gaussian', choices=['gaussian', 'linear'],
#                         help='soft-nms decay method')
#     parser.add_argument('--soft_nms_sigma', type=float, default=0.5,
#                         help='soft-nms gaussian sigma (typical 0.5)')
#     parser.add_argument('--soft_nms_score_thresh', type=float, default=0.001,
#                         help='drop boxes whose decayed score falls below this threshold during soft-nms')

#     # post-vectorization area filter (in crs units; e.g., m^2 in projected crs or ft^2 for EPSG:2968)
#     parser.add_argument('--min_polygon_area', type=float, default=0.0,
#                         help='drop polygons with area < this value (area is in raster crs units; use projected crs for meaningful m^2)')

#     # mask-based window filtering
#     parser.add_argument('--mask_path', type=str, default=None,
#                         help='optional polygon mask to restrict windows to areas of interest')
#     parser.add_argument('--min_cover_frac', type=float, default=0.0,
#                         help='minimum fraction of a window that must intersect the mask to keep it (0..1)')
#     parser.add_argument('--mask_mode', type=str, default='raster', choices=['raster', 'vector'],
#                         help="mask mode: 'raster' (fast; rasterize once) or 'vector' (polygon intersects)")

#     # final output masking controls
#     parser.add_argument('--require_mask_intersection', action='store_true',
#                         help='drop detections that do not intersect the mask (requires --mask_path)')
#     parser.add_argument('--clip_to_mask', action='store_true',
#                         help='clip detections to the mask geometry before writing (implies intersection; requires --mask_path)')

#     # final polygon-evel dedupe
#     parser.add_argument('--dedupe_after_vectorization', action='store_true',
#                         help='remove overlapping polygons per class using polygon iou (keeps highest score)')
#     parser.add_argument('--dedupe_iou', type=float, default=0.5,
#                         help='polygon iou threshold for dedupe (0..1)')


#     return parser.parse_args()


# # -------
# # main
# # -------

# def main():
#     args = parse_args()

#     # fast conv selection
#     cudnn.benchmark = True
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")

#     # ddp init (no-op for single process)
#     local_rank = init_distributed_if_needed()
#     world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
#     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

#     # resolve raster on rank 0, then barrier
#     if is_main_process():
#         args.raster_path = resolve_raster_input(args.raster_path)
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()
#         args.raster_path = resolve_raster_input(args.raster_path)

#     # logger
#     log_path = os.path.join(os.path.dirname(args.out_vector) or ".", "inference.log")
#     logger = setup_logging(log_path, rank)
#     t0 = time.time()
#     if is_main_process():
#         logger.info("[debug] starting inference (world_size=%d)", world_size)

#     # early checkpoint existence check (rank 0)
#     if is_main_process():
#         if not os.path.exists(args.checkpoint):
#             logger.error("[fatal] checkpoint not found: %s", args.checkpoint)
#             if dist.is_available() and dist.is_initialized():
#                 dist.barrier()
#             sys.exit(1)
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()

#     # model + classes
#     classes, task_from_ckpt = load_classes_and_task_from_checkpoint(args.checkpoint)
#     if args.task != task_from_ckpt:
#         logger.warning("[warn] overriding task from checkpoint (%s) with arg (%s)", task_from_ckpt, args.task)

#     num_classes = len(classes) + 1
#     device = args.device
#     if device == 'cuda' and torch.cuda.is_available():
#         device = f'cuda:{local_rank}'
#         torch.cuda.set_device(local_rank)

#     if args.task == 'instance_seg':
#         model = build_maskrcnn_model(num_classes=num_classes, pretrained=False, device=device)
#     else:
#         model = build_fasterrcnn_model(num_classes=num_classes, pretrained=False, device=device)

#     state = torch.load(args.checkpoint, map_location=device, weights_only=False)
#     model.load_state_dict(state["model_state"])
#     model.eval()
#     if is_main_process():
#         logger.info("[debug] model loaded and set to eval (device=%s)", device)


#     # build full grid once
#     windows_all = make_grid_windows(args.raster_path, tile_size=args.tile_size, stride=args.stride)
#     total_before = len(windows_all)
#     # optional: filter by polygon mask BEFORE sharding
#     if args.mask_path:
#         if args.mask_mode == "raster":
#             mask_raster = rasterize_mask_aligned(args.raster_path, args.mask_path)
#             windows_all = filter_windows_by_mask_raster(
#                 mask_raster, windows_all, min_cover_frac=max(0.0, float(args.min_cover_frac))
#             )
#         else:  # vector
#             windows_all = filter_windows_by_mask(
#                 raster_path=args.raster_path,
#                 windows=windows_all,
#                 mask_path=args.mask_path,
#                 min_cover_frac=max(0.0, float(args.min_cover_frac)),
#             )
#         if is_main_process():
#             logger.info("[debug] mask filtered windows=%d (from %d) [mode=%s, min_cover_frac=%.3f]",
#                         len(windows_all), total_before, args.mask_mode, float(args.min_cover_frac))
#     if is_main_process():
#         logger.info("[debug] total windows=%d, per-rank≈%d",
#                     len(windows_all), (len(windows_all) + world_size - 1) // world_size)
#     # shard AFTER filtering
#     windows = windows_all[rank::world_size]

#     # buffers (boxes/scores/labels on cpu lists; masks kept as cpu uint8 per detection)
#     all_boxes, all_scores, all_labels = [], [], []
#     tile_transforms = []          # align 1:1 with detections
#     masks_flat: list[torch.Tensor] = []  # each is uint8 cpu (h,w)

#     # sliding-window loop
#     with rasterio.open(args.raster_path) as src:
#         normalizer = build_normalizer(args.normalize)
#         use_cuda = device.startswith("cuda")
#         amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

#         # warn if crs is geographic since area filter would be in degrees^2
#         try:
#             if src.crs and src.crs.is_geographic:
#                 if is_main_process() and args.min_polygon_area > 0:
#                     logger.warning("[warn] raster crs is geographic; min_polygon_area uses degrees^2. consider reprojecting to a projected crs for meaningful m^2.")
#         except Exception:
#             pass

#         batch_imgs: list[torch.Tensor] = []
#         batch_transforms: list = []
#         batch_windows: list = []
#         batch_size = max(1, int(args.infer_batch))

#         def run_batch():
#             if not batch_imgs:
#                 return
#             with torch.inference_mode():
#                 with amp_ctx:
#                     preds_list = model(batch_imgs)
#             for preds, win, tform in zip(preds_list, batch_windows, batch_transforms):
#                 boxes = preds["boxes"]; scores = preds["scores"]; labels = preds["labels"]
#                 keep = scores >= args.score_thresh
#                 boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
#                 boxes_global = adjust_boxes_to_global(boxes, win)

#                 # append cpu tensors for boxes/scores/labels
#                 all_boxes.append(boxes_global.to("cpu"))
#                 all_scores.append(scores.to("cpu"))
#                 all_labels.append(labels.to("cpu"))

#                 # append transforms and masks per detection (uint8 cpu to save ram)
#                 if args.task == 'instance_seg' and 'masks' in preds:
#                     m = preds['masks'][keep].squeeze(1)  # (n,h,w) float on device
#                     if m.ndim == 2:  # single mask edge case
#                         m = m.unsqueeze(0)
#                     if m.shape[0] > 0:
#                         m_cpu = (m.to("cpu") > 0.5).to(torch.uint8)  # threshold + pack to uint8
#                         for k in range(m_cpu.shape[0]):
#                             masks_flat.append(m_cpu[k])
#                             tile_transforms.append(tform)
#                     # if no masks survived score threshold, nothing to append
#                 else:
#                     # detection-only: add tform per detection count
#                     for _ in range(boxes.shape[0]):
#                         tile_transforms.append(tform)

#             batch_imgs.clear(); batch_transforms.clear(); batch_windows.clear()

#         for i, window in enumerate(windows, start=1):
#             if is_main_process() and (i % 50 == 1 or i == len(windows)):
#                 logger.info("[debug] rank %d processing window %d/%d", rank, i, len(windows))
#             img = src.read(window=window)
#             tform = src.window_transform(window)
#             if img.shape[0] >= 3:
#                 img = img[:3]
#             else:
#                 repeats = (3 + img.shape[0] - 1) // img.shape[0]
#                 img = np.concatenate([img] * repeats, axis=0)[:3]
#             h, w = img.shape[-2:]
#             if h < 32 or w < 32:
#                 continue
#             if not np.isfinite(img).all():
#                 continue
#             scale = 65535.0 if src.dtypes[0] in ("uint16", "int16") else 255.0
#             tensor = torch.from_numpy(img).float().div_(scale).clamp_(0.0, 1.0)
#             if normalizer is not None:
#                 tensor = normalizer(tensor)
#             if use_cuda:
#                 tensor = tensor.pin_memory()
#             tensor = tensor.to(device, non_blocking=True)

#             batch_imgs.append(tensor)
#             batch_transforms.append(tform)
#             batch_windows.append(window)
#             if len(batch_imgs) >= batch_size:
#                 run_batch()
#         run_batch()

#     # concat boxes/scores/labels and run per-rank hard nms on device
#     boxes = torch.cat(all_boxes, dim=0).to(device) if all_boxes else torch.empty((0, 4), device=device)
#     scores = torch.cat(all_scores, dim=0).to(device) if all_scores else torch.empty((0,), device=device)
#     labels = torch.cat(all_labels, dim=0).to(device) if all_labels else torch.empty((0,), dtype=torch.int64, device=device)

#     keep_indices_all: list[int] = []
#     if len(boxes) > 0:
#         for cls_idx in labels.unique():
#             cls_mask = labels == cls_idx
#             if cls_mask.any():
#                 kept = nms(boxes[cls_mask], scores[cls_mask], args.nms_iou_thresh)
#                 base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
#                 keep_indices_all.extend(base_idx[kept].tolist())

#     if keep_indices_all:
#         keep_t_gpu = torch.tensor(keep_indices_all, dtype=torch.long, device=device)
#         boxes = boxes[keep_t_gpu]
#         scores = scores[keep_t_gpu]
#         labels = labels[keep_t_gpu]
#     else:
#         boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]

#     # move kept boxes/scores/labels to cpu; select corresponding masks/transforms on cpu
#     boxes = boxes.to("cpu")
#     scores = scores.to("cpu")
#     labels = labels.to("cpu")

#     # tile_transforms aligns 1:1 with detections appended; masks_flat too (for instance_seg)
#     if keep_indices_all:
#         keep_indices_cpu = torch.tensor(keep_indices_all, dtype=torch.long).tolist()
#         if args.task == 'instance_seg':
#             kept_masks_flat = [masks_flat[i] for i in keep_indices_cpu]
#         else:
#             kept_masks_flat = []
#         kept_tile_transforms = [tile_transforms[i] for i in keep_indices_cpu]
#     else:
#         kept_masks_flat = []
#         kept_tile_transforms = []

#     # -----------------------------
#     # optional: global (cross-rank) nms (soft or hard) implemented via file-based exchange
#     # -----------------------------
#     if args.global_nms and dist.is_available() and dist.is_initialized() and world_size > 1:
#         # write per-rank metadata (no masks) to disk to avoid massive object collectives
#         out_dir = os.path.dirname(args.out_vector) or "."
#         os.makedirs(out_dir, exist_ok=True)
#         meta_path = os.path.join(out_dir, f"global_meta_rank{rank}.npz")
#         np.savez_compressed(
#             meta_path,
#             boxes=boxes.numpy(),        # (n, 4)
#             scores=scores.numpy(),      # (n,)
#             labels=labels.numpy(),      # (n,)
#         )

#         # sync so rank 0 can read all meta
#         dist.barrier()

#         if is_main_process():
#             # stitch all meta into one set
#             per_rank_counts = []
#             per_rank_boxes, per_rank_scores, per_rank_labels = [], [], []
#             for r in range(world_size):
#                 p = os.path.join(out_dir, f"global_meta_rank{r}.npz")
#                 if not os.path.exists(p):
#                     per_rank_counts.append(0)
#                     per_rank_boxes.append(torch.empty((0, 4)))
#                     per_rank_scores.append(torch.empty((0,)))
#                     per_rank_labels.append(torch.empty((0,), dtype=torch.int64))
#                     continue
#                 try:
#                     data = np.load(p)
#                     b = torch.from_numpy(data["boxes"]).float()
#                     s = torch.from_numpy(data["scores"]).float()
#                     l = torch.from_numpy(data["labels"]).long()
#                 except Exception:
#                     # if read fails, skip gracefully
#                     b = torch.empty((0, 4)); s = torch.empty((0,)); l = torch.empty((0,), dtype=torch.int64)
#                 per_rank_counts.append(b.shape[0])
#                 per_rank_boxes.append(b); per_rank_scores.append(s); per_rank_labels.append(l)

#             if sum(per_rank_counts) > 0:
#                 g_boxes  = torch.cat(per_rank_boxes,  dim=0)
#                 g_scores = torch.cat(per_rank_scores, dim=0)
#                 g_labels = torch.cat(per_rank_labels, dim=0)

#                 # global per-class nms (soft or hard) on cuda:0 if available
#                 nms_device = "cuda:0" if torch.cuda.is_available() else "cpu"
#                 g_boxes_d, g_scores_d, g_labels_d = g_boxes.to(nms_device), g_scores.to(nms_device), g_labels.to(nms_device)

#                 keep_all: list[int] = []
#                 for cls_idx in g_labels_d.unique():
#                     cls_mask = g_labels_d == cls_idx
#                     if not cls_mask.any():
#                         continue
#                     if args.soft_nms:
#                         kept_local = soft_nms(
#                             boxes=g_boxes_d[cls_mask].contiguous(),
#                             scores=g_scores_d[cls_mask].contiguous(),
#                             iou_thresh=args.nms_iou_thresh,
#                             sigma=args.soft_nms_sigma,
#                             score_thresh=args.soft_nms_score_thresh,
#                             method=args.soft_nms_method,
#                         )
#                         base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1).cpu()
#                         keep_all.extend(base_idx[kept_local].tolist())
#                     else:
#                         kept = nms(g_boxes_d[cls_mask], g_scores_d[cls_mask], args.nms_iou_thresh)
#                         base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1).cpu()
#                         keep_all.extend(base_idx[kept.cpu()].tolist())

#                 # split global keep indices back into per-rank indices (relative to each rank)
#                 per_rank_keeps = [[] for _ in range(world_size)]
#                 offsets = np.cumsum([0] + per_rank_counts[:-1]).tolist()
#                 for gi in keep_all:
#                     # find rank for this global index
#                     # linear scan is fine; if counts are huge, switch to bisect
#                     for r, offset in enumerate(offsets):
#                         if gi < offset + per_rank_counts[r]:
#                             local_i = gi - offset
#                             per_rank_keeps[r].append(int(local_i))
#                             break

#                 # write keep lists to disk (json) for each rank
#                 for r in range(world_size):
#                     with open(os.path.join(out_dir, f"global_keep_rank{r}.json"), "w") as f:
#                         json.dump(per_rank_keeps[r], f)
#             else:
#                 # no detections: write empty keep lists
#                 for r in range(world_size):
#                     with open(os.path.join(out_dir, f"global_keep_rank{r}.json"), "w") as f:
#                         json.dump([], f)

#         # barrier so everyone waits for keep lists
#         dist.barrier()

#         # each rank loads its keep list and filters local arrays before vectorization
#         out_dir = os.path.dirname(args.out_vector) or "."
#         keep_path = os.path.join(out_dir, f"global_keep_rank{rank}.json")
#         if os.path.exists(keep_path):
#             with open(keep_path, "r") as f:
#                 local_keep_list = json.load(f)
#         else:
#             local_keep_list = []

#         if len(local_keep_list) > 0:
#             keep_t = torch.tensor(local_keep_list, dtype=torch.long)
#             boxes  = boxes.index_select(0, keep_t)
#             scores = scores.index_select(0, keep_t)
#             labels = labels.index_select(0, keep_t)
#             # filter masks/transforms by the same kept indices
#             if args.task == 'instance_seg':
#                 kept_masks_flat       = [kept_masks_flat[i] for i in local_keep_list]
#             kept_tile_transforms = [kept_tile_transforms[i] for i in local_keep_list]
#         else:
#             boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]
#             kept_masks_flat = [] if args.task == 'instance_seg' else []
#             kept_tile_transforms = []

#     # open raster once for crs/transform (global)
#     with rasterio.open(args.raster_path) as src:
#         crs = src.crs
#         global_transform = src.transform

#     # vectorize to polygons (build geoms + attrs)
#     geoms, class_names, score_vals = [], [], []
#     if args.task == 'instance_seg' and boxes.numel() > 0 and len(kept_masks_flat) > 0:
#         # iterate masks (uint8 cpu) with corresponding transforms and attrs
#         for m_u8, tform, s, l in zip(kept_masks_flat, kept_tile_transforms, scores.tolist(), labels.tolist()):
#             m_np = m_u8.numpy().astype('uint8')
#             for geom, val in rio_features.shapes(m_np, transform=tform):
#                 if int(val) == 0:
#                     continue
#                 from shapely.geometry import shape as shapely_shape
#                 geoms.append(shapely_shape(geom))
#                 class_names.append(classes[l - 1])
#                 score_vals.append(float(s))
#     else:
#         for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
#             xmin, ymin, xmax, ymax = b
#             x0, y0 = global_transform * (xmin, ymin)
#             x1, y1 = global_transform * (xmax, ymax)
#             geom = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
#             geoms.append(geom)
#             class_names.append(classes[l - 1])
#             score_vals.append(float(s))

#     gdf = gpd.GeoDataFrame({"classname": class_names, "score": score_vals}, geometry=geoms, crs=crs)


#     # enforce final mask intersection or clipping (optional)
#     if args.mask_path and (args.require_mask_intersection or args.clip_to_mask):
#         mask_gdf = gpd.read_file(args.mask_path)
#         if mask_gdf.crs != gdf.crs:
#             mask_gdf = mask_gdf.to_crs(gdf.crs)
#         if args.clip_to_mask:
#             # clip to keep only the portion inside the mask
#             gdf = gpd.overlay(gdf, mask_gdf[["geometry"]], how="intersection")
#         else:
#             # just require that each feature intersects the mask
#             mask_union = mask_gdf.unary_union
#             gdf = gdf[gdf.geometry.intersects(mask_union)].copy()



#     # polygon-level dedupe (optional)
#     if args.dedupe_after_vectorization and len(gdf) > 1:
#         gdf = dedupe_polygons_by_iou(gdf, class_col="classname", score_col="score",
#                                     iou_thresh=float(args.dedupe_iou))


#     # post-vectorization area filter
#     pre_count = len(gdf)
#     if args.min_polygon_area > 0 and pre_count > 0:
#         gdf = gdf[gdf.geometry.area >= args.min_polygon_area].copy()
#     post_count = len(gdf)

#     out_dir = os.path.dirname(args.out_vector) or "."
#     os.makedirs(out_dir, exist_ok=True)

#     # output: if global_nms → write per-rank partial then rank 0 merges (already deduped globally)
#     # otherwise (no global_nms) → still write partials + merge (may contain duplicates if stride < tile_size)
#     layer_name = "detections"
#     base, ext = os.path.splitext(args.out_vector)
#     partial_path = f"{base}_rank{rank}{ext if ext.lower()=='.gpkg' else '.gpkg'}"
#     gdf.to_file(partial_path, driver="GPKG", layer=layer_name)
#     kept = post_count
#     dropped = pre_count - post_count
#     logger.info("[summary] rank %d wrote %d detections to %s (dropped %d < min_area=%.3f)",
#                 rank, kept, partial_path, dropped, args.min_polygon_area)

#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()

#     if is_main_process():
#         import pandas as pd
#         parts = [f"{base}_rank{r}{ext if ext.lower()=='.gpkg' else '.gpkg'}" for r in range(world_size)]
#         gdfs = []
#         for p in parts:
#             if os.path.exists(p):
#                 try:
#                     gdfs.append(gpd.read_file(p, layer=layer_name))
#                 except Exception:
#                     logger.exception("[warn] failed reading %s; skipping", p)
#         if len(gdfs) > 0:
#             merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs if gdfs[0].crs else crs)
#         else:
#             merged = gpd.GeoDataFrame({"classname": [], "score": []}, geometry=[], crs=crs)
#         final_path = args.out_vector if args.out_vector.lower().endswith('.gpkg') else base + '.gpkg'
#         merged.to_file(final_path, driver="GPKG", layer=layer_name)
#         logger.info("[done] merged %d partials → %s (elapsed %.2fs)", len(gdfs), final_path, time.time() - t0)

#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()
#         dist.destroy_process_group()


# if __name__ == '__main__':
#     main()











# BACKUP 2 - OOM error

# from __future__ import annotations
# import os
# import json
# import argparse
# import time
# from datetime import timedelta
# import sys
# import glob
# import logging
# from contextlib import nullcontext

# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.backends.cudnn as cudnn
# from torchvision.ops import nms
# from torchvision import transforms
# import geopandas as gpd
# from shapely.geometry import box as shapely_box
# import rasterio
# from rasterio import features as rio_features
# from affine import Affine  # for reconstructing tile transforms when gathered

# # add project root to path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
# from src.utils.tiling import make_grid_windows, adjust_boxes_to_global


# def setup_logging(log_path: str, rank: int):
#     os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
#     logger = logging.getLogger(f"inference_rank{rank}")
#     logger.setLevel(logging.INFO)
#     logger.handlers.clear()
#     fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
#     sh = logging.StreamHandler(sys.stdout if rank == 0 else open(os.devnull, 'w'))
#     sh.setFormatter(fmt)
#     fh = logging.FileHandler(log_path.replace(".log", f"_rank{rank}.log"), mode="w", encoding="utf-8")
#     fh.setFormatter(fmt)
#     logger.addHandler(sh)
#     logger.addHandler(fh)
#     return logger


# def build_normalizer(norm_type: str):
#     if norm_type.lower() == "imagenet":
#         return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     return None


# def parse_args():
#     parser = argparse.ArgumentParser(description='sliding-window inference for boxes or instance masks (ddp-ready)')
#     parser.add_argument('--task', type=str, default='detection', choices=['detection', 'instance_seg'])
#     parser.add_argument('--raster_path', type=str, required=True)
#     parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'])
#     parser.add_argument('--checkpoint', type=str, required=True)
#     parser.add_argument('--out_vector', type=str, default='detections.gpkg')
#     parser.add_argument('--tile_size', type=int, default=1024)
#     parser.add_argument('--stride', type=int, default=512)
#     parser.add_argument('--score_thresh', type=float, default=0.5)
#     parser.add_argument('--nms_iou_thresh', type=float, default=0.5)
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--infer_batch', type=int, default=4, help='windows per forward pass per rank')
#     parser.add_argument('--global_nms', action='store_true', help='gather detections to rank 0 and run a second global class-wise nms (recommended when stride < tile_size)')
#     return parser.parse_args()


# def build_vrt_from_tifs(files, out_vrt):
#     try:
#         from osgeo import gdal
#     except ImportError:
#         raise RuntimeError("python gdal (osgeo.gdal) is not installed.")
#     gdal.UseExceptions()
#     os.makedirs(os.path.dirname(out_vrt) or ".", exist_ok=True)
#     vrt = gdal.BuildVRT(out_vrt, [os.path.abspath(f) for f in files])
#     if vrt is None:
#         raise RuntimeError("gdal.BuildVRT failed to create the vrt.")
#     vrt.FlushCache()
#     vrt = None


# def resolve_raster_input(path_or_dir):
#     if os.path.isdir(path_or_dir):
#         files = sorted(glob.glob(os.path.join(path_or_dir, "*.tif")))
#         if not files:
#             raise FileNotFoundError(f"no .tif files found in {path_or_dir}")
#         parent = os.path.dirname(os.path.abspath(path_or_dir))
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
#         if not os.path.exists(out_vrt):
#             build_vrt_from_tifs(files, out_vrt)
#         return out_vrt
#     if any(ch in path_or_dir for ch in ["*", "?", "["]):
#         files = sorted(glob.glob(path_or_dir))
#         if not files:
#             raise FileNotFoundError(f"glob matched no .tif files: {path_or_dir}")
#         tile_dir = os.path.dirname(os.path.abspath(path_or_dir))
#         parent = os.path.dirname(tile_dir)
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
#         if not os.path.exists(out_vrt):
#             build_vrt_from_tifs(files, out_vrt)
#         return out_vrt
#     return path_or_dir


# def is_main_process() -> bool:
#     return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# def init_distributed_if_needed():
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         backend = 'nccl' if torch.cuda.is_available() else 'gloo'
#         local_rank = int(os.environ.get('LOCAL_RANK', 0))
#         if torch.cuda.is_available():
#             torch.cuda.set_device(local_rank)
#         dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
#         # touch device once so nccl knows mapping
#         if torch.cuda.is_available():
#             _ = torch.zeros(1, device=f'cuda:{local_rank}')
#         return local_rank
#     return int(os.environ.get('LOCAL_RANK', 0))


# def load_classes_and_task_from_checkpoint(ckpt_path: str):
#     data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     if isinstance(data, dict):
#         classes = data.get('classes', None)
#         task = data.get('task', 'detection')
#         if classes is not None:
#             return classes, task
#     json_path = os.path.join(os.path.dirname(ckpt_path), 'classes.json')
#     if os.path.exists(json_path):
#         with open(json_path) as f:
#             return json.load(f)['classes'], 'detection'
#     raise ValueError('could not determine classes from checkpoint or sidecar json')


# def main():
#     args = parse_args()

#     cudnn.benchmark = True
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")

#     local_rank = init_distributed_if_needed()
#     world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
#     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

#     # resolve raster on rank 0 then barrier
#     if is_main_process():
#         args.raster_path = resolve_raster_input(args.raster_path)
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()
#         args.raster_path = resolve_raster_input(args.raster_path)

#     # logger
#     log_path = os.path.join(os.path.dirname(args.out_vector) or ".", "inference.log")
#     logger = setup_logging(log_path, rank)
#     t0 = time.time()
#     if is_main_process():
#         logger.info("[debug] starting inference (world_size=%d)", world_size)

#     # early checkpoint existence check (rank 0)
#     if is_main_process():
#         if not os.path.exists(args.checkpoint):
#             logger.error("[fatal] checkpoint not found: %s", args.checkpoint)
#             if dist.is_available() and dist.is_initialized():
#                 dist.barrier()
#             sys.exit(1)
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()

#     # model + classes
#     classes, task_from_ckpt = load_classes_and_task_from_checkpoint(args.checkpoint)
#     if args.task != task_from_ckpt:
#         logger.warning("[warn] overriding task from checkpoint (%s) with arg (%s)", task_from_ckpt, args.task)

#     num_classes = len(classes) + 1
#     device = args.device
#     if device == 'cuda' and torch.cuda.is_available():
#         device = f'cuda:{local_rank}'
#         torch.cuda.set_device(local_rank)

#     if args.task == 'instance_seg':
#         model = build_maskrcnn_model(num_classes=num_classes, pretrained=False, device=device)
#     else:
#         model = build_fasterrcnn_model(num_classes=num_classes, pretrained=False, device=device)

#     state = torch.load(args.checkpoint, map_location=device, weights_only=False)
#     model.load_state_dict(state["model_state"])
#     model.eval()
#     if is_main_process():
#         logger.info("[debug] model loaded and set to eval (device=%s)", device)

#     # grid → shard across ranks (round-robin)
#     windows_all = make_grid_windows(args.raster_path, tile_size=args.tile_size, stride=args.stride)
#     if is_main_process():
#         logger.info("[debug] total windows=%d, per-rank≈%d", len(windows_all), (len(windows_all) + world_size - 1) // world_size)
#     windows = windows_all[rank::world_size]

#     all_boxes, all_scores, all_labels = [], [], []
#     tile_transforms = []
#     mask_slices = []

#     with rasterio.open(args.raster_path) as src:
#         normalizer = build_normalizer(args.normalize)
#         use_cuda = device.startswith("cuda")
#         amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

#         batch_imgs: list[torch.Tensor] = []
#         batch_transforms: list = []
#         batch_windows: list = []
#         batch_size = max(1, int(args.infer_batch))

#         def run_batch():
#             if not batch_imgs:
#                 return
#             with torch.inference_mode():
#                 with amp_ctx:
#                     preds_list = model(batch_imgs)
#             for preds, win, tform in zip(preds_list, batch_windows, batch_transforms):
#                 boxes = preds["boxes"]; scores = preds["scores"]; labels = preds["labels"]
#                 keep = scores >= args.score_thresh
#                 boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
#                 boxes_global = adjust_boxes_to_global(boxes, win)
#                 all_boxes.append(boxes_global)
#                 all_scores.append(scores)
#                 all_labels.append(labels)
#                 if args.task == 'instance_seg' and 'masks' in preds:
#                     m = preds['masks'][keep]
#                     if m.shape[0] > 0:
#                         mask_slices.append(m)
#                         tile_transforms.extend([tform] * m.shape[0])
#                 else:
#                     if boxes.shape[0] > 0:
#                         tile_transforms.extend([tform] * boxes.shape[0])
#             batch_imgs.clear(); batch_transforms.clear(); batch_windows.clear()

#         for i, window in enumerate(windows, start=1):
#             if is_main_process() and (i % 50 == 1 or i == len(windows)):
#                 logger.info("[debug] rank %d processing window %d/%d", rank, i, len(windows))
#             img = src.read(window=window)
#             tform = src.window_transform(window)
#             if img.shape[0] >= 3:
#                 img = img[:3]
#             else:
#                 repeats = (3 + img.shape[0] - 1) // img.shape[0]
#                 img = np.concatenate([img] * repeats, axis=0)[:3]
#             h, w = img.shape[-2:]
#             if h < 32 or w < 32:
#                 continue
#             if not np.isfinite(img).all():
#                 continue
#             scale = 65535.0 if src.dtypes[0] in ("uint16", "int16") else 255.0
#             tensor = torch.from_numpy(img).float().div_(scale).clamp_(0.0, 1.0)
#             if normalizer is not None:
#                 tensor = normalizer(tensor)
#             if use_cuda:
#                 tensor = tensor.pin_memory()
#             tensor = tensor.to(device, non_blocking=True)

#             batch_imgs.append(tensor)
#             batch_transforms.append(tform)
#             batch_windows.append(window)
#             if len(batch_imgs) >= batch_size:
#                 run_batch()
#         run_batch()

#     # concat and per-rank nms on device (first pass)
#     boxes = torch.cat(all_boxes, dim=0).to(device) if all_boxes else torch.empty((0, 4), device=device)
#     scores = torch.cat(all_scores, dim=0).to(device) if all_scores else torch.empty((0,), device=device)
#     labels = torch.cat(all_labels, dim=0).to(device) if all_labels else torch.empty((0,), dtype=torch.int64, device=device)
#     masks = torch.cat(mask_slices, dim=0).to(device) if mask_slices else torch.empty((0, 1, 1, 1), device=device)

#     if len(boxes) > 0:
#         keep_indices: list[int] = []
#         for cls_idx in labels.unique():
#             cls_mask = labels == cls_idx
#             if cls_mask.any():
#                 k = nms(boxes[cls_mask], scores[cls_mask], args.nms_iou_thresh)
#                 base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
#                 keep_indices.extend(base_idx[k].tolist())
#         if keep_indices:
#             keep_indices_t = torch.tensor(keep_indices, dtype=torch.long, device=device)
#             boxes = boxes[keep_indices_t]; scores = scores[keep_indices_t]; labels = labels[keep_indices_t]
#             if args.task == 'instance_seg' and masks.numel() > 0:
#                 masks = masks[keep_indices_t]
#                 tile_transforms = [tile_transforms[i] for i in keep_indices]
#         else:
#             boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]; masks = masks[:0]; tile_transforms = []

#     # move to cpu to reduce gpu mem usage before potential gather
#     boxes = boxes.to("cpu"); scores = scores.to("cpu"); labels = labels.to("cpu"); masks = masks.to("cpu")

#     # if requested, gather all detections to rank 0 and apply a second global nms
#     if args.global_nms and dist.is_available() and dist.is_initialized() and world_size > 1:
#         # serialize tile transforms to tuples for safe pickling
#         tforms_serialized = [tuple(t) for t in tile_transforms]
#         payload = {
#             "boxes": boxes,               # tensor cpu [N, 4] in global pixel coords
#             "scores": scores,             # tensor cpu [N]
#             "labels": labels,             # tensor cpu [N]
#             "has_masks": (args.task == 'instance_seg' and masks.numel() > 0),
#             "masks": masks if (args.task == 'instance_seg' and masks.numel() > 0) else None,  # tensor cpu [N,1,h,w]
#             "tforms": tforms_serialized,  # list of 6-tuples
#         }
#         gathered: list | None = [None for _ in range(world_size)] if is_main_process() else None
#         dist.gather_object(payload, gather_list=gathered, dst=0)

#         if is_main_process():
#             # stitch all payloads
#             boxes_list, scores_list, labels_list = [], [], []
#             masks_list, tforms_list = [], []
#             for p in gathered:
#                 if p is None:  # just in case
#                     continue
#                 if p["boxes"] is not None and p["boxes"].numel() > 0:
#                     boxes_list.append(p["boxes"])
#                     scores_list.append(p["scores"])
#                     labels_list.append(p["labels"])
#                     if p["has_masks"] and p["masks"] is not None and p["masks"].numel() > 0:
#                         masks_list.append(p["masks"])
#                         tforms_list.extend(p["tforms"])
#                     else:
#                         # still need to extend transforms even if masks absent; we stored per-detection tforms earlier
#                         tforms_list.extend(p["tforms"])

#             if len(boxes_list) > 0:
#                 g_boxes = torch.cat(boxes_list, dim=0)
#                 g_scores = torch.cat(scores_list, dim=0)
#                 g_labels = torch.cat(labels_list, dim=0)
#                 g_masks = torch.cat(masks_list, dim=0) if len(masks_list) > 0 else None
#                 # run global class-wise nms on cpu or cuda:0 for speed
#                 nms_device = "cuda:0" if torch.cuda.is_available() else "cpu"
#                 g_boxes_d = g_boxes.to(nms_device)
#                 g_scores_d = g_scores.to(nms_device)
#                 g_labels_d = g_labels.to(nms_device)

#                 keep_all: list[int] = []
#                 for cls_idx in g_labels_d.unique():
#                     cls_mask = g_labels_d == cls_idx
#                     if cls_mask.any():
#                         k = nms(g_boxes_d[cls_mask], g_scores_d[cls_mask], args.nms_iou_thresh)
#                         base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
#                         keep_all.extend(base_idx[k].tolist())

#                 if len(keep_all) > 0:
#                     keep_t = torch.tensor(keep_all, dtype=torch.long, device="cpu")
#                     boxes = g_boxes.index_select(0, keep_t).contiguous()
#                     scores = g_scores.index_select(0, keep_t).contiguous()
#                     labels = g_labels.index_select(0, keep_t).contiguous()
#                     # reorder transforms and masks if present
#                     tile_transforms = [Affine(*tforms_list[i]) for i in keep_all]
#                     if g_masks is not None:
#                         masks = g_masks.index_select(0, keep_t).contiguous()
#                     else:
#                         masks = torch.empty((0, 1, 1, 1))
#                 else:
#                     boxes = torch.empty((0, 4))
#                     scores = torch.empty((0,))
#                     labels = torch.empty((0,), dtype=torch.int64)
#                     masks = torch.empty((0, 1, 1, 1))
#                     tile_transforms = []

#         # make sure ranks wait for rank 0 to finish nms
#         dist.barrier()

#         # non-zero ranks exit early; rank 0 continues to vectorization
#         if not is_main_process():
#             dist.destroy_process_group()
#             return

#     # open raster once for crs/transform
#     with rasterio.open(args.raster_path) as src:
#         crs = src.crs
#         global_transform = src.transform

#     # vectorize
#     geoms, class_names, score_vals = [], [], []
#     if args.task == 'instance_seg' and boxes.numel() > 0 and masks.numel() > 0:
#         masks_np = (masks.squeeze(1).numpy() > 0.5).astype('uint8')
#         for m, tform, s, l in zip(masks_np, tile_transforms, scores.tolist(), labels.tolist()):
#             for geom, val in rio_features.shapes(m, transform=tform):
#                 if int(val) == 0:
#                     continue
#                 from shapely.geometry import shape as shapely_shape
#                 geoms.append(shapely_shape(geom))
#                 class_names.append(classes[l - 1])
#                 score_vals.append(float(s))
#     else:
#         for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
#             xmin, ymin, xmax, ymax = b
#             x0, y0 = global_transform * (xmin, ymin)
#             x1, y1 = global_transform * (xmax, ymax)
#             geom = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
#             geoms.append(geom)
#             class_names.append(classes[l - 1])
#             score_vals.append(float(s))

#     gdf = gpd.GeoDataFrame({"classname": class_names, "score": score_vals}, geometry=geoms, crs=crs)

#     out_dir = os.path.dirname(args.out_vector) or "."
#     os.makedirs(out_dir, exist_ok=True)

#     # when global_nms is enabled, only rank 0 reaches here; otherwise each rank writes a partial and merges
#     if args.global_nms and is_main_process():
#         final_path = args.out_vector if args.out_vector.lower().endswith('.gpkg') else os.path.splitext(args.out_vector)[0] + '.gpkg'
#         gdf.to_file(final_path, driver="GPKG", layer="detections")
#         logger.info("[done] global nms → wrote %d detections to %s (elapsed %.2fs)", len(gdf), final_path, time.time() - t0)
#     elif not args.global_nms:
#         # legacy multi-rank partials + merge
#         layer_name = "detections"
#         base, ext = os.path.splitext(args.out_vector)
#         partial_path = f"{base}_rank{rank}{ext if ext.lower()=='.gpkg' else '.gpkg'}"
#         gdf.to_file(partial_path, driver="GPKG", layer=layer_name)
#         logger.info("[summary] rank %d wrote %d detections to %s", rank, len(gdf), partial_path)

#         if dist.is_available() and dist.is_initialized():
#             dist.barrier()
#         if is_main_process():
#             import pandas as pd
#             parts = [f"{base}_rank{r}{ext if ext.lower()=='.gpkg' else '.gpkg'}" for r in range(world_size)]
#             gdfs = []
#             for p in parts:
#                 if os.path.exists(p):
#                     try:
#                         gdfs.append(gpd.read_file(p, layer=layer_name))
#                     except Exception:
#                         logger.exception("[warn] failed reading %s; skipping", p)
#             if len(gdfs) > 0:
#                 merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs if gdfs[0].crs else crs)
#             else:
#                 merged = gpd.GeoDataFrame({"classname": [], "score": []}, geometry=[], crs=crs)
#             final_path = args.out_vector if args.out_vector.lower().endswith('.gpkg') else base + '.gpkg'
#             merged.to_file(final_path, driver="GPKG", layer=layer_name)
#             logger.info("[done] merged %d partials → %s (elapsed %.2fs)", len(gdfs), final_path, time.time() - t0)

#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()
#         dist.destroy_process_group()


# if __name__ == '__main__':
#     main()









# BACKUP WORKING

# from __future__ import annotations
# import os
# import json
# import argparse
# import torch
# import torch.backends.cudnn as cudnn
# from contextlib import nullcontext
# import numpy as np
# import geopandas as gpd
# from shapely.geometry import box as shapely_box
# import rasterio
# from rasterio import features as rio_features
# from torchvision.ops import nms
# from torchvision import transforms
# import sys
# import logging
# import time
# import glob

# # add project root to path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
# from src.utils.tiling import make_grid_windows, adjust_boxes_to_global


# def setup_logging(log_path: str):
#     os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
#     logger = logging.getLogger("inference")
#     logger.setLevel(logging.INFO)
#     logger.handlers.clear()
#     fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
#     sh = logging.StreamHandler(sys.stdout)
#     sh.setFormatter(fmt)
#     fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
#     fh.setFormatter(fmt)
#     logger.addHandler(sh)
#     logger.addHandler(fh)
#     return logger


# def build_normalizer(norm_type: str):
#     if norm_type.lower() == "imagenet":
#         return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     return None


# def parse_args():
#     parser = argparse.ArgumentParser(description='sliding-window inference for boxes or instance masks')
#     parser.add_argument('--task', type=str, default='detection', choices=['detection', 'instance_seg'])
#     parser.add_argument('--raster_path', type=str, required=True)
#     parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'])
#     parser.add_argument('--checkpoint', type=str, required=True)
#     parser.add_argument('--out_vector', type=str, default='detections.gpkg')
#     parser.add_argument('--tile_size', type=int, default=1024)
#     parser.add_argument('--stride', type=int, default=512)
#     parser.add_argument('--score_thresh', type=float, default=0.5)
#     parser.add_argument('--nms_iou_thresh', type=float, default=0.5)
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     args = parser.parse_args()
#     return args


# def build_vrt_from_tifs(files, out_vrt):
#     # build a vrt using python gdal only
#     try:
#         from osgeo import gdal
#     except ImportError:
#         raise RuntimeError("python gdal (osgeo.gdal) is not installed.")

#     gdal.UseExceptions()
#     os.makedirs(os.path.dirname(out_vrt) or ".", exist_ok=True)

#     vrt = gdal.BuildVRT(out_vrt, [os.path.abspath(f) for f in files])
#     if vrt is None:
#         raise RuntimeError("gdal.BuildVRT failed to create the vrt.")
#     vrt.FlushCache()
#     vrt = None


# def resolve_raster_input(path_or_dir):
#     """
#     if path_or_dir is a directory of .tif tiles or a *.tif glob:
#       → build a vrt one level up from the tiles folder (re-use if exists).
#     otherwise:
#       → return path_or_dir unchanged.
#     """
#     # directory of tifs
#     if os.path.isdir(path_or_dir):
#         files = sorted(glob.glob(os.path.join(path_or_dir, "*.tif")))
#         if not files:
#             raise FileNotFoundError(f"no .tif files found in {path_or_dir}")

#         parent = os.path.dirname(os.path.abspath(path_or_dir))
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")

#         if not os.path.exists(out_vrt):
#             build_vrt_from_tifs(files, out_vrt)
#         return out_vrt

#     # glob (*.tif)
#     if any(ch in path_or_dir for ch in ["*", "?", "["]):
#         files = sorted(glob.glob(path_or_dir))
#         if not files:
#             raise FileNotFoundError(f"glob matched no .tif files: {path_or_dir}")

#         tile_dir = os.path.dirname(os.path.abspath(path_or_dir))
#         parent = os.path.dirname(tile_dir)
#         parent_name = os.path.basename(parent)
#         out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")

#         if not os.path.exists(out_vrt):
#             build_vrt_from_tifs(files, out_vrt)
#         return out_vrt

#     # passthrough
#     return path_or_dir


# def load_classes_and_task_from_checkpoint(ckpt_path: str):
#     data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     if isinstance(data, dict):
#         classes = data.get('classes', None)
#         task = data.get('task', 'detection')
#         if classes is not None:
#             return classes, task
#     # fallback to json next to checkpoint
#     json_path = os.path.join(os.path.dirname(ckpt_path), 'classes.json')
#     if os.path.exists(json_path):
#         with open(json_path) as f:
#             return json.load(f)['classes'], 'detection'
#     raise ValueError('could not determine classes from checkpoint or sidecar json')


# def main():
#     args = parse_args()

#     # enable cudnn autotune and prefer fast matmul where supported
#     cudnn.benchmark = True
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")

#     # convert tiles to vrt if needed
#     args.raster_path = resolve_raster_input(args.raster_path)

#     log_path = os.path.join(os.path.dirname(args.out_vector) or ".", "inference.log")
#     logger = setup_logging(log_path)
#     t0 = time.time()
#     logger.info("[debug] starting inference")

#     classes, task_from_ckpt = load_classes_and_task_from_checkpoint(args.checkpoint)
#     # allow overriding by arg, but warn if mismatch
#     if args.task != task_from_ckpt:
#         logger.warning("[warn] overriding task from checkpoint (%s) with arg (%s)", task_from_ckpt, args.task)

#     num_classes = len(classes) + 1
#     if args.task == 'instance_seg':
#         model = build_maskrcnn_model(num_classes=num_classes, pretrained=False, device=args.device)
#     else:
#         model = build_fasterrcnn_model(num_classes=num_classes, pretrained=False, device=args.device)

#     state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
#     model.load_state_dict(state["model_state"])
#     model.eval()
#     logger.info("[debug] model loaded and set to eval (device=%s)", args.device)

#     windows = make_grid_windows(args.raster_path, tile_size=args.tile_size, stride=args.stride)
#     logger.info("[debug] num_windows=%d", len(windows))

#     all_boxes, all_scores, all_labels = [], [], []
#     tile_transforms = []  # keep per-detection tile transform for correct vectorization
#     mask_slices = []      # store masks aligned with detections

#     with rasterio.open(args.raster_path) as src:
#         normalizer = build_normalizer(args.normalize)
#         use_cuda = str(args.device).startswith("cuda")
#         amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

#         # batch multiple windows per forward to reduce python and kernel overhead
#         batch_imgs: list[torch.Tensor] = []
#         batch_transforms: list = []
#         batch_windows: list = []
#         batch_size = 8  # tune based on vram; 2–8 is typical for 1024px tiles

#         # small helper to run a batch and collect outputs
#         def run_batch():
#             if not batch_imgs:
#                 return
#             with torch.inference_mode():
#                 with amp_ctx:
#                     preds_list = model(batch_imgs)
#             for preds, win, tform in zip(preds_list, batch_windows, batch_transforms):
#                 boxes = preds["boxes"]
#                 scores = preds["scores"]
#                 labels = preds["labels"]
#                 keep = scores >= args.score_thresh
#                 boxes = boxes[keep]
#                 scores = scores[keep]
#                 labels = labels[keep]
#                 # keep everything on device; convert to cpu only after nms
#                 boxes_global = adjust_boxes_to_global(boxes, win)
#                 all_boxes.append(boxes_global)
#                 all_scores.append(scores)
#                 all_labels.append(labels)
#                 if args.task == 'instance_seg' and 'masks' in preds:
#                     m = preds['masks'][keep]
#                     if m.shape[0] > 0:
#                         mask_slices.append(m)
#                         tile_transforms.extend([tform] * m.shape[0])
#                 else:
#                     if boxes.shape[0] > 0:
#                         tile_transforms.extend([tform] * boxes.shape[0])

#             # clear batch containers
#             batch_imgs.clear()
#             batch_transforms.clear()
#             batch_windows.clear()

#         for i, window in enumerate(windows, start=1):
#             logger.info("[debug] processing window %d/%d", i, len(windows))
#             img = src.read(window=window)
#             tile_transform = src.window_transform(window)
#             if img.shape[0] >= 3:
#                 img = img[:3]
#             else:
#                 repeats = (3 + img.shape[0] - 1) // img.shape[0]
#                 img = np.concatenate([img] * repeats, axis=0)[:3]
#             h, w = img.shape[-2:]
#             if h < 32 or w < 32:
#                 logger.warning("[warn] skipping tiny or empty tile %d: shape=%s", i, img.shape)
#                 continue
#             if not np.isfinite(img).all():
#                 logger.warning("[warn] skipping tile %d with NaN or INF values", i)
#                 continue
#             scale = 65535.0 if src.dtypes[0] in ("uint16", "int16") else 255.0
#             tensor = torch.from_numpy(img).float().div_(scale).clamp_(0.0, 1.0)
#             if normalizer is not None:
#                 tensor = normalizer(tensor)
#             # pin host memory and use non_blocking copy if on cuda
#             if use_cuda:
#                 tensor = tensor.pin_memory()
#             tensor = tensor.to(args.device, non_blocking=True)

#             try:
#                 # enqueue into batch
#                 batch_imgs.append(tensor)
#                 batch_transforms.append(tile_transform)
#                 batch_windows.append(window)
#                 # run batch if full
#                 if len(batch_imgs) >= batch_size:
#                     run_batch()
#             except Exception:
#                 logger.exception("[error] failed on window %d/%d; skipping", i, len(windows))
#                 # drop the partially filled batch item on error
#                 if batch_imgs:
#                     batch_imgs.pop()
#                     batch_transforms.pop()
#                     batch_windows.pop()
#                 continue
#             finally:
#                 # no periodic empty_cache; let the allocator reuse blocks
#                 pass

#         # flush any remainder
#         run_batch()

#     # concatenate on device
#     device = args.device
#     boxes = torch.cat(all_boxes, dim=0).to(device) if all_boxes else torch.empty((0, 4), device=device)
#     scores = torch.cat(all_scores, dim=0).to(device) if all_scores else torch.empty((0,), device=device)
#     labels = torch.cat(all_labels, dim=0).to(device) if all_labels else torch.empty((0,), dtype=torch.int64, device=device)
#     masks = torch.cat(mask_slices, dim=0).to(device) if mask_slices else torch.empty((0, 1, 1, 1), device=device)

#     logger.info("[debug] pre-nms count=%d (score_thresh=%.3f)", len(boxes), args.score_thresh)

#     # class-wise nms on gpu
#     if len(boxes) > 0:
#         keep_indices: list[int] = []
#         for cls_idx in labels.unique():
#             cls_mask = labels == cls_idx
#             if cls_mask.any():
#                 k = nms(boxes[cls_mask], scores[cls_mask], args.nms_iou_thresh)
#                 base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
#                 keep_indices.extend(base_idx[k].tolist())
#         if keep_indices:
#             keep_indices_t = torch.tensor(keep_indices, dtype=torch.long, device=device)
#             boxes = boxes[keep_indices_t]
#             scores = scores[keep_indices_t]
#             labels = labels[keep_indices_t]
#             if args.task == 'instance_seg' and masks.numel() > 0:
#                 masks = masks[keep_indices_t]
#                 tile_transforms = [tile_transforms[i] for i in keep_indices]
#         else:
#             boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]
#             masks = masks[:0]; tile_transforms = []

#     logger.info("[summary] post-nms count=%d", len(boxes))

#     # build geoms
#     geoms, class_names, score_vals = [], [], []

#     with rasterio.open(args.raster_path) as src:
#         crs = src.crs
#         global_transform = src.transform

#     # move tensors to cpu for vectorization and output
#     boxes = boxes.to("cpu")
#     scores = scores.to("cpu")
#     labels = labels.to("cpu")
#     masks = masks.to("cpu")

#     if args.task == 'instance_seg' and len(boxes) > 0 and masks.numel() > 0:
#         # vectorize each mask to polygons using the respective tile transform
#         masks_np = (masks.squeeze(1).numpy() > 0.5).astype('uint8')  # (n, h, w)
#         for idx, (m, tform, s, l) in enumerate(zip(masks_np, tile_transforms, scores.tolist(), labels.tolist())):
#             for geom, val in rio_features.shapes(m, transform=tform):
#                 if int(val) == 0:
#                     continue
#                 # geom is geojson-like mapping in map coords; convert to shapely geometry
#                 from shapely.geometry import shape as shapely_shape
#                 geoms.append(shapely_shape(geom))
#                 class_names.append(classes[l - 1])
#                 score_vals.append(float(s))
#     else:
#         # fallback: write boxes as polygons in map coordinates
#         for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
#             xmin, ymin, xmax, ymax = b
#             x0, y0 = global_transform * (xmin, ymin)
#             x1, y1 = global_transform * (xmax, ymax)
#             geom = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
#             geoms.append(geom)
#             class_names.append(classes[l - 1])
#             score_vals.append(float(s))

#     gdf = gpd.GeoDataFrame({"classname": class_names, "score": score_vals}, geometry=geoms, crs=crs)

#     out_dir = os.path.dirname(args.out_vector) or "."
#     os.makedirs(out_dir, exist_ok=True)
#     # always write gpkg to support polygons
#     layer_name = "detections"
#     out_path = args.out_vector if args.out_vector.lower().endswith('.gpkg') else os.path.splitext(args.out_vector)[0] + '.gpkg'
#     if len(gdf) == 0:
#         gdf.to_file(out_path, driver="GPKG", layer=layer_name)
#         logger.info("[summary] detections=0; wrote empty layer to %s", out_path)
#     else:
#         gdf.to_file(out_path, driver="GPKG", layer=layer_name)
#         logger.info("[summary] detections=%d; wrote %s", len(gdf), out_path)

#     summary = {
#         "detections": int(len(gdf)),
#         "score_thresh": float(args.score_thresh),
#         "nms_iou_thresh": float(args.nms_iou_thresh),
#         "out_vector": out_path,
#         "raster_path": args.raster_path,
#         "checkpoint": args.checkpoint,
#         "task": args.task,
#         "elapsed_sec": round(time.time() - t0, 2),
#     }
#     summary_path = os.path.join(out_dir, "inference_summary.json")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)
#     logger.info("[done] wrote summary to %s (elapsed %.2fs)", summary_path, summary["elapsed_sec"])


# if __name__ == '__main__':
#     main()
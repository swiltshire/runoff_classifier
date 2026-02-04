
# all variable names are snake_case and all comments are lowercase
from __future__ import annotations
import os
import json
import argparse
import torch
from contextlib import nullcontext
import numpy as np
import geopandas as gpd
from shapely.geometry import box as shapely_box
import rasterio
from rasterio import features as rio_features
from torchvision.ops import nms
from torchvision import transforms
import sys
import logging
import time

# add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
from src.utils.tiling import make_grid_windows, adjust_boxes_to_global


def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def build_normalizer(norm_type: str):
    if norm_type.lower() == "imagenet":
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='sliding-window inference for boxes or instance masks')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'instance_seg'])
    parser.add_argument('--raster_path', type=str, required=True)
    parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_vector', type=str, default='detections.gpkg')
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--score_thresh', type=float, default=0.5)
    parser.add_argument('--nms_iou_thresh', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    return args


def load_classes_and_task_from_checkpoint(ckpt_path: str):
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        classes = data.get('classes', None)
        task = data.get('task', 'detection')
        if classes is not None:
            return classes, task
    # fallback to json next to checkpoint
    json_path = os.path.join(os.path.dirname(ckpt_path), 'classes.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)['classes'], 'detection'
    raise ValueError('could not determine classes from checkpoint or sidecar json')


def main():
    args = parse_args()
    log_path = os.path.join(os.path.dirname(args.out_vector) or ".", "inference.log")
    logger = setup_logging(log_path)
    t0 = time.time()
    logger.info("[debug] starting inference")

    classes, task_from_ckpt = load_classes_and_task_from_checkpoint(args.checkpoint)
    # allow overriding by arg, but warn if mismatch
    if args.task != task_from_ckpt:
        logger.warning("[warn] overriding task from checkpoint (%s) with arg (%s)", task_from_ckpt, args.task)

    num_classes = len(classes) + 1
    if args.task == 'instance_seg':
        model = build_maskrcnn_model(num_classes=num_classes, pretrained=False, device=args.device)
    else:
        model = build_fasterrcnn_model(num_classes=num_classes, pretrained=False, device=args.device)

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    logger.info("[debug] model loaded and set to eval (device=%s)", args.device)

    windows = make_grid_windows(args.raster_path, tile_size=args.tile_size, stride=args.stride)
    logger.info("[debug] num_windows=%d", len(windows))

    all_boxes, all_scores, all_labels = [], [], []
    tile_transforms = []  # keep per-detection tile transform for correct vectorization
    mask_slices = []      # store masks aligned with detections

    with rasterio.open(args.raster_path) as src:
        normalizer = build_normalizer(args.normalize)
        use_cuda = str(args.device).startswith("cuda")
        amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

        for i, window in enumerate(windows, start=1):
            logger.info("[debug] processing window %d/%d", i, len(windows))
            img = src.read(window=window)
            tile_transform = src.window_transform(window)
            if img.shape[0] >= 3:
                img = img[:3]
            else:
                repeats = (3 + img.shape[0] - 1) // img.shape[0]
                img = np.concatenate([img] * repeats, axis=0)[:3]
            h, w = img.shape[-2:]
            if h < 32 or w < 32:
                logger.warning("[warn] skipping tiny or empty tile %d: shape=%s", i, img.shape)
                continue
            if not np.isfinite(img).all():
                logger.warning("[warn] skipping tile %d with NaN or INF values", i)
                continue
            scale = 65535.0 if src.dtypes[0] in ("uint16", "int16") else 255.0
            tensor = torch.from_numpy(img).float().div_(scale).clamp_(0.0, 1.0)
            if normalizer is not None:
                tensor = normalizer(tensor)
            tensor = tensor.to(args.device, non_blocking=False)

            try:
                with torch.inference_mode():
                    with amp_ctx:
                        preds = model([tensor])[0]
                boxes = preds["boxes"]
                scores = preds["scores"]
                labels = preds["labels"]

                keep = scores >= args.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                boxes_global = adjust_boxes_to_global(boxes, window)
                all_boxes.append(boxes_global.to("cpu"))
                all_scores.append(scores.to("cpu"))
                all_labels.append(labels.to("cpu"))

                if args.task == 'instance_seg' and 'masks' in preds:
                    masks = preds['masks'][keep].to("cpu")  # (n, 1, h, w)
                    mask_slices.append(masks)
                    if masks.shape[0] > 0:
                        tile_transforms.extend([tile_transform] * masks.shape[0])
                else:
                    if boxes.shape[0] > 0:
                        tile_transforms.extend([tile_transform] * boxes.shape[0])

            except Exception:
                logger.exception("[error] failed on window %d/%d; skipping", i, len(windows))
                continue
            finally:
                del tensor
                if use_cuda and (i % 10 == 0):
                    torch.cuda.empty_cache()

    # concatenate
    boxes = torch.cat(all_boxes, dim=0) if len(all_boxes) else torch.empty((0, 4))
    scores = torch.cat(all_scores, dim=0) if len(all_scores) else torch.empty((0,))
    labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.empty((0,), dtype=torch.int64)
    masks = torch.cat(mask_slices, dim=0) if len(mask_slices) else torch.empty((0, 1, 1, 1))

    logger.info("[debug] pre-nms count=%d (score_thresh=%.3f)", len(boxes), args.score_thresh)

    # class-wise nms
    if len(boxes) > 0:
        keep_indices = []
        for cls_idx in labels.unique():
            cls_mask = labels == cls_idx
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            if cls_boxes.numel() == 0:
                continue
            keep = nms(cls_boxes, cls_scores, args.nms_iou_thresh)
            base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
            keep_indices.extend(base_idx[keep].tolist())
        if len(keep_indices) > 0:
            keep_indices = torch.tensor(keep_indices, dtype=torch.long)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            if args.task == 'instance_seg' and masks.numel() > 0:
                masks = masks[keep_indices]
                tile_transforms = [tile_transforms[i] for i in keep_indices.tolist()]
        else:
            boxes = boxes[:0]
            scores = scores[:0]
            labels = labels[:0]
            masks = masks[:0]
            tile_transforms = []

    logger.info("[summary] post-nms count=%d", len(boxes))

    # build geoms
    geoms, class_names, score_vals = [], [], []

    with rasterio.open(args.raster_path) as src:
        crs = src.crs
        global_transform = src.transform

    if args.task == 'instance_seg' and len(boxes) > 0 and masks.numel() > 0:
        # vectorize each mask to polygons using the respective tile transform
        masks_np = (masks.squeeze(1).numpy() > 0.5).astype('uint8')  # (n, h, w)
        for idx, (m, tform, s, l) in enumerate(zip(masks_np, tile_transforms, scores.tolist(), labels.tolist())):
            for geom, val in rio_features.shapes(m, transform=tform):
                if int(val) == 0:
                    continue
                # geom is geojson-like mapping in map coords; convert to shapely geometry
                from shapely.geometry import shape as shapely_shape
                geoms.append(shapely_shape(geom))
                class_names.append(classes[l - 1])
                score_vals.append(float(s))
    else:
        # fallback: write boxes as polygons in map coordinates
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            xmin, ymin, xmax, ymax = b
            x0, y0 = global_transform * (xmin, ymin)
            x1, y1 = global_transform * (xmax, ymax)
            geom = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            geoms.append(geom)
            class_names.append(classes[l - 1])
            score_vals.append(float(s))

    gdf = gpd.GeoDataFrame({"classname": class_names, "score": score_vals}, geometry=geoms, crs=crs)

    out_dir = os.path.dirname(args.out_vector) or "."
    os.makedirs(out_dir, exist_ok=True)
    # always write gpkg to support polygons
    layer_name = "detections"
    out_path = args.out_vector if args.out_vector.lower().endswith('.gpkg') else os.path.splitext(args.out_vector)[0] + '.gpkg'
    if len(gdf) == 0:
        gdf.to_file(out_path, driver="GPKG", layer=layer_name)
        logger.info("[summary] detections=0; wrote empty layer to %s", out_path)
    else:
        gdf.to_file(out_path, driver="GPKG", layer=layer_name)
        logger.info("[summary] detections=%d; wrote %s", len(gdf), out_path)

    summary = {
        "detections": int(len(gdf)),
        "score_thresh": float(args.score_thresh),
        "nms_iou_thresh": float(args.nms_iou_thresh),
        "out_vector": out_path,
        "raster_path": args.raster_path,
        "checkpoint": args.checkpoint,
        "task": args.task,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    summary_path = os.path.join(out_dir, "inference_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("[done] wrote summary to %s (elapsed %.2fs)", summary_path, summary["elapsed_sec"])


if __name__ == '__main__':
    main()

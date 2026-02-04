
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
from rasterio.windows import Window
from torchvision.ops import nms
from torchvision import transforms
import sys


# add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.models.model import build_fasterrcnn_model
from src.utils.tiling import make_grid_windows, adjust_boxes_to_global


import logging
import time

# basic logger to both console and file
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
    # returns a callable that normalizes a CxHxW float tensor
    if norm_type.lower() == "imagenet":
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return None  # 'none'

def parse_args():
    parser = argparse.ArgumentParser(description='sliding-window inference for object detection')
    parser.add_argument('--raster_path', type=str, required=True)
    parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_vector', type=str, default='detections.shp')
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--score_thresh', type=float, default=0.5)
    parser.add_argument('--nms_iou_thresh', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    return args


def load_classes_from_checkpoint(ckpt_path: str):
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict) and 'classes' in data:
        return data['classes']
    # fallback to json next to checkpoint
    json_path = os.path.join(os.path.dirname(ckpt_path), 'classes.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)['classes']
    raise ValueError('could not determine classes from checkpoint or sidecar json')



def read_window_rgb(raster_path: str, window: Window):
    # read as uint8 directly
    with rasterio.open(raster_path) as src:
        img = src.read(window=window)  # typically uint8 or uint16
        transform = src.window_transform(window)
        full_transform = src.transform
        crs = src.crs

    # ensure 3 channels
    if img.shape[0] >= 3:
        img = img[:3]
    else:
        # repeat channels without allocating large temporary arrays
        repeats = (3 + img.shape[0] - 1) // img.shape[0]
        img = np.concatenate([img] * repeats, axis=0)[:3]

    # NORMALIZE LATER inside torch, not in numpy
    # return as uint8 to save memory
    return img, transform, full_transform, crs




def main():
    args = parse_args()

    log_path = os.path.join(os.path.dirname(args.out_vector) or ".", "inference.log")
    logger = setup_logging(log_path)
    t0 = time.time()
    logger.info("[debug] starting inference")

    classes = load_classes_from_checkpoint(args.checkpoint)
    num_classes = len(classes) + 1
    model = build_fasterrcnn_model(num_classes=num_classes, pretrained=False, device=args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    logger.info("[debug] model loaded and set to eval (device=%s)", args.device)

    # rpn/roi reductions
    windows = make_grid_windows(args.raster_path, tile_size=args.tile_size, stride=args.stride)
    logger.info("[debug] num_windows=%d", len(windows))

    all_boxes, all_scores, all_labels = [], [], []

    # open the raster once
    with rasterio.open(args.raster_path) as src:
        transform = src.transform
        crs = src.crs
        normalizer = build_normalizer(args.normalize)
        
        # autocast only on cuda
        use_cuda = str(args.device).startswith("cuda")
        amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

        for i, window in enumerate(windows, start=1):
            logger.info("[debug] processing window %d/%d", i, len(windows))

            # read from the already-open dataset
            img = src.read(window=window)
            # ensure 3 channels
            if img.shape[0] >= 3:
                img = img[:3]
            else:
                repeats = (3 + img.shape[0] - 1) // img.shape[0]
                img = np.concatenate([img] * repeats, axis=0)[:3]

            # basic sanity checks
            h, w = img.shape[-2:]
            if h < 32 or w < 32:
                logger.warning("[warn] skipping tiny or empty tile %d: shape=%s", i, img.shape)
                continue
            if not np.isfinite(img).all():
                logger.warning("[warn] skipping tile %d with NaN or INF values", i)
                continue

            # to tensor and device
            # dtype-aware scaling to [0,1]
            scale = 65535.0 if src.dtypes[0] in ("uint16", "int16") else 255.0
            tensor = torch.from_numpy(img).float().div_(scale).clamp_(0.0, 1.0)

            # optional imagenet normalization to match training
            if normalizer is not None:
                tensor = normalizer(tensor)

            tensor = tensor.to(args.device, non_blocking=False)

            try:
                with torch.inference_mode():
                    # amp is optional; nest to avoid graph building
                    with amp_ctx:
                        preds = model([tensor])[0]

                boxes = preds["boxes"]
                scores = preds["scores"]
                labels = preds["labels"]

                keep = scores >= args.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                boxes = adjust_boxes_to_global(boxes, window)

                all_boxes.append(boxes.to("cpu"))
                all_scores.append(scores.to("cpu"))
                all_labels.append(labels.to("cpu"))

            except Exception as e:
                # this guarantees any failure is logged with stack trace, not silent
                logger.exception("[error] failed on window %d/%d; skipping", i, len(windows))
                continue
            finally:
                # proactively free per-iteration GPU memory
                del tensor
                if use_cuda and (i % 10 == 0):
                    torch.cuda.empty_cache()
                    # optional: log memory usage
                    try:
                        alloc = torch.cuda.memory_allocated() / (1024**2)
                        reserv = torch.cuda.memory_reserved() / (1024**2)
                        logger.info("[mem] cuda allocated=%.1fMB reserved=%.1fMB", alloc, reserv)
                    except Exception:
                        pass

    logger.info("[debug] finished windows; collected %d window-level tensors", len(all_boxes))

    # concatenate and nms
    boxes = torch.cat(all_boxes, dim=0) if len(all_boxes) else torch.empty((0, 4))
    scores = torch.cat(all_scores, dim=0) if len(all_scores) else torch.empty((0,))
    labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.empty((0,), dtype=torch.int64)
    logger.info("[debug] pre-nms count=%d (score_thresh=%.3f)", len(boxes), args.score_thresh)

    if len(boxes) > 0:
        keep_indices = []
        for cls_idx in labels.unique():
            cls_mask = labels == cls_idx
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            if cls_boxes.numel() == 0:
                continue
            keep = nms(cls_boxes, cls_scores, args.nms_iou_thresh)
            keep_indices.extend(torch.nonzero(cls_mask, as_tuple=False).squeeze(1)[keep].tolist())
        if len(keep_indices) > 0:
            keep_indices = torch.tensor(keep_indices, dtype=torch.long)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        else:
            boxes = boxes[:0]
            scores = scores[:0]
            labels = labels[:0]

    logger.info("[summary] post-nms count=%d", len(boxes))

    # build geoms and write (we already have transform & crs from the opened src)
    geoms, class_names, score_vals = [], [], []
    for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        xmin, ymin, xmax, ymax = b
        x0, y0 = transform * (xmin, ymin)
        x1, y1 = transform * (xmax, ymax)
        geom = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        geoms.append(geom)
        class_names.append(classes[l - 1])
        score_vals.append(float(s))

    gdf = gpd.GeoDataFrame({"classname": class_names, "score": score_vals}, geometry=geoms, crs=crs)

    out_dir = os.path.dirname(args.out_vector) or "."
    os.makedirs(out_dir, exist_ok=True)

    if len(gdf) == 0:
        out_written = os.path.splitext(args.out_vector)[0] + ".gpkg"
        gdf.to_file(out_written, driver="GPKG", layer="detections")
        logger.info("[summary] detections=0; wrote empty layer to %s", out_written)
    else:
        gdf.to_file(args.out_vector)
        out_written = args.out_vector
        logger.info("[summary] detections=%d; wrote %s", len(gdf), out_written)

    summary = {
        "detections": int(len(gdf)),
        "score_thresh": float(args.score_thresh),
        "nms_iou_thresh": float(args.nms_iou_thresh),
        "out_vector": out_written,
        "raster_path": args.raster_path,
        "checkpoint": args.checkpoint,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    summary_path = os.path.join(out_dir, "inference_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("[done] wrote summary to %s (elapsed %.2fs)", summary_path, summary["elapsed_sec"])



if __name__ == '__main__':
    main()

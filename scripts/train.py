from __future__ import annotations
import os
import json
import argparse
import time
from datetime import timedelta
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import sys
import glob
from rasterio.windows import Window  # for (de)serializing training windows

# add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.crs_utils import normalize_labels_crs
from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
from src.data.dataset import ObjectDetectionTilesDataset, detection_collate_fn
from src.utils.tiling import make_label_centered_training_windows


def parse_args():
    parser = argparse.ArgumentParser(description='train detection or instance segmentation on geospatial tiles (ddp-ready)')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'instance_seg'])
    parser.add_argument('--raster_path', type=str, required=True)
    parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'])
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--max_per_class', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='number of batches preloaded per worker (ignored when num_workers=0)')    
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='micro-batches per optimizer step (accumulated gradients)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--classes_json', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=1)
    return parser.parse_args()


def build_vrt_from_tifs(files, out_vrt):
    try:
        from osgeo import gdal
    except ImportError:
        raise RuntimeError("python gdal (osgeo.gdal) is not installed.")
    gdal.UseExceptions()
    os.makedirs(os.path.dirname(out_vrt) or ".", exist_ok=True)
    vrt = gdal.BuildVRT(out_vrt, [os.path.abspath(f) for f in files])
    if vrt is None:
        raise RuntimeError("gdal.BuildVRT failed to create the vrt.")
    vrt.FlushCache()
    vrt = None


def resolve_raster_input(path_or_dir):
    # directory of tifs
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*.tif")))
        if not files:
            raise FileNotFoundError(f"no .tif files found in {path_or_dir}")
        parent = os.path.dirname(os.path.abspath(path_or_dir))
        parent_name = os.path.basename(parent)
        out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
        if not os.path.exists(out_vrt):
            build_vrt_from_tifs(files, out_vrt)
        return out_vrt
    # glob (*.tif)
    if any(ch in path_or_dir for ch in ["*", "?", "["]):
        files = sorted(glob.glob(path_or_dir))
        if not files:
            raise FileNotFoundError(f"glob matched no .tif files: {path_or_dir}")
        tile_dir = os.path.dirname(os.path.abspath(path_or_dir))
        parent = os.path.dirname(tile_dir)
        parent_name = os.path.basename(parent)
        out_vrt = os.path.join(parent, f"{parent_name}_mosaic.vrt")
        if not os.path.exists(out_vrt):
            build_vrt_from_tifs(files, out_vrt)
        return out_vrt
    return path_or_dir


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def init_distributed_if_needed():
    # torchrun sets these env vars; this is a no-op on single-gpu
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
        # touch the local device once so nccl knows which device this rank uses
        if torch.cuda.is_available():
            _ = torch.zeros(1, device=f'cuda:{local_rank}')
        return local_rank
    return int(os.environ.get('LOCAL_RANK', 0))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    local_rank = init_distributed_if_needed()
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    # resolve raster (rank 0), then barrier so all ranks see it
    if is_main_process():
        args.raster_path = resolve_raster_input(args.raster_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        args.raster_path = resolve_raster_input(args.raster_path)

    # normalize labels (rank 0), then barrier
    if is_main_process():
            # simple startup banner; print only on rank 0
        print(f"INFO [debug] starting training (world_size={world_size})")
        
        args.labels_path = normalize_labels_crs(
            labels_path=args.labels_path,
            raster_path=args.raster_path,
            out_path=None,
            prefer_gpkg=False,
            fix_invalid=True,
            overwrite=True,
        )
        print(f"INFO [debug] using labels: {args.labels_path}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # classes
    if args.classes_json and os.path.exists(args.classes_json):
        with open(args.classes_json) as f:
            classes = json.load(f)['classes']
    else:
        classes = ["Bank_Erosion", "Spillway", "Culvert_Structure", "Tile_Inlet", "Tile_Outlet"]

    # build training windows on rank 0, cache to json with explicit fields
    windows_cache = os.path.join(args.out_dir, "train_windows.json")
    if is_main_process():
        tile_windows = make_label_centered_training_windows(
            raster_path=args.raster_path,
            labels_path=args.labels_path,
            tile_size=args.tile_size,
            max_per_class=args.max_per_class,
            classname_field='Classname',
            jitter=64,
        )
        windows_json = [
            {"col_off": float(w.col_off), "row_off": float(w.row_off), "width": float(w.width), "height": float(w.height)}
            for w in tile_windows
        ]
        with open(windows_cache, "w") as f:
            json.dump(windows_json, f)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    # all ranks load the same windows
    with open(windows_cache, "r") as f:
        windows_json = json.load(f)
    tile_windows = [Window(w["col_off"], w["row_off"], w["width"], w["height"]) for w in windows_json]

    dataset = ObjectDetectionTilesDataset(
        raster_path=args.raster_path,
        labels_path=args.labels_path,
        classes=classes,
        tile_windows=tile_windows,
        classname_field='Classname',
        normalize=args.normalize,
        include_masks=(args.task == 'instance_seg'),
    )

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        persistent_workers=(True if args.num_workers > 0 else False),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    # print batches per rank
    if is_main_process():
        print(f"INFO [debug] dataloader batches per rank ≈ {len(dataloader)}")
    rank_id = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


    # device per rank
    device = args.device
    if device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)

    num_classes = len(classes) + 1
    if args.task == 'instance_seg':
        model = build_maskrcnn_model(num_classes=num_classes, pretrained=True, device=device)
    else:
        model = build_fasterrcnn_model(num_classes=num_classes, pretrained=True, device=device)

    if dist.is_available() and dist.is_initialized() and device.startswith('cuda'):
        model = ddp(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # save classes for inference (rank 0)
    if is_main_process():
        with open(os.path.join(args.out_dir, 'classes.json'), 'w') as f:
            json.dump({'classes': classes, 'task': args.task}, f, indent=2)

    use_cuda = device.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    # setup gradient accumulation
    accum = max(1, int(args.grad_accum))
    step_i = 0

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss_local = 0.0
        pbar = tqdm(dataloader, desc=f'epoch {epoch}/{args.epochs}', disable=not is_main_process())
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

            if step_i % accum == 0:
                optimizer.zero_grad(set_to_none=True)

            # skip ddp gradient allreduce on all but the last micro-batch in this window
            ddp_sync = ((step_i + 1) % accum == 0)
            if hasattr(model, "no_sync") and not ddp_sync:
                sync_ctx = model.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                scaler.scale(losses / accum).backward()

            if ddp_sync:
                scaler.step(optimizer)
                scaler.update()

            step_i += 1

            epoch_loss_local += float(losses.detach().item())
            if is_main_process():
                pbar.set_postfix(loss=f"{float(losses):.3f}")

        # reduce average loss to rank 0 for logging
        if dist.is_available() and dist.is_initialized():
            device_for_reduce = device if use_cuda else 'cpu'
            loss_tensor = torch.tensor([epoch_loss_local], device=device_for_reduce)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss_global = loss_tensor.item() / world_size
        else:
            epoch_loss_global = epoch_loss_local
        if is_main_process():
            avg_loss = epoch_loss_global / max(1, len(dataloader))
            print(f'epoch {epoch} average loss (global): {avg_loss:.4f}')

        if is_main_process() and (epoch % args.save_every == 0):
            ckpt_path = os.path.join(args.out_dir, f'model_epoch_{epoch}.pth')
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({'model_state': state_dict, 'classes': classes, 'task': args.task}, ckpt_path)
            print(f'saved checkpoint: {ckpt_path}')

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main_process():
        final_path = os.path.join(args.out_dir, 'model_final.pth')
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({'model_state': state_dict, 'classes': classes, 'task': args.task}, final_path)
        print(f'saved final model: {final_path}')

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()



# all variable names are snake_case and all comments are lowercase
from __future__ import annotations
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.crs_utils import normalize_labels_crs
from src.models.model import build_fasterrcnn_model, build_maskrcnn_model
from src.data.dataset import ObjectDetectionTilesDataset, detection_collate_fn
from src.utils.tiling import make_label_centered_training_windows


def parse_args():
    parser = argparse.ArgumentParser(description='train detection or instance segmentation on geospatial tiles')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'instance_seg'],
                        help='training task: detection (boxes) or instance_seg (mask r-cnn)')
    parser.add_argument('--raster_path', type=str, required=True, help='path to rgb(a) geotiff')
    parser.add_argument('--normalize', type=str, default='none', choices=['none', 'imagenet'],
                        help='input normalization: none=[0,1]; imagenet=mean/std')
    parser.add_argument('--labels_path', type=str, required=True, help='path to polygon vector with Classname field')
    parser.add_argument('--out_dir', type=str, default='outputs', help='output directory')
    parser.add_argument('--tile_size', type=int, default=512, help='tile size in pixels')
    parser.add_argument('--max_per_class', type=int, default=500, help='max training windows per class')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--classes_json', type=str, default=None, help='optional json file with {"classes": [...]}' )
    parser.add_argument('--save_every', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # normalize training labels to raster crs so geometry aligns with imagery
    args.labels_path = normalize_labels_crs(
        labels_path=args.labels_path,
        raster_path=args.raster_path,
        out_path=None,
        prefer_gpkg=False,
        fix_invalid=True,
        overwrite=True,
    )
    print(f"[info] using labels: {args.labels_path}")

    # load classes
    if args.classes_json and os.path.exists(args.classes_json):
        with open(args.classes_json) as f:
            classes = json.load(f)['classes']
    else:
        classes = ["Bank_Erosion", "Spillway", "Culvert_Structure", "Tile_Inlet", "Tile_Outlet"]

    # build training windows centered around labels to ensure positive samples
    tile_windows = make_label_centered_training_windows(
        raster_path=args.raster_path,
        labels_path=args.labels_path,
        tile_size=args.tile_size,
        max_per_class=args.max_per_class,
        classname_field='Classname',
        jitter=64,
    )

    dataset = ObjectDetectionTilesDataset(
        raster_path=args.raster_path,
        labels_path=args.labels_path,
        classes=classes,
        tile_windows=tile_windows,
        classname_field='Classname',
        normalize=args.normalize,
        include_masks=(args.task == 'instance_seg'),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    num_classes = len(classes) + 1
    if args.task == 'instance_seg':
        model = build_maskrcnn_model(num_classes=num_classes, pretrained=True, device=args.device)
    else:
        model = build_fasterrcnn_model(num_classes=num_classes, pretrained=True, device=args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # save class mapping for inference
    with open(os.path.join(args.out_dir, 'classes.json'), 'w') as f:
        json.dump({'classes': classes}, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(dataloader, desc=f'epoch {epoch}/{args.epochs}'):
            images = [img.to(args.device) for img in images]
            # move tensors in targets to device (masks included if present)
            targets = [{k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f'epoch {epoch} average loss: {avg_loss:.4f}')
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f'model_epoch_{epoch}.pth')
            torch.save({'model_state': model.state_dict(), 'classes': classes, 'task': args.task}, ckpt_path)
            print(f'saved checkpoint: {ckpt_path}')

    # save final model
    final_path = os.path.join(args.out_dir, 'model_final.pth')
    torch.save({'model_state': model.state_dict(), 'classes': classes, 'task': args.task}, final_path)
    print(f'saved final model: {final_path}')


if __name__ == '__main__':
    main()

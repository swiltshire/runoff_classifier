# all variable names are snake_case and all comments are lowercase
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features as rio_features
from affine import Affine
import geopandas as gpd
from shapely.geometry import box
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from rasterio.transform import rowcol

def world_to_pixel(x: float, y: float, transform: Affine) -> Tuple[float, float]:
    inv = ~transform
    col, row = inv * (x, y)
    return col, row

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

class ObjectDetectionTilesDataset(Dataset):
    """
    a dataset that serves tiled chips from a large geotiff and produces
    torchvision-style detection targets from polygon labels.

    each sample returns:
      image: float tensor (3,h,w), rgb in [0,1]
      target: dict with keys: boxes, labels, area, iscrowd, image_id, (optional) masks
    """

    def __init__(
        self,
        raster_path: str,
        labels_path: str,
        classes: List[str],
        tile_windows: List[Window],
        classname_field: str = 'Classname',
        normalize: str = 'none',
        include_masks: bool = False,
    ) -> None:
        super().__init__()
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.classname_field = classname_field
        self.classes = classes
        self.class_to_idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 is background
        self.tile_windows = tile_windows
        self.normalize = normalize
        self.include_masks = include_masks

        self.imagenet_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ) if self.normalize == 'imagenet' else None

        # read geodataframe and ensure crs aligns with raster
        with rasterio.open(self.raster_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.height = src.height
            self.width = src.width
            gdf = gpd.read_file(self.labels_path)
            if gdf.crs is None:
                raise ValueError('labels must have a valid crs')
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)
            self.gdf = gdf

    def __len__(self) -> int:
        return len(self.tile_windows)

    def _window_bounds_world(self, window: Window) -> Tuple[float, float, float, float]:
        # returns (minx, miny, maxx, maxy) in world coordinates
        minx, miny = (window.col_off, window.row_off) * self.transform
        maxx, maxy = ((window.col_off + window.width), (window.row_off + window.height)) * self.transform
        x0, x1 = sorted([minx, maxx])
        y0, y1 = sorted([miny, maxy])
        return (x0, y0, x1, y1)

    def _labels_for_window(self, window: Window, tile_h: int, tile_w: int):
        # compute window polygon in raster crs
        minx, miny, maxx, maxy = self._window_bounds_world(window)
        window_geom = box(minx, miny, maxx, maxy)
        # pick candidates that intersect the window
        hits = self.gdf[self.gdf.geometry.intersects(window_geom)]
        boxes_list, labels_list, masks_list = [], [], []

        # compute the tile's affine transform (pixel -> world) once
        with rasterio.open(self.raster_path) as src:
            tile_transform = src.window_transform(window)

        for _, row in hits.iterrows():
            geom = row.geometry.intersection(window_geom)
            if geom is None or geom.is_empty:
                continue
            # bounds of the clipped geometry in world coords
            gx0, gy0, gx1, gy1 = geom.bounds
            r0, c0 = rowcol(self.transform, gx0, gy1, op=float)
            r1, c1 = rowcol(self.transform, gx1, gy0, op=float)
            x0 = c0 - window.col_off
            x1 = c1 - window.col_off
            y0 = r0 - window.row_off
            y1 = r1 - window.row_off
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            w, h = float(window.width), float(window.height)
            x_min = max(0.0, min(w, x_min))
            x_max = max(0.0, min(w, x_max))
            y_min = max(0.0, min(h, y_min))
            y_max = max(0.0, min(h, y_max))
            if (x_max - x_min) <= 0.0 or (y_max - y_min) <= 0.0:
                continue
            cls_name = row[self.classname_field]
            cls_id = self.class_to_idx.get(cls_name, 0)
            if cls_id <= 0:
                continue
            boxes_list.append([x_min, y_min, x_max, y_max])
            labels_list.append(cls_id)

            if self.include_masks:
                # rasterize the clipped geometry into tile pixel space
                mask = rio_features.rasterize(
                    [(geom, 1)],
                    out_shape=(tile_h, tile_w),
                    transform=tile_transform,
                    fill=0,
                    dtype='uint8',
                    all_touched=False,
                )
                masks_list.append(mask)

        if len(boxes_list) == 0:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int64)
            if self.include_masks:
                masks_np = np.zeros((0, tile_h, tile_w), dtype=np.uint8)
            else:
                masks_np = None
        else:
            boxes_np = np.array(boxes_list, dtype=np.float32)
            labels_np = np.array(labels_list, dtype=np.int64)
            masks_np = np.stack(masks_list, axis=0).astype('uint8') if self.include_masks else None

        return boxes_np, labels_np, masks_np

    def __getitem__(self, idx: int):
        window = self.tile_windows[idx]
        with rasterio.open(self.raster_path) as src:
            img = src.read(window=window)
            tile_transform = src.window_transform(window)
        if img.shape[0] >= 3:
            img = img[:3, :, :]
        else:
            repeats = (3 + img.shape[0] - 1) // img.shape[0]
            img = np.concatenate([img] * repeats, axis=0)[:3]
        img = img.astype(np.float32)
        with rasterio.open(self.raster_path) as _src:
            dtype = _src.dtypes[0]
            scale = 65535.0 if dtype in ("uint16", "int16") else 255.0
        img = (img / scale).clip(0.0, 1.0)
        image_tensor = torch.from_numpy(img)
        if self.imagenet_norm is not None:
            image_tensor = self.imagenet_norm(image_tensor)

        tile_h, tile_w = int(window.height), int(window.width)
        boxes_np, labels_np, masks_np = self._labels_for_window(window, tile_h, tile_w)

        target = {
            'boxes': torch.from_numpy(boxes_np),
            'labels': torch.from_numpy(labels_np),
            'area': torch.tensor([
                (b[2] - b[0]) * (b[3] - b[1]) for b in boxes_np
            ], dtype=torch.float32),
            'iscrowd': torch.zeros((boxes_np.shape[0],), dtype=torch.int64),
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'transform': tile_transform,
        }
        if self.include_masks and masks_np is not None:
            target['masks'] = torch.from_numpy(masks_np)

        return image_tensor, target

def detection_collate_fn(batch):
    # collate function for torchvision detection models
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets
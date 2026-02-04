
# all variable names are snake_case and all comments are lowercase

from __future__ import annotations
from typing import List, Dict, Tuple
import random
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from affine import Affine
import geopandas as gpd
from shapely.geometry import box, Polygon
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from rasterio.transform import rowcol
from rasterio.windows import bounds as window_bounds



# utility: convert world xy to pixel ij using inverse affine

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
      target: dict with keys: boxes, labels, area, iscrowd, image_id
    """

    def __init__(
        self,
        raster_path: str,
        labels_path: str,
        classes: List[str],
        tile_windows: List[Window],
        classname_field: str = 'Classname',
        normalize: str = "None"
    ) -> None:
        super().__init__()
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.classname_field = classname_field
        self.classes = classes
        self.class_to_idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 is background
        self.tile_windows = tile_windows
        self.normalize = normalize

        
        # prepare a normalization transform if requested
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
        # account for north-up vs south-up by sorting
        x0, x1 = sorted([minx, maxx])
        y0, y1 = sorted([miny, maxy])
        return (x0, y0, x1, y1)

    
    def _labels_for_window(self, window: Window) -> Tuple[np.ndarray, np.ndarray]:
        # compute window polygon in raster crs
        minx, miny, maxx, maxy = self._window_bounds_world(window)
        window_geom = box(minx, miny, maxx, maxy)

        # pick candidates that intersect the window
        hits = self.gdf[self.gdf.geometry.intersects(window_geom)]

        boxes, labels = [], []
        for _, row in hits.iterrows():
            geom = row.geometry.intersection(window_geom)
            if geom is None or geom.is_empty:
                continue

            # bounds of the clipped geometry in world coords
            gx0, gy0, gx1, gy1 = geom.bounds

            # convert to pixel indices on the *full image*.
            # important: use (x_min, y_max) and (x_max, y_min) for north-up transforms,
            # or simply sort after conversion.
            r0, c0 = rowcol(self.transform, gx0, gy1, op=float)
            r1, c1 = rowcol(self.transform, gx1, gy0, op=float)

            # localize to the tile (subtract window origin), then sort
            x0 = c0 - window.col_off
            x1 = c1 - window.col_off
            y0 = r0 - window.row_off
            y1 = r1 - window.row_off

            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])

            # clamp to tile extents (inclusive upper bound is fine for floats)
            w, h = float(window.width), float(window.height)
            x_min = max(0.0, min(w, x_min))
            x_max = max(0.0, min(w, x_max))
            y_min = max(0.0, min(h, y_min))
            y_max = max(0.0, min(h, y_max))

            # enforce non-degenerate boxes
            if (x_max - x_min) <= 0.0 or (y_max - y_min) <= 0.0:
                continue

            cls_name = row[self.classname_field]
            cls_id = self.class_to_idx.get(cls_name, 0)  # 1..K; 0 = background/unknown
            if cls_id <= 0:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(cls_id)

        if boxes:
            return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
        else:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


    def __getitem__(self, idx: int):
        window = self.tile_windows[idx]
        with rasterio.open(self.raster_path) as src:
            img = src.read(window=window)  # (bands, h, w)
            transform = src.window_transform(window)
        # use first 3 bands (rgb). if there are more bands, ignore extras.
        if img.shape[0] >= 3:
            img = img[:3, :, :]
        else:
            # pad to 3 channels if needed
            img = np.repeat(img, repeats=3 // img.shape[0] + 1, axis=0)[:3]

        
        img = img.astype(np.float32)
        # dtype-aware scaling to [0,1]
        with rasterio.open(self.raster_path) as _src:
            dtype = _src.dtypes[0]
        scale = 65535.0 if dtype in ("uint16", "int16") else 255.0
        img = (img / scale).clip(0.0, 1.0)
        image_tensor = torch.from_numpy(img)  # CxHxW float32 in [0,1]

        # add this:
        if self.imagenet_norm is not None:
            # torchvision transforms expect tensors in CxHxW
            image_tensor = self.imagenet_norm(image_tensor)


        boxes_np, labels_np = self._labels_for_window(window)
        target = {
            'boxes': torch.from_numpy(boxes_np),
            'labels': torch.from_numpy(labels_np),
            'area': torch.tensor([
                (b[2] - b[0]) * (b[3] - b[1]) for b in boxes_np
            ], dtype=torch.float32),
            'iscrowd': torch.zeros((boxes_np.shape[0],), dtype=torch.int64),
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'transform': transform
        }
        return image_tensor, target


def detection_collate_fn(batch):
    # collate function for torchvision detection models
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets

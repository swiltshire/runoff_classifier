from __future__ import annotations
from typing import List, Tuple
import random
import math
import rasterio
from rasterio.windows import Window
import geopandas as gpd


def make_grid_windows(raster_path: str, tile_size: int, stride: int) -> List[Window]:
    # create a list of sliding windows that cover the raster
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
    windows = []
    for row_off in range(0, height - tile_size + 1, stride):
        for col_off in range(0, width - tile_size + 1, stride):
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
    # add border tiles if the image is not an exact multiple
    if (height - tile_size) % stride != 0:
        row_off = height - tile_size
        for col_off in range(0, width - tile_size + 1, stride):
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
    if (width - tile_size) % stride != 0:
        col_off = width - tile_size
        for row_off in range(0, height - tile_size + 1, stride):
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
    # ensure bottom-right corner included
    windows.append(Window(col_off=width - tile_size, row_off=height - tile_size, width=tile_size, height=tile_size))
    # deduplicate
    unique = {(w.col_off, w.row_off): w for w in windows}
    return list(unique.values())


def make_label_centered_training_windows(raster_path: str,
                                         labels_path: str,
                                         tile_size: int,
                                         max_per_class: int | None = None,
                                         classname_field: str = 'Classname',
                                         jitter: int = 64) -> List[Window]:
    # create training windows centered (with jitter) around label centroids
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        transform = src.transform
    gdf = gpd.read_file(labels_path)
    if gdf.crs is None:
        raise ValueError('labels must have a valid crs')
    # reproject if needed
    # note: we assume labels already match raster crs in the dataset class

    windows = []
    grouped = gdf.groupby(classname_field)
    for cls, group in grouped:
        count = 0
        for _, row in group.iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            inv = ~transform
            col, row_px = inv * (cx, cy)
            # apply random jitter in pixel space
            jx = random.randint(-jitter, jitter)
            jy = random.randint(-jitter, jitter)
            col = int(col + jx)
            row_px = int(row_px + jy)
            # compute top-left corner for a tile centered on (col,row)
            col_off = max(0, min(col - tile_size // 2, width - tile_size))
            row_off = max(0, min(row_px - tile_size // 2, height - tile_size))
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
            count += 1
            if max_per_class is not None and count >= max_per_class:
                break
    # deduplicate windows
    unique = {(w.col_off, w.row_off): w for w in windows}
    return list(unique.values())


def adjust_boxes_to_global(boxes, window):
    # convert window-local pixel boxes to global pixel boxes
    if boxes.numel() == 0:
        return boxes
    offset = boxes.new_tensor([window.col_off, window.row_off, window.col_off, window.row_off])
    return boxes + offset

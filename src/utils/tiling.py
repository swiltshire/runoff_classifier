from __future__ import annotations
from typing import List, Tuple
import random
import math
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box


def filter_windows_by_mask(
    raster_path: str,
    windows: List[Window],
    mask_path: str,
    min_cover_frac: float = 0.0,
) -> List[Window]:
    """
    Keep only windows whose map-space rectangle intersects the mask polygons.
    Optionally require that the intersection area covers at least `min_cover_frac`
    of the window polygon (0.0..1.0).
    """
    if min_cover_frac < 0.0 or min_cover_frac > 1.0:
        raise ValueError("min_cover_frac must be in [0,1].")

    gdf = gpd.read_file(mask_path)
    if gdf.empty:
        return []  # nothing to search
    with rasterio.open(raster_path) as src:
        win_to_world = src.transform
        raster_crs = src.crs

    # reproject mask -> raster CRS if needed, dissolve for faster intersects
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    mask_union = gdf.unary_union  # shapely geometry

    kept = []
    # precompute window area in map units when we need coverage ratio
    need_area = (min_cover_frac > 0.0)

    for w in windows:
        # map-space rectangle of the window
        # NOTE: transform * (col,row) yields map coords; window is in pixel coords
        x0, y0 = win_to_world * (w.col_off,               w.row_off)
        x1, y1 = win_to_world * (w.col_off + w.width,     w.row_off + w.height)
        wx0, wx1 = sorted([x0, x1])
        wy0, wy1 = sorted([y0, y1])
        w_geom = box(wx0, wy0, wx1, wy1)

        if not w_geom.intersects(mask_union):
            continue
        if need_area:
            inter = w_geom.intersection(mask_union)
            if inter.is_empty:
                continue
            cover = inter.area / w_geom.area if w_geom.area > 0 else 0.0
            if cover < min_cover_frac:
                continue
        kept.append(w)
    return kept


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
        for _, label_row in group.iterrows():
            cx, cy = label_row.geometry.centroid.x, label_row.geometry.centroid.y
            inv = ~transform
            col, row_px = inv * (cx, cy)
            # apply random jitter in pixel space
            jx = random.randint(-jitter, jitter)
            jy = random.randint(-jitter, jitter)
            col = int(col + jx)
            row_px = int(row_px + jy)
            # compute top-left corner for a tile centered on (col,row_px)
            col_off = max(0, min(col - tile_size // 2, width - tile_size))
            row_off = max(0, min(row_px - tile_size // 2, height - tile_size))
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
            count += 1
            if max_per_class is not None and count >= max_per_class:
                break
    # deduplicate windows while preserving list
    unique = {(w.col_off, w.row_off): w for w in windows}
    return list(unique.values())


def adjust_boxes_to_global(boxes, window):
    # convert window-local pixel boxes to global pixel boxes
    if boxes.numel() == 0:
        return boxes
    offset = boxes.new_tensor([window.col_off, window.row_off, window.col_off, window.row_off])
    return boxes + offset

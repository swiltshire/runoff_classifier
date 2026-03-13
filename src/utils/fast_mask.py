# fast_mask.py
from __future__ import annotations
from typing import List
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

def rasterize_mask_aligned(raster_path: str, mask_path: str) -> np.ndarray:
    gdf = gpd.read_file(mask_path)
    with rasterio.open(raster_path) as src:
        dst_crs = src.crs
        height, width = src.height, src.width
        transform = src.transform
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True  # set True to be more inclusive
    )
    return mask  # 0/1 array aligned to the raster

def filter_windows_by_mask_raster(
    mask: np.ndarray,
    windows: List[Window],
    min_cover_frac: float = 0.0
) -> List[Window]:
    kept = []
    area = None
    if min_cover_frac > 0.0:
        area = None  # computed per-window
    for w in windows:
        r0 = int(w.row_off); r1 = int(w.row_off + w.height)
        c0 = int(w.col_off); c1 = int(w.col_off + w.width)
        chip = mask[r0:r1, c0:c1]
        if chip.size == 0:
            continue
        cover = float(chip.sum()) / float(chip.size)
        if cover >= min_cover_frac:
            kept.append(w)
    return kept
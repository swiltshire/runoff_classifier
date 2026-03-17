# src/utils/fast_mask.py
from __future__ import annotations
import os
import time
import hashlib
from typing import Iterable, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.ops import unary_union

def _cache_key(raster_path: str, mask_path: str, crs_wkt: str, height: int, width: int, transform) -> str:
    # build a simple cache key from file mtimes + geometry
    try:
        r_mtime = os.path.getmtime(raster_path)
    except Exception:
        r_mtime = 0.0
    try:
        m_mtime = os.path.getmtime(mask_path)
    except Exception:
        m_mtime = 0.0
    blob = f"{raster_path}|{r_mtime}|{mask_path}|{m_mtime}|{crs_wkt}|{height}|{width}|{tuple(transform)}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def rasterize_mask_aligned(
    raster_path: str,
    mask_path: str,
    *,
    all_touched: bool = True,
    simplify_tol_px: float = 0.0,   # e.g., 0.25..1.0: simplify geometries at ~pixel scale
    block_px: int = 2048,          # tile size for windowed rasterization
    use_cache: bool = True,
    cache_dir: str | None = None,
) -> np.ndarray:
    """
    fast rasterizer: prefilters geometries to raster extent, optionally simplifies to pixel scale,
    rasterizes in windows using a spatial index, merges into a single uint8 mask, and caches result.

    returns: uint8 mask with shape (height, width) aligned to raster pixel grid.
    """
    gdf = gpd.read_file(mask_path)

    # open raster once; collect geometry
    with rasterio.open(raster_path) as src:
        dst_crs = src.crs
        height, width = src.height, src.width
        transform = src.transform
        crs_wkt = dst_crs.to_wkt() if dst_crs else ""
        # raster bounds in map units
        left, bottom, right, top = src.bounds
        raster_bounds_poly = box(left, bottom, right, top)
        # pixel size (abs for y)
        px_w = transform.a
        px_h = -transform.e if transform.e < 0 else transform.e

    # optional: cache
    if use_cache:
        cache_dir = cache_dir or (os.path.dirname(raster_path) or ".")
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(raster_path, mask_path, crs_wkt, height, width, transform)
        cache_path = os.path.join(cache_dir, f"mask_{key}.npy")
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path, mmap_mode=None)
            except Exception:
                pass  # fall through to rebuild

    # reproject mask to raster crs if needed
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)

    # quick prefilter: keep only geoms intersecting raster extent
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.intersects(raster_bounds_poly)].copy()
    if len(gdf) == 0:
        mask = np.zeros((height, width), dtype=np.uint8)
        if use_cache:
            np.save(cache_path, mask)
        return mask

    # optional simplify at pixel scale (reduces vertex counts drastically on dense masks)
    if simplify_tol_px and (simplify_tol_px > 0):
        tol_units = max(px_w, px_h) * float(simplify_tol_px)
        gdf["geometry"] = gdf.geometry.simplify(tol_units, preserve_topology=True)

    # spatial index to get only relevant geoms per window
    sindex = gdf.sindex

    # allocate output mask once
    out = np.zeros((height, width), dtype=np.uint8)

    # iterate windows in blocks (windowed rasterization)
    # use strides of block_px; ensure we include right/bottom edges
    for r0 in range(0, height, block_px):
        r1 = min(height, r0 + block_px)
        for c0 in range(0, width, block_px):
            c1 = min(width, c0 + block_px)
            if r1 <= r0 or c1 <= c0:
                continue

            # compute window bounds in map units
            # pixel -> map: (col,row) via affine
            x0, y0 = transform * (c0, r0)
            x1, y1 = transform * (c1, r1)
            w_left, w_right = sorted([x0, x1])
            w_bottom, w_top = sorted([y0, y1])
            win_poly = box(w_left, w_bottom, w_right, w_top)

            # query sindex for candidates; filter precisely by intersects
            cand_idx = list(sindex.intersection((w_left, w_bottom, w_right, w_top)))
            if not cand_idx:
                continue
            sub = gdf.iloc[cand_idx]
            sub = sub[sub.geometry.intersects(win_poly)]

            if len(sub) == 0:
                continue

            # rasterize only this subset into a small chip, then write back
            chip = rasterize(
                [(geom, 1) for geom in sub.geometry if geom and not geom.is_empty],
                out_shape=(r1 - r0, c1 - c0),
                transform=rasterio.windows.transform(
                    ((r0, r1), (c0, c1)), transform
                ),
                fill=0,
                dtype="uint8",
                all_touched=all_touched,
            )
            # merge into output (max is fine for binary)
            out[r0:r1, c0:c1] = np.maximum(out[r0:r1, c0:c1], chip)

    if use_cache:
        np.save(cache_path, out)
    return out
    

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
# src/utils/fast_mask.py
from __future__ import annotations
import os
import hashlib
from typing import Optional, List

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box



def _cache_key(
    raster_path: str,
    mask_path: str,
    height: int,
    width: int,
    transform,
    crs_wkt: str,
    all_touched: bool,
    simplify_tol_px: float,
) -> str:
    # build a simple stable key from file mtimes and raster geometry + settings
    try:
        r_mtime = os.path.getmtime(raster_path)
    except Exception:
        r_mtime = 0.0
    try:
        m_mtime = os.path.getmtime(mask_path)
    except Exception:
        m_mtime = 0.0
    blob = "|".join(
        [
            raster_path,
            str(r_mtime),
            mask_path,
            str(m_mtime),
            str(height),
            str(width),
            ",".join(map(str, transform)),
            crs_wkt or "",
            f"touched={int(bool(all_touched))}",
            f"simplify={simplify_tol_px:.6f}",
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ensure_mask_npy(
    raster_path: str,
    mask_path: str,
    *,
    cache_dir: Optional[str] = None,
    all_touched: bool = True,
    simplify_tol_px: float = 0.0,  # 0 = off; try 0.25..1.0 to simplify at pixel scale
    overwrite: bool = False,
) -> str:
    """
    create (or reuse) a cached uint8 mask .npy aligned to the raster grid.
    returns the .npy path. use np.load(path, mmap_mode='r') to read quickly.

    strategy:
      - open raster once, grab (height, width, transform, crs)
      - read mask shapefile, reproject if needed
      - clip to raster bounds (cheap)
      - optional simplify at pixel scale
      - one-shot rasterize over full extent
      - save to .npy and return its path
    """
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        dst_crs = src.crs
        crs_wkt = dst_crs.to_wkt() if dst_crs else ""
        left, bottom, right, top = src.bounds
        raster_poly = box(left, bottom, right, top)

    # cache path
    cache_dir = cache_dir or (os.path.dirname(raster_path) or ".")
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(
        raster_path, mask_path, height, width, transform, crs_wkt, all_touched, simplify_tol_px
    )
    npy_path = os.path.join(cache_dir, f"mask_{key}.npy")

    if os.path.exists(npy_path) and not overwrite:
        return npy_path

    # read + reproject + clip
    gdf = gpd.read_file(mask_path)
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)

    # drop empties and clip to raster bounds (fast)
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.intersects(raster_poly)]
    if len(gdf) == 0:
        mask = np.zeros((height, width), dtype=np.uint8)
        np.save(npy_path, mask)  # cache empty
        return npy_path

    # optional simplify at the pixel scale to cut vertex counts
    if simplify_tol_px and simplify_tol_px > 0:
        px_w = transform.a
        px_h = -transform.e if transform.e < 0 else transform.e
        tol_units = max(px_w, px_h) * float(simplify_tol_px)
        # preserve_topology avoids slivers disappearing
        gdf["geometry"] = gdf.geometry.simplify(tol_units, preserve_topology=True)
        # re-filter any empties that might result from simplify
        gdf = gdf[~gdf.geometry.is_empty]

    # one-shot rasterize (fast path, predictable)
    shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    np.save(npy_path, mask)
    return npy_path


def load_mask_cached(npy_path: str, mmap: bool = True) -> np.ndarray:
    """load the cached mask; memory-map by default for fast, low-ram reads."""
    if mmap:
        return np.load(npy_path, mmap_mode="r")
    return np.load(npy_path)


def get_mask(
    raster_path: str,
    mask_path: str,
    *,
    cache_dir: Optional[str] = None,
    all_touched: bool = True,
    simplify_tol_px: float = 0.0,
    overwrite: bool = False,
    mmap: bool = True,
) -> np.ndarray:
    """
    convenience wrapper: build/reuse the cache and return the mask ndarray.
    """
    npy = ensure_mask_npy(
        raster_path=raster_path,
        mask_path=mask_path,
        cache_dir=cache_dir,
        all_touched=all_touched,
        simplify_tol_px=simplify_tol_px,
        overwrite=overwrite,
    )
    return load_mask_cached(npy, mmap=mmap)





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


# may be able to speed this up using:

# def filter_windows_by_mask_raster(
#     mask: np.ndarray,
#     windows: List[Window],
#     min_cover_frac: float = 0.0
# ) -> List[Window]:
#     # make sure mask is 0/1 uint8 for fast sums
#     if mask.dtype != np.uint8:
#         mask = (mask > 0).astype(np.uint8, copy=False)

#     # summed-area table with a one-pixel zero border
#     i = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.uint32)
#     i[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0), axis=1)

#     kept = []
#     for w in windows:
#         r0 = int(w.row_off); r1 = int(w.row_off + w.height)
#         c0 = int(w.col_off); c1 = int(w.col_off + w.width)
#         if r1 <= r0 or c1 <= c0:
#             continue

#         win_sum = int(i[r1, c1] - i[r0, c1] - i[r1, c0] + i[r0, c0])
#         if min_cover_frac <= 0.0:
#             if win_sum > 0:
#                 kept.append(w)
#             continue

#         area = (r1 - r0) * (c1 - c0)
#         if area <= 0:
#             continue

#         cover = win_sum / float(area)
#         if cover >= min_cover_frac:
#             kept.append(w)

#     return kept

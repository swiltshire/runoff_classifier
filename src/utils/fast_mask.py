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



# def _cache_key(
#     raster_path: str,
#     mask_path: str,
#     height: int,
#     width: int,
#     transform,
#     crs_wkt: str,
#     all_touched: bool,
#     simplify_tol_px: float,
# ) -> str:
#     # build a simple stable key from file mtimes and raster geometry + settings
#     try:
#         r_mtime = os.path.getmtime(raster_path)
#     except Exception:
#         r_mtime = 0.0
#     try:
#         m_mtime = os.path.getmtime(mask_path)
#     except Exception:
#         m_mtime = 0.0
#     blob = "|".join(
#         [
#             raster_path,
#             str(r_mtime),
#             mask_path,
#             str(m_mtime),
#             str(height),
#             str(width),
#             ",".join(map(str, transform)),
#             crs_wkt or "",
#             f"touched={int(bool(all_touched))}",
#             f"simplify={simplify_tol_px:.6f}",
#         ]
#     )
#     return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# def ensure_mask_npy(
#     raster_path: str,
#     mask_path: str,
#     *,
#     cache_dir: Optional[str] = None,
#     all_touched: bool = True,
#     simplify_tol_px: float = 0.0,  # 0 = off; try 0.25..1.0 to simplify at pixel scale
#     overwrite: bool = False,
# ) -> str:
#     """
#     create (or reuse) a cached uint8 mask .npy aligned to the raster grid.
#     returns the .npy path. use np.load(path, mmap_mode='r') to read quickly.

#     strategy:
#       - open raster once, grab (height, width, transform, crs)
#       - read mask shapefile, reproject if needed
#       - clip to raster bounds (cheap)
#       - optional simplify at pixel scale
#       - one-shot rasterize over full extent
#       - save to .npy and return its path
#     """
#     with rasterio.open(raster_path) as src:
#         height, width = src.height, src.width
#         transform = src.transform
#         dst_crs = src.crs
#         crs_wkt = dst_crs.to_wkt() if dst_crs else ""
#         left, bottom, right, top = src.bounds
#         raster_poly = box(left, bottom, right, top)

#     # cache path
#     cache_dir = cache_dir or (os.path.dirname(raster_path) or ".")
#     os.makedirs(cache_dir, exist_ok=True)
#     key = _cache_key(
#         raster_path, mask_path, height, width, transform, crs_wkt, all_touched, simplify_tol_px
#     )
#     npy_path = os.path.join(cache_dir, f"mask_{key}.npy")

#     if os.path.exists(npy_path) and not overwrite:
#         return npy_path

#     # read + reproject + clip
#     gdf = gpd.read_file(mask_path)
#     if gdf.crs != dst_crs:
#         gdf = gdf.to_crs(dst_crs)

#     # drop empties and clip to raster bounds (fast)
#     gdf = gdf[~gdf.geometry.is_empty]
#     gdf = gdf[gdf.geometry.intersects(raster_poly)]
#     if len(gdf) == 0:
#         mask = np.zeros((height, width), dtype=np.uint8)
#         np.save(npy_path, mask)  # cache empty
#         return npy_path

#     # optional simplify at the pixel scale to cut vertex counts
#     if simplify_tol_px and simplify_tol_px > 0:
#         px_w = transform.a
#         px_h = -transform.e if transform.e < 0 else transform.e
#         tol_units = max(px_w, px_h) * float(simplify_tol_px)
#         # preserve_topology avoids slivers disappearing
#         gdf["geometry"] = gdf.geometry.simplify(tol_units, preserve_topology=True)
#         # re-filter any empties that might result from simplify
#         gdf = gdf[~gdf.geometry.is_empty]

#     # one-shot rasterize (fast path, predictable)
#     shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
#     mask = rasterize(
#         shapes,
#         out_shape=(height, width),
#         transform=transform,
#         fill=0,
#         dtype="uint8",
#         all_touched=all_touched,
#     )
#     np.save(npy_path, mask)
#     return npy_path


# def load_mask_cached(npy_path: str, mmap: bool = True) -> np.ndarray:
#     """load the cached mask; memory-map by default for fast, low-ram reads."""
#     if mmap:
#         return np.load(npy_path, mmap_mode="r")
#     return np.load(npy_path)



# def get_mask(
#     raster_path: str,
#     mask_path: str,
#     cache_dir: str,
#     simplify_tol_px: float = 0.5,
#     downsample: int = 1,
#     load_only: bool = False,
# ):
#     """
#     Rasterize polygon mask aligned to raster grid (optionally downsampled),
#     and cache as .npy. Subsequent calls load from cache.

#     Parameters
#     ----------
#     raster_path : str
#         Path to raster or VRT
#     mask_path : str
#         Path to polygon mask vector
#     cache_dir : str
#         Where to store cached mask
#     simplify_tol_px : float
#         Polygon simplification tolerance in pixel units
#     downsample : int
#         Integer downsampling factor (e.g. 8 → 1/8 resolution)
#     load_only : bool
#         If True, only load cached mask; error if missing
#     """

#     os.makedirs(cache_dir, exist_ok=True)

#     base = os.path.splitext(os.path.basename(raster_path))[0]
#     suffix = f"_mask_ds{downsample}.npy"
#     cache_path = os.path.join(cache_dir, base + suffix)

#     # ---- fast path: cached ----
#     if os.path.exists(cache_path):
#         return np.load(cache_path, mmap_mode="r")

#     if load_only:
#         raise FileNotFoundError(f"cached mask not found: {cache_path}")

#     # ---- build mask ----
#     with rasterio.open(raster_path) as src:
#         height = src.height // downsample
#         width = src.width // downsample

#         # scale transform for downsampling
#         transform = src.transform * src.transform.scale(downsample, downsample)
#         crs = src.crs

#     gdf = gpd.read_file(mask_path)
#     if gdf.crs != crs:
#         gdf = gdf.to_crs(crs)

#     if simplify_tol_px and simplify_tol_px > 0:
#         gdf["geometry"] = gdf.geometry.simplify(
#             simplify_tol_px * downsample,
#             preserve_topology=True,
#         )

#     shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]

#     mask = rio_features.rasterize(
#         shapes,
#         out_shape=(height, width),
#         transform=transform,
#         fill=0,
#         dtype="uint8",
#     )

#     np.save(cache_path, mask)
#     return mask


def get_mask_clipped(
    raster_path: str,
    mask_path: str,
    cache_dir: str,
    *,
    downsample: int = 8,
    simplify_tol_px: float = 0.5,
    load_only: bool = False,
):
    """
    AOI‑clipped, downsampled polygon mask rasterization.

    Returns
    -------
    mask : np.ndarray (uint8)
        Downsampled raster mask (AOI bbox only)
    meta : dict
        {
          "row0": int,  # AOI top‑left pixel (full‑res)
          "col0": int,
          "downsample": int,
        }
    """

    os.makedirs(cache_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(raster_path))[0]
    tag = f"_mask_ds{downsample}_clipped"
    mask_path_npy = os.path.join(cache_dir, base + tag + ".npy")
    meta_path_json = os.path.join(cache_dir, base + tag + ".json")

    # ---------- load cached ----------
    if os.path.exists(mask_path_npy) and os.path.exists(meta_path_json):
        mask = np.load(mask_path_npy, mmap_mode="r")
        with open(meta_path_json, "r") as f:
            meta = json.load(f)
        return mask, meta

    if load_only:
        raise FileNotFoundError("AOI‑clipped mask cache missing")

    # ---------- open raster ----------
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs
        H, W = src.height, src.width

    # ---------- load + reproject AOI ----------
    gdf = gpd.read_file(mask_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    if simplify_tol_px and simplify_tol_px > 0:
        gdf["geometry"] = gdf.geometry.simplify(
            simplify_tol_px * downsample,
            preserve_topology=True,
        )

    # ---------- union geometry ----------
    geom = gdf.unary_union
    if geom.is_empty:
        raise ValueError("AOI geometry is empty after union")

    # ---------- AOI bounds in pixel coords ----------
    minx, miny, maxx, maxy = geom.bounds
    inv = ~transform

    c0, r1 = inv * (minx, miny)
    c1, r0 = inv * (maxx, maxy)

    r0, r1 = sorted([int(r0), int(r1)])
    c0, c1 = sorted([int(c0), int(c1)])

    # clip to raster
    r0 = max(0, min(H, r0))
    r1 = max(0, min(H, r1))
    c0 = max(0, min(W, c0))
    c1 = max(0, min(W, c1))

    if r1 <= r0 or c1 <= c0:
        raise ValueError("AOI outside raster extent")

    # ---------- downsampled window ----------
    out_h = max(1, (r1 - r0) // downsample)
    out_w = max(1, (c1 - c0) // downsample)

    # adjust transform
    sub_transform = (
        transform
        * rasterio.Affine.translation(c0, r0)
        * rasterio.Affine.scale(downsample, downsample)
    )

    # ---------- rasterize ----------
    mask = rio_features.rasterize(
        [(geom, 1)],
        out_shape=(out_h, out_w),
        transform=sub_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    # ---------- cache ----------
    meta = dict(
        row0=int(r0),
        col0=int(c0),
        downsample=int(downsample),
    )

    np.save(mask_path_npy, mask)
    with open(meta_path_json, "w") as f:
        json.dump(meta, f)

    return mask, meta





def filter_windows_by_mask_raster(
    mask: np.ndarray,
    windows: List[Window],
    min_cover_frac: float = 0.0,
    *,
    row0: int,
    col0: int,
    downsample: int,
) -> List[Window]:
    """
    Filter windows using an AOI-clipped, downsampled raster mask.

    Parameters
    ----------
    mask : np.ndarray
        Downsampled mask raster (AOI bbox only), shape (H_ds, W_ds)
    windows : list[Window]
        Raster windows in full-resolution pixel coordinates
    min_cover_frac : float
        Minimum fraction [0..1] of mask coverage required to keep a window
    row0, col0 : int
        Top-left pixel (full-resolution) of the AOI mask bounding box
    downsample : int
        Downsampling factor used to create the mask

    Returns
    -------
    list[Window]
        Windows that meet the mask coverage criterion
    """

    kept: List[Window] = []
    Hm, Wm = mask.shape

    for w in windows:
        # full-resolution window pixel bounds
        r0 = int(w.row_off)
        r1 = int(w.row_off + w.height)
        c0 = int(w.col_off)
        c1 = int(w.col_off + w.width)

        # map to mask (downsampled) coordinates
        mr0 = (r0 - row0) // downsample
        mr1 = (r1 - row0) // downsample
        mc0 = (c0 - col0) // downsample
        mc1 = (c1 - col0) // downsample

        # clip to mask array bounds
        mr0 = max(0, min(Hm, mr0))
        mr1 = max(0, min(Hm, mr1))
        mc0 = max(0, min(Wm, mc0))
        mc1 = max(0, min(Wm, mc1))

        if mr1 <= mr0 or mc1 <= mc0:
            # window lies completely outside AOI mask bbox
            continue

        chip = mask[mr0:mr1, mc0:mc1]
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

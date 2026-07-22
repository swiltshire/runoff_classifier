# src/utils/fast_mask.py
from __future__ import annotations
import os
import json
import time
import logging
import hashlib
from typing import Optional, List
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features as rio_features
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box


def clear_mask_cache(raster_path: str, cache_dir: str, downsample: int = 16):
    """
    Clear cached AOI mask files for a given raster.
    Called when VRT is regenerated to prevent spatial misalignment.
    """
    logger = logging.getLogger("mask_raster")
    base = os.path.splitext(os.path.basename(raster_path))[0]
    tag = f"_mask_ds{downsample}_clipped"
    mask_path_npy = os.path.join(cache_dir, base + tag + ".npy")
    meta_path_json = os.path.join(cache_dir, base + tag + ".json")
    
    cleared = False
    if os.path.exists(mask_path_npy):
        os.remove(mask_path_npy)
        cleared = True
    if os.path.exists(meta_path_json):
        os.remove(meta_path_json)
        cleared = True
    
    if cleared:
        logger.info("[mask] cleared cache for %s", base)


def get_mask_clipped(
    raster_path: str,
    mask_path: str,
    cache_dir: str,
    *,
    downsample: int = 16,
    load_only: bool = False,
):
    """
    AOI-clipped, downsampled polygon mask rasterization with progress logging.
    """

    logger = logging.getLogger("mask_raster")
    logger.propagate = True

    os.makedirs(cache_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(raster_path))[0]
    tag = f"_mask_ds{downsample}_clipped"
    mask_path_npy = os.path.join(cache_dir, base + tag + ".npy")
    meta_path_json = os.path.join(cache_dir, base + tag + ".json")

    # ---------- load cached ----------
    if os.path.exists(mask_path_npy) and os.path.exists(meta_path_json):
        logger.info("[mask] loading cached mask %s", mask_path_npy)
        mask = np.load(mask_path_npy, mmap_mode="r")
        with open(meta_path_json, "r") as f:
            meta = json.load(f)
        return mask, meta

    if load_only:
        raise FileNotFoundError(f"cached mask not found: {mask_path_npy}")

    t_start = time.time()
    logger.info("[mask] starting AOI mask build (downsample=%dx)", downsample)

    # ---------- open raster ----------
    with rasterio.open(raster_path) as src:
        raster_transform = src.transform
        raster_crs = src.crs
        H, W = src.height, src.width

    logger.info("[mask] raster opened (size=%dx%d)", W, H)

    # ---------- load AOI polygons ----------
    t0 = time.time()
    gdf = gpd.read_file(mask_path)
    logger.info("[mask] loaded %d AOI features in %.2fs", len(gdf), time.time() - t0)

    if gdf.crs != raster_crs:
        t0 = time.time()
        gdf = gdf.to_crs(raster_crs)
        logger.info("[mask] reprojected AOI to raster CRS in %.2fs", time.time() - t0)

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    logger.info("[mask] AOI features after clean: %d", len(gdf))

    # ---------- AOI bounds (CHEAP, no union) ----------
    minx, miny, maxx, maxy = gdf.total_bounds
    inv = ~raster_transform

    c0, r1 = inv * (minx, miny)
    c1, r0 = inv * (maxx, maxy)

    r0, r1 = sorted((int(r0), int(r1)))
    c0, c1 = sorted((int(c0), int(c1)))

    r0 = max(0, min(H, r0))
    r1 = max(0, min(H, r1))
    c0 = max(0, min(W, c0))
    c1 = max(0, min(W, c1))

    logger.info(
        "[mask] AOI pixel bounds rows=(%d:%d), cols=(%d:%d)",
        r0, r1, c0, c1
    )

    if r1 <= r0 or c1 <= c0:
        raise ValueError("AOI lies outside raster extent")

    # ---------- downsampled grid ----------
    out_h = max(1, (r1 - r0) // downsample)
    out_w = max(1, (c1 - c0) // downsample)

    sub_transform = (
        raster_transform
        * rasterio.Affine.translation(c0, r0)
        * rasterio.Affine.scale(downsample, downsample)
    )

    logger.info(
        "[mask] rasterizing AOI → grid %dx%d (downsample=%dx)",
        out_w, out_h, downsample
    )

    # ---------- rasterization with geometry progress ----------
    def shape_generator():
        n = len(gdf)
        log_every = max(1, n // 10)  # ~10 progress messages
        for i, geom in enumerate(gdf.geometry, start=1):
            if i % log_every == 0 or i == n:
                logger.info("[mask] rasterizing geometry %d / %d", i, n)
            yield (geom, 1)

    t0 = time.time()
    mask = rio_features.rasterize(
        shape_generator(),
        out_shape=(out_h, out_w),
        transform=sub_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    logger.info("[mask] rasterization completed in %.2fs", time.time() - t0)

    # ---------- cache ----------
    meta = {
        "row0": int(r0),
        "col0": int(c0),
        "downsample": int(downsample),
    }

    np.save(mask_path_npy, mask)
    with open(meta_path_json, "w") as f:
        json.dump(meta, f)

    logger.info(
        "[mask] cached AOI mask → %s (total time %.2fs)",
        mask_path_npy, time.time() - t_start
    )

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


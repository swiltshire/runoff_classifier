# src/utils/blank_diagnostics.py
"""
Diagnostic helpers for investigating blank/NoData imagery regions and whether
they correlate with model detections or training labels.

Not part of the main train/inference pipeline - these are one-off investigation
tools (e.g. the Carroll County "blacked-out tiles with detections in them"
question), meant to be called from notebook cells or ad-hoc scripts.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import box


def _read_window_stats(src: rasterio.io.DatasetReader, bounds, min_size_px: int = 3):
    """Read the pixel window under `bounds` (a geometry's bounding box in the
    raster's own CRS), padding tiny windows up to at least `min_size_px` on a
    side so very small point-like features still sample a few pixels. Returns
    a (band array (bands, rows, cols), window) tuple, or None if the bounds
    don't overlap the raster at all.
    """
    full = Window(0, 0, src.width, src.height)
    try:
        win = from_bounds(*bounds, transform=src.transform)
    except Exception:
        return None
    win = win.round_offsets().round_lengths()
    win = win.intersection(full) if _windows_overlap(win, full) else None
    if win is None or win.width < 1 or win.height < 1:
        return None
    if win.width < min_size_px or win.height < min_size_px:
        cx = win.col_off + win.width / 2.0
        cy = win.row_off + win.height / 2.0
        half = min_size_px / 2.0
        padded = Window(cx - half, cy - half, min_size_px, min_size_px)
        if not _windows_overlap(padded, full):
            return None
        win = padded.intersection(full)
    if win.width < 1 or win.height < 1:
        return None
    return src.read(window=win), win


def _windows_overlap(a: Window, b: Window) -> bool:
    return (
        a.col_off < b.col_off + b.width
        and a.col_off + a.width > b.col_off
        and a.row_off < b.row_off + b.height
        and a.row_off + a.height > b.row_off
    )


def classify_geometries_by_raster_content(
    raster_path: str,
    gdf: gpd.GeoDataFrame,
    blank_frac_thresh: float = 0.9,
    min_window_px: int = 3,
) -> gpd.GeoDataFrame:
    """For each geometry in `gdf`, sample the underlying pixels from
    `raster_path` and classify it based on the fraction of NoData/zero pixels
    under the geometry's own footprint (rasterized to the pixel grid, NOT
    just its bounding box - important for irregular/elongated polygons, e.g.
    Mask R-CNN instance segmentation outputs, where the bounding box can
    include a lot of area outside the actual detected shape):
      - "real_imagery": pct_blank <= (1 - blank_frac_thresh)
      - "blank_nodata": pct_blank >= blank_frac_thresh
      - "mixed": in between
      - "outside_raster": bounding box doesn't overlap the raster at all
      - "invalid_geometry": null/empty geometry

    "Blank" pixels are defined as: pixels equal to the raster's NoData value
    if one is set (`src.nodata`), otherwise pixels that are exactly zero
    across all bands (the common case for imagery that was never tagged with
    a real NoData value but is just zero-filled outside real coverage).

    Adds columns: pct_blank (polygon-footprint-based, used for
    pixel_category), pct_blank_bbox (the previous bounding-box-based
    figure, kept for before/after comparison), pixel_mean, pixel_std
    (both polygon-footprint-based), pixel_category.
    Returns a NEW GeoDataFrame (copy) - does not mutate the input.
    """
    out = gdf.copy().reset_index(drop=True)

    pct_blank_col: List[Optional[float]] = []
    pct_blank_bbox_col: List[Optional[float]] = []
    mean_col: List[Optional[float]] = []
    std_col: List[Optional[float]] = []
    category_col: List[str] = []

    with rasterio.open(raster_path) as src:
        if out.crs is not None and src.crs is not None and str(out.crs) != str(src.crs):
            out = out.to_crs(src.crs)
        nodata_val = src.nodata

        for geom in out.geometry:
            if geom is None or geom.is_empty:
                pct_blank_col.append(None)
                pct_blank_bbox_col.append(None)
                mean_col.append(None)
                std_col.append(None)
                category_col.append("invalid_geometry")
                continue

            result = _read_window_stats(src, geom.bounds, min_size_px=min_window_px)
            if result is None:
                pct_blank_col.append(None)
                pct_blank_bbox_col.append(None)
                mean_col.append(None)
                std_col.append(None)
                category_col.append("outside_raster")
                continue
            arr, win = result
            if arr is None or arr.size == 0:
                pct_blank_col.append(None)
                pct_blank_bbox_col.append(None)
                mean_col.append(None)
                std_col.append(None)
                category_col.append("outside_raster")
                continue

            if nodata_val is not None:
                blank_mask = np.all(arr == nodata_val, axis=0)
            else:
                blank_mask = np.all(arr == 0, axis=0)

            pct_blank_bbox = float(blank_mask.mean())
            pct_blank_bbox_col.append(pct_blank_bbox)

            # Restrict to pixels actually inside the geometry's own footprint
            # (not just its bounding box) - rasterize the geometry onto this
            # window's pixel grid. Falls back to the full bbox if the
            # geometry is too thin/small to rasterize to any pixel (rare
            # edge case, e.g. a sub-pixel sliver).
            win_transform = src.window_transform(win)
            footprint_mask = geometry_mask(
                [geom], out_shape=blank_mask.shape, transform=win_transform, invert=True
            )
            if not footprint_mask.any():
                footprint_mask = np.ones_like(blank_mask, dtype=bool)

            pct_blank = float(blank_mask[footprint_mask].mean())
            pct_blank_col.append(pct_blank)
            footprint_arr = arr[:, footprint_mask]
            mean_col.append(float(footprint_arr.mean()))
            std_col.append(float(footprint_arr.std()))

            if pct_blank >= blank_frac_thresh:
                category_col.append("blank_nodata")
            elif pct_blank <= (1.0 - blank_frac_thresh):
                category_col.append("real_imagery")
            else:
                category_col.append("mixed")

    out["pct_blank"] = pct_blank_col
    out["pct_blank_bbox"] = pct_blank_bbox_col
    out["pixel_mean"] = mean_col
    out["pixel_std"] = std_col
    out["pixel_category"] = category_col
    return out


def summarize_categories(gdf: gpd.GeoDataFrame, label: str = "features") -> None:
    """Print a quick count/percentage breakdown of the `pixel_category`
    column produced by `classify_geometries_by_raster_content`.
    """
    total = len(gdf)
    print(f"\n{label}: {total} total")
    if total == 0 or "pixel_category" not in gdf.columns:
        return
    counts = gdf["pixel_category"].value_counts(dropna=False)
    for cat, n in counts.items():
        print(f"  {cat}: {n} ({100.0 * n / total:.1f}%)")


def find_corrupted_tiles(tile_paths: List[str], max_workers: int = 16) -> List[Tuple[str, str]]:
    """Attempt a full-resolution read of every tile in `tile_paths` (in
    parallel - this is I/O-bound) and return a list of `(tile_path,
    error_message)` for any that fail to read - e.g. a truncated/corrupted
    GeoTIFF from an interrupted write (observed after a large parallel
    force-rebuild: `TIFFReadEncodedTile() failed` / garbage row/col offsets).

    This is a pure read-integrity check - it does NOT classify blank vs real
    content (see `per_tile_blank_summary` for that, which uses a fast
    decimated read and would not reliably catch this kind of corruption).
    """
    def _check(p: str) -> Optional[Tuple[str, str]]:
        try:
            with rasterio.open(p) as src:
                src.read()
            return None
        except Exception as e:
            return (p, str(e))

    bad: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_check, p) for p in tile_paths]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                bad.append(result)
    return bad


def per_tile_blank_summary(tile_paths: List[str], downsample: int = 8) -> gpd.GeoDataFrame:
    """For each canonical tile in `tile_paths`, compute the overall fraction of
    blank/NoData pixels (reading a decimated/downsampled overview for speed,
    not the full-resolution tile) and return one row per tile with its bounding
    box as geometry - suitable for a county-wide "% blank per tile" overview
    plot.

    `downsample`: read each tile at roughly 1/downsample of its native
    resolution (e.g. 8 = an 8x-decimated read) to keep this fast even for
    large 4096x4096 chips.
    """
    rows = []
    crs = None
    for p in tile_paths:
        with rasterio.open(p) as src:
            crs = src.crs
            out_h = max(1, src.height // downsample)
            out_w = max(1, src.width // downsample)
            arr = src.read(out_shape=(src.count, out_h, out_w))
            nodata_val = src.nodata
            if nodata_val is not None:
                blank_mask = np.all(arr == nodata_val, axis=0)
            else:
                blank_mask = np.all(arr == 0, axis=0)
            pct_blank = float(blank_mask.mean())
            geom = box(*src.bounds)
        rows.append(
            {
                "tile_path": p,
                "tile_name": os.path.basename(p),
                "pct_blank": pct_blank,
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, crs=crs)

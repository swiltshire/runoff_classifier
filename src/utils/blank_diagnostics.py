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
from typing import List, Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import box


def _read_window_stats(src: rasterio.io.DatasetReader, bounds, min_size_px: int = 3):
    """Read the pixel window under `bounds` (a geometry's bounding box in the
    raster's own CRS), padding tiny windows up to at least `min_size_px` on a
    side so very small point-like features still sample a few pixels. Returns
    the raw band array (bands, rows, cols), or None if the bounds don't
    overlap the raster at all.
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
    return src.read(window=win)


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
    """For each geometry in `gdf`, sample the underlying pixel window from
    `raster_path` and classify it based on the fraction of NoData/zero pixels
    under the geometry's bounding box:
      - "real_imagery": pct_blank <= (1 - blank_frac_thresh)
      - "blank_nodata": pct_blank >= blank_frac_thresh
      - "mixed": in between
      - "outside_raster": bounding box doesn't overlap the raster at all
      - "invalid_geometry": null/empty geometry

    "Blank" pixels are defined as: pixels equal to the raster's NoData value
    if one is set (`src.nodata`), otherwise pixels that are exactly zero
    across all bands (the common case for imagery that was never tagged with
    a real NoData value but is just zero-filled outside real coverage).

    Adds columns: pct_blank, pixel_mean, pixel_std, pixel_category.
    Returns a NEW GeoDataFrame (copy) - does not mutate the input.
    """
    out = gdf.copy().reset_index(drop=True)

    pct_blank_col: List[Optional[float]] = []
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
                mean_col.append(None)
                std_col.append(None)
                category_col.append("invalid_geometry")
                continue

            arr = _read_window_stats(src, geom.bounds, min_size_px=min_window_px)
            if arr is None or arr.size == 0:
                pct_blank_col.append(None)
                mean_col.append(None)
                std_col.append(None)
                category_col.append("outside_raster")
                continue

            if nodata_val is not None:
                blank_mask = np.all(arr == nodata_val, axis=0)
            else:
                blank_mask = np.all(arr == 0, axis=0)

            pct_blank = float(blank_mask.mean())
            pct_blank_col.append(pct_blank)
            mean_col.append(float(arr.mean()))
            std_col.append(float(arr.std()))

            if pct_blank >= blank_frac_thresh:
                category_col.append("blank_nodata")
            elif pct_blank <= (1.0 - blank_frac_thresh):
                category_col.append("real_imagery")
            else:
                category_col.append("mixed")

    out["pct_blank"] = pct_blank_col
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

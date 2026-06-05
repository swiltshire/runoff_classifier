# scripts/prepare_multicounty_training.py

import os
import sys
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.coords import BoundingBox
from rasterio.windows import Window
from shapely.geometry import box
import matplotlib.pyplot as plt

from utils.make_vrt import write_mosaic_vrt
from utils.tiling import make_label_centered_training_windows


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _log(msg: str):
    print(f"[prepare] {msg}", flush=True)


def _fail(msg: str):
    raise RuntimeError(f"[prepare][FATAL] {msg}")


def _normalize_class_column(
    gdf: gpd.GeoDataFrame,
    county: str,
) -> gpd.GeoDataFrame:
    class_cols = [c for c in ("Classname", "classname", "Class") if c in gdf.columns]

    if len(class_cols) == 0:
        _fail(f"{county}: missing class column (expected 'Classname', 'classname', or 'Class')")

    if len(class_cols) > 1:
        _fail(f"{county}: multiple class name fields present — ambiguous schema")

    if class_cols[0] == "Class":
        gdf = gdf.rename(columns={"Class": "Classname"})
    
    if class_cols[0] == "classname":
        gdf = gdf.rename(columns={"classname": "Classname"})

    # defensive cleanup
    gdf["Classname"] = gdf["Classname"].astype(str).str.strip()

    return gdf


def _raster_bounds(raster_path: str) -> BoundingBox:
    with rasterio.open(raster_path) as ds:
        return ds.bounds


def _bounds_intersect(a: BoundingBox, b: BoundingBox) -> bool:
    return box(*a).intersects(box(*b))


def _plot_overlay(raster_path: str, gdf: gpd.GeoDataFrame, title: str):
    with rasterio.open(raster_path) as ds:
        img = ds.read([1, 2, 3])
        transform = ds.transform

    fig, ax = plt.subplots(figsize=(8, 8))
    show(img, transform=transform, ax=ax)
    gdf.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=1)
    ax.set_title(title)
    plt.show()


def window_to_dict(w: Window) -> dict:
    """Convert rasterio Window to JSON-serializable dict."""
    return {
        "col_off": w.col_off,
        "row_off": w.row_off,
        "width": w.width,
        "height": w.height,
    }


def dict_to_window(d: dict) -> Window:
    """Convert JSON dict back to rasterio Window."""
    return Window(
        col_off=d["col_off"],
        row_off=d["row_off"],
        width=d["width"],
        height=d["height"],
    )


def save_windows_json(windows: List[Window], path: str):
    """Save list of Windows to JSON file."""
    data = [window_to_dict(w) for w in windows]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(f"Saved {len(windows)} windows → {path}")


def load_windows_json(path: str) -> List[Window]:
    """Load list of Windows from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    windows = [dict_to_window(d) for d in data]
    _log(f"Loaded {len(windows)} windows from {path}")
    return windows


# -----------------------------------------------------------------------------
# Single-County Processing (for incremental workflow)
# -----------------------------------------------------------------------------

def prepare_single_county_training(
    county: str,
    shapefile_path: str,
    tiles_dir: str,
    tile_size: int = 512,
    max_per_class: Optional[int] = None,
    verified_only: bool = True,
    debug_plots: bool = False,
) -> Tuple[gpd.GeoDataFrame, List[Window], any]:
    """
    Prepare training data for a single county.
    
    Returns
    -------
    tuple
        (labels_gdf, windows_list, target_crs)
    """
    tiles_dir = Path(tiles_dir)
    shapefile_path = Path(shapefile_path)
    
    _log(f"Processing county: {county}")
    
    # -----
    # Load tiles
    # -----
    tiles = sorted(tiles_dir.glob("*.tif"))
    if not tiles:
        _fail(f"No tiles found in {tiles_dir}")
    
    _log(f"  Found {len(tiles)} tiles")
    
    # -----
    # Establish CRS from first tile
    # -----
    with rasterio.open(tiles[0]) as ds:
        target_crs = ds.crs
    
    _log(f"  CRS: {target_crs}")
    
    # -----
    # Load + validate labels
    # -----
    _log(f"  Loading labels from {shapefile_path.name}")
    
    gdf = gpd.read_file(shapefile_path, engine="pyogrio")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    
    # Normalize class column title
    gdf = _normalize_class_column(gdf, county=county)
    
    if verified_only and "VerifiedTr" in gdf.columns:
        gdf = gdf[gdf["VerifiedTr"] == 1]
    
    if gdf.empty:
        _fail(f"{county}: has zero usable labels")
    
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    # Validate overlap with tiles
    overlaps = 0
    for t in tiles:
        if _bounds_intersect(_raster_bounds(t), gdf.total_bounds):
            overlaps += 1
    
    if overlaps == 0:
        _fail(f"{county}: labels do not overlap any tiles")
    
    _log(f"  Labels overlap {overlaps}/{len(tiles)} tiles")
    
    if debug_plots:
        _plot_overlay(str(tiles[0]), gdf, title=f"{county} label alignment")
    
    gdf["county"] = county
    
    # -----
    # Build per-county VRT
    # -----
    vrt_path = tiles_dir / f"{county}_mosaic.vrt"
    if vrt_path.exists():
        vrt_path.unlink()
    
    write_mosaic_vrt(str(vrt_path), [str(t) for t in tiles])
    _log(f"  VRT built → {vrt_path.name}")
    
    # -----
    # Generate training windows
    # -----
    _log(f"  Generating training windows (max_per_class={max_per_class})")
    windows = make_label_centered_training_windows(
        str(vrt_path),
        shapefile_path,
        tile_size=tile_size,
        max_per_class=max_per_class,
    )
    _log(f"  Generated {len(windows)} windows")
    
    if "Classname" in gdf.columns:
        _log(f"  Class distribution:")
        for cls, count in gdf["Classname"].value_counts().items():
            _log(f"    {cls}: {count}")
    
    return gdf, windows, target_crs




def prepare_multicounty_training(
    county_data_dir: str,
    selected_counties: List[str],
    out_labels: str,
    out_vrt: str,
    *,
    tile_size: int = 512,
    max_per_class: Optional[int] = None,
    verified_only: bool = True,
    debug_plots: bool = False,
):
    """
    Prepare a multi-county training dataset:
      - merge shapefiles (per-county)
      - validate alignment
      - build a training VRT

    Uses incremental per-county processing for memory efficiency.

    Parameters
    ----------
    county_data_dir : str
        Root directory containing county subdirs
    selected_counties : list[str]
        Counties selected in Jupyter
    out_labels : str
        Output GPKG for merged labels
    out_vrt : str
        Output VRT path
    tile_size : int
        Training window tile size (pixels)
    max_per_class : int, optional
        Max training windows per class per county
    verified_only : bool
        Keep only VerifiedTr == 1 if the column exists
    debug_plots : bool
        Show raster/label overlay plots for each county
    """

    county_data_dir = Path(county_data_dir)
    out_labels = Path(out_labels)
    out_vrt = Path(out_vrt)

    if not selected_counties:
        _fail("No counties selected")

    _log(f"Preparing training data for {len(selected_counties)} counties (incremental)")

    # -------------------------------------------------------------------------
    # Collect shapefiles + tiles per county
    # -------------------------------------------------------------------------

    county_to_shp: Dict[str, Path] = {}
    county_to_tiles: Dict[str, Path] = {}

    for c in selected_counties:
        cdir = county_data_dir / c

        shp_files = sorted(cdir.glob("*.shp"))
        if not shp_files:
            _fail(f"No shapefile found in {cdir}")
        if len(shp_files) > 1:
            _log(f"WARNING: multiple shapefiles in {cdir}, using {shp_files[0].name}")

        tiles_dir = cdir / "tiles"
        if not tiles_dir.exists():
            _fail(f"No tiles directory in {cdir}")

        county_to_shp[c] = shp_files[0]
        county_to_tiles[c] = tiles_dir

    # -------------------------------------------------------------------------
    # Process each county incrementally
    # -------------------------------------------------------------------------

    gdfs = []
    poison_counties = []
    target_crs = None

    for c in selected_counties:
        try:
            gdf, windows, crs = prepare_single_county_training(
                county=c,
                shapefile_path=str(county_to_shp[c]),
                tiles_dir=str(county_to_tiles[c]),
                tile_size=tile_size,
                max_per_class=max_per_class,
                verified_only=verified_only,
                debug_plots=debug_plots,
            )
            
            if target_crs is None:
                target_crs = crs
            elif target_crs != crs:
                _log(f"WARNING: {c} CRS {crs} differs from target {target_crs}, reprojecting")
                gdf = gdf.to_crs(target_crs)
            
            gdfs.append(gdf)
            _log(f"✓ {c} complete: {len(gdf)} labels, {len(windows)} windows")
            
        except RuntimeError as e:
            _log(f"✗ {c} failed: {e}")
            poison_counties.append(c)
            continue

    if not gdfs:
        _fail("No valid counties left after processing")

    if poison_counties:
        _log(f"WARNING: excluded {len(poison_counties)} counties:")
        for c in poison_counties:
            _log(f"  - {c}")

    # -------------------------------------------------------------------------
    # Merge labels
    # -------------------------------------------------------------------------

    merged = pd.concat(gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=target_crs)

    _log(f"Merged labels: {len(merged)} total features from {len(gdfs)} counties")

    # Sanity: class distribution
    if "Classname" in merged.columns:
        _log("Class distribution:")
        for cls, count in merged["Classname"].value_counts().items():
            _log(f"  {cls}: {count}")

    # -------------------------------------------------------------------------
    # Write labels
    # -------------------------------------------------------------------------

    out_labels.parent.mkdir(parents=True, exist_ok=True)
    if out_labels.exists():
        out_labels.unlink()

    keep_cols = [c for c in merged.columns if c in ("Classname", "VerifiedTr", "county", "geometry")]
    merged[keep_cols].to_file(
        out_labels,
        driver="GPKG",
        layer="labels",
        engine="pyogrio",
    )

    _log(f"Wrote labels → {out_labels}")

    # -------------------------------------------------------------------------
    # Build VRT from all county tiles
    # -------------------------------------------------------------------------

    all_tiles = []
    for tiles_dir in county_to_tiles.values():
        tiles = sorted(tiles_dir.glob("*.tif"))
        all_tiles.extend(tiles)

    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    if out_vrt.exists():
        out_vrt.unlink()

    write_mosaic_vrt(str(out_vrt), [str(t) for t in all_tiles])

    with rasterio.open(out_vrt) as ds:
        _log(f"VRT built → {out_vrt}")
        _log(f"VRT bounds: {ds.bounds}")
        _log(f"VRT res: {ds.res}")
        _log(f"VRT CRS: {ds.crs}")

    _log("Multi-county training preparation COMPLETE")

# src/utils/prepare_multicounty_training.py

import os
import sys
import glob
from pathlib import Path
from typing import List, Dict

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.coords import BoundingBox
from shapely.geometry import box
import matplotlib.pyplot as plt

from utils.make_vrt import write_mosaic_vrt


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Log message with prepare prefix."""
    print(f"[prepare] {msg}", flush=True)


def _fail(msg: str) -> None:
    """Raise a fatal error with prepare prefix."""
    raise RuntimeError(f"[prepare][FATAL] {msg}")


def _normalize_class_column(
    gdf: gpd.GeoDataFrame,
    county: str,
) -> gpd.GeoDataFrame:
    """Normalize class column naming across counties."""
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
    """Get bounds from a raster file."""
    with rasterio.open(raster_path) as ds:
        return ds.bounds


def _bounds_intersect(a: BoundingBox, b: BoundingBox) -> bool:
    """Check if two bounding boxes intersect."""
    return box(*a).intersects(box(*b))


def _plot_overlay(raster_path: str, gdf: gpd.GeoDataFrame, title: str) -> None:
    """Plot raster with label overlays for debugging."""
    with rasterio.open(raster_path) as ds:
        img = ds.read([1, 2, 3])
        transform = ds.transform
        raster_crs = ds.crs
        raster_bounds = ds.bounds
        raster_res = ds.res

    # Diagnostic output
    print(f"\n[plot] Raster: {Path(raster_path).name}")
    print(f"[plot]   CRS: {raster_crs}")
    print(f"[plot]   Bounds: {raster_bounds}")
    print(f"[plot]   Resolution: {raster_res}")
    print(f"[plot] Labels:")
    print(f"[plot]   CRS: {gdf.crs}")
    print(f"[plot]   Total bounds: {gdf.total_bounds}")
    print(f"[plot]   Num features: {len(gdf)}")

    # Ensure labels are in same CRS as raster
    gdf_plot = gdf.copy()
    if gdf_plot.crs != raster_crs:
        print(f"[plot]   Converting labels from {gdf_plot.crs} to {raster_crs}")
        gdf_plot = gdf_plot.to_crs(raster_crs)
        print(f"[plot]   Bounds after conversion: {gdf_plot.total_bounds}")

    # Check overlap
    raster_box = box(*raster_bounds)
    gdf_box = box(*gdf_plot.total_bounds)
    overlaps = raster_box.intersects(gdf_box)
    print(f"[plot]   Spatial overlap: {overlaps}")
    if overlaps:
        intersection = raster_box.intersection(gdf_box)
        print(f"[plot]   Intersection bounds: {intersection.bounds}")

    fig, ax = plt.subplots(figsize=(10, 10))
    show(img, transform=transform, ax=ax)
    gdf_plot.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2, alpha=0.8)
    
    # Set axis limits to show raster + labels (use larger bounds to see both)
    all_bounds = box(*raster_bounds).union(box(*gdf_plot.total_bounds)).bounds
    margin = max(500, (all_bounds[2] - all_bounds[0]) * 0.1)  # 10% margin or 500 units, whichever is larger
    
    ax.set_xlim(all_bounds[0] - margin, all_bounds[2] + margin)
    ax.set_ylim(all_bounds[1] - margin, all_bounds[3] + margin)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f"X ({raster_crs})")
    ax.set_ylabel(f"Y ({raster_crs})")
    plt.tight_layout()
    plt.show()
    print(f"[plot] Plot complete.\n")


def _filter_singleclass_labels(
    gdf: gpd.GeoDataFrame,
    focus_class: str,
) -> gpd.GeoDataFrame:
    """
    Filter merged labels for single-class training with 50/50 hard/easy negatives.
    
    Strategy:
    - Keep: focus_class (all examples)
    - Background (50%): all non-focus classes that are NOT "Background" (hard negatives - wrong class)
    - Background (50%): all existing "Background" class entries (easy negatives - unverified or truly negative)
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Merged labels with 'Classname' column
    focus_class : str
        Target class name (e.g., 'Tile_Outlet')
    
    Returns
    -------
    GeoDataFrame
        Filtered labels with only focus_class and Background classes
    """
    
    gdf = gdf.copy()
    
    # Validate focus_class exists
    if focus_class not in gdf["Classname"].unique():
        _fail(f"Focus class '{focus_class}' not found in labels. Available: {gdf['Classname'].unique()}")
    
    # Separate focus class
    focus = gdf[gdf["Classname"] == focus_class].copy()
    
    # Separate hard negatives (other classes, not Background)
    hard_negatives = gdf[(gdf["Classname"] != focus_class) & (gdf["Classname"] != "Background")].copy()
    
    # Easy negatives (already Background)
    easy_negatives = gdf[gdf["Classname"] == "Background"].copy()
    
    _log(f"  {focus_class}: {len(focus)} examples")
    _log(f"  Hard negatives (other classes): {len(hard_negatives)} examples")
    _log(f"  Easy negatives (Background): {len(easy_negatives)} examples")
    
    # 50/50 split for negatives
    n_hard = len(hard_negatives)
    n_easy = len(easy_negatives)
    
    if n_hard == 0 and n_easy == 0:
        _fail(f"No negative examples available for {focus_class} training")
    
    if n_hard == 0:
        _log(f"  WARNING: No other classes available; using only Background")
        background = easy_negatives
    elif n_easy == 0:
        _log(f"  WARNING: No Background examples; using only other classes")
        background = hard_negatives
    else:
        # Standard 50/50 split: use max count to avoid data loss
        target_count = max(n_hard, n_easy)
        hard_sample = hard_negatives.sample(n=min(target_count, n_hard), replace=True, random_state=42)
        easy_sample = easy_negatives.sample(n=min(target_count, n_easy), replace=True, random_state=42)
        background = pd.concat([hard_sample, easy_sample], ignore_index=True)
    
    # Assign classes
    focus["Classname"] = focus_class
    background["Classname"] = "Background"
    
    # Combine and return
    result = pd.concat([focus, background], ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)
    
    _log(f"  Final: {len(focus)} {focus_class} + {len(background)} Background = {len(result)} total")
    
    return result


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def prepare_multicounty_training(
    county_data_dir: str,
    selected_counties: List[str],
    out_labels: str,
    out_vrt: str,
    *,
    verified_only: bool = True,
    debug_plots: bool = False,
    single_class: bool = False,
    focus_class: str = None,
) -> None:
    """
    Prepare a multi-county training dataset:
      - merge shapefiles
      - validate alignment
      - build a training VRT
      - optionally filter for single-class training with 50/50 hard/easy negatives

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
    verified_only : bool
        Keep only VerifiedTr == 1 if the column exists
    debug_plots : bool
        Show raster/label overlay plots for each county
    single_class : bool
        If True, filter labels for single-class training (requires focus_class)
    focus_class : str
        Target class for single-class training (e.g., 'Tile_Outlet')
    """

    county_data_dir = Path(county_data_dir)
    out_labels = Path(out_labels)
    out_vrt = Path(out_vrt)

    if not selected_counties:
        _fail("No counties selected")

    _log(f"Preparing training data for {len(selected_counties)} counties")

    # -------------------------------------------------------------------------
    # Collect shapefiles + tiles
    # -------------------------------------------------------------------------

    county_to_shp: Dict[str, Path] = {}
    county_to_tiles: Dict[str, List[Path]] = {}

    for c in selected_counties:
        cdir = county_data_dir / c

        shp_files = sorted(cdir.glob("*.shp"))
        if not shp_files:
            _fail(f"No shapefile found in {cdir}")
        if len(shp_files) > 1:
            _log(f"WARNING: multiple shapefiles in {cdir}, using {shp_files[0].name}")

        tiles = sorted((cdir / "tiles").glob("*.tif"))
        if not tiles:
            _fail(f"No tiles found in {cdir}/tiles")

        county_to_shp[c] = shp_files[0]
        county_to_tiles[c] = tiles

    # -------------------------------------------------------------------------
    # Establish target CRS from first tile
    # -------------------------------------------------------------------------

    first_tile = county_to_tiles[selected_counties[0]][0]
    with rasterio.open(first_tile) as ds:
        target_crs = ds.crs
        ref_bounds = ds.bounds
        ref_res = ds.res

    _log(f"Target CRS: {target_crs} (from first tile: {first_tile.name})")
    _log(f"Reference resolution: {ref_res}")
    
    # Sanity check: all tiles should be in same CRS
    crs_counts = {}
    crs_by_county = {}
    for c in selected_counties:
        for t in county_to_tiles[c][:1]:  # check first tile of each county
            with rasterio.open(t) as ds:
                crs_str = str(ds.crs)
                crs_counts[crs_str] = crs_counts.get(crs_str, 0) + 1
                crs_by_county[c] = crs_str
    
    if len(crs_counts) > 1:
        _log(f"WARNING: Tiles have mixed CRS!")
        for crs_str, count in sorted(crs_counts.items()):
            _log(f"  {crs_str}: {count} counties")
        _log(f"Per-county CRS mapping:")
        for c in sorted(crs_by_county.keys()):
            _log(f"  {c}: {crs_by_county[c]}")
    else:
        _log(f"✓ All tiles in same CRS: {list(crs_counts.keys())[0]}")

    # -------------------------------------------------------------------------
    # Load + validate labels per county
    # -------------------------------------------------------------------------

    gdfs = []
    poison_counties = []

    for c in selected_counties:
        shp = county_to_shp[c]
        tiles = county_to_tiles[c]

        _log(f"Loading labels for {c}")

        gdf = gpd.read_file(shp, engine="pyogrio")
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

        # Normalize class column title
        gdf = _normalize_class_column(gdf, county=c)

        # Handle verified labels: keep both positive and negative examples
        if verified_only and "VerifiedTr" in gdf.columns:
            # Keep rows where VerifiedTr is 0 (negative) or 1 (positive)
            gdf = gdf[gdf["VerifiedTr"].isin([0, 1])]
            
            # Assign "Background" class to negative examples (VerifiedTr == 0)
            gdf.loc[gdf["VerifiedTr"] == 0, "Classname"] = "Background"
            
            # Log class distribution for this county
            n_positives = (gdf["VerifiedTr"] == 1).sum()
            n_negatives = (gdf["VerifiedTr"] == 0).sum()
            _log(f"  {c}: {n_positives} positive examples, {n_negatives} negative (Background) examples")

        if gdf.empty:
            _log(f"WARNING: {c} has zero usable labels")
            poison_counties.append(c)
            continue

        if gdf.crs != target_crs:
            _log(f"  Converting labels from {gdf.crs} to {target_crs}")
            gdf_bounds_before = gdf.total_bounds
            gdf = gdf.to_crs(target_crs)
            gdf_bounds_after = gdf.total_bounds
            _log(f"  Bounds before conversion: {gdf_bounds_before}")
            _log(f"  Bounds after conversion:  {gdf_bounds_after}")
        else:
            _log(f"  Labels already in {target_crs}")
            gdf_bounds_after = gdf.total_bounds
            _log(f"  Bounds: {gdf_bounds_after}")
        
        # Sanity check: bounds should be in feet (large numbers), not degrees (small numbers)
        bounds_range = gdf_bounds_after[2] - gdf_bounds_after[0]  # max_x - min_x
        if bounds_range < 1000:  # less than 1000 units = probably degrees, not feet
            _log(f"WARNING: {c} bounds suspiciously small (range: {bounds_range}). May be in degrees instead of feet!")
            _log(f"  Attempting to force convert to {target_crs}...")
            gdf = gdf.to_crs(target_crs)
            gdf_bounds_after = gdf.total_bounds
            _log(f"  Bounds after force convert: {gdf_bounds_after}")

        if debug_plots:
            _plot_overlay(tiles[0], gdf, title=f"{c} label alignment")

        gdf["county"] = c
        gdfs.append(gdf)

    if not gdfs:
        _fail("No valid counties left after validation")

    if poison_counties:
        _log(f"WARNING: excluding {len(poison_counties)} counties due to alignment issues:")
        for c in poison_counties:
            _log(f"  - {c}")

    # -------------------------------------------------------------------------
    # Merge labels
    # -------------------------------------------------------------------------

    merged = pd.concat(gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=target_crs)

    _log(f"Merged labels: {len(merged)} total features")

    # Sanity: class distribution
    if "Classname" in merged.columns:
        _log("Class distribution before filtering:")
        print(merged["Classname"].value_counts())

    # -------------------------------------------------------------------------
    # Single-class filtering (optional)
    # -------------------------------------------------------------------------
    
    if single_class:
        if not focus_class:
            _fail("single_class=True requires focus_class parameter")
        
        _log(f"\nFiltering for single-class training: focus_class='{focus_class}'")
        _log(f"Using 50/50 hard/easy negatives strategy")
        
        merged = _filter_singleclass_labels(merged, focus_class=focus_class)
        
        _log("Single-class filtering complete. Final class distribution:")
        print(merged["Classname"].value_counts())

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
    # Build VRT
    # -------------------------------------------------------------------------

    all_tiles = []
    for tiles in county_to_tiles.values():
        all_tiles.extend(tiles)

    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    if out_vrt.exists():
        out_vrt.unlink()

    write_mosaic_vrt(out_vrt, all_tiles)

    with rasterio.open(out_vrt) as ds:
        _log(f"VRT built → {out_vrt}")
        _log(f"VRT bounds: {ds.bounds}")
        _log(f"VRT res: {ds.res}")
        _log(f"VRT CRS: {ds.crs}")

    _log("Multi-county training preparation COMPLETE")

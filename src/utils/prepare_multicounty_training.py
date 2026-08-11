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
from rasterio.merge import merge as rio_merge
from rasterio.coords import BoundingBox
from shapely.geometry import box
import matplotlib.pyplot as plt

from utils.make_vrt import write_mosaic_vrt
from utils.indiana_cogs import project_root, CANONICAL_CRS as DEFAULT_CANONICAL_CRS


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


def _plot_overlay(raster_paths, gdf: gpd.GeoDataFrame, title: str) -> None:
    """Plot raster with label overlays for debugging.

    raster_paths may be a single raster path, or a list of raster paths — in
    the latter case all tiles are merged into a single mosaic and plotted
    together on one figure (instead of one figure per tile).
    """
    if isinstance(raster_paths, (str, Path)):
        raster_paths = [raster_paths]

    if len(raster_paths) == 1:
        with rasterio.open(raster_paths[0]) as ds:
            img = ds.read([1, 2, 3])
            transform = ds.transform
            raster_crs = ds.crs
            raster_bounds = ds.bounds
            raster_res = ds.res
        raster_label = Path(raster_paths[0]).name
    else:
        srcs = [rasterio.open(p) for p in raster_paths]
        try:
            img, transform = rio_merge(srcs, indexes=[1, 2, 3])
            raster_crs = srcs[0].crs
            raster_res = srcs[0].res
        finally:
            for s in srcs:
                s.close()
        height, width = img.shape[1], img.shape[2]
        raster_bounds = BoundingBox(*rasterio.transform.array_bounds(height, width, transform))
        raster_label = f"mosaic of {len(raster_paths)} tiles"

    # Diagnostic output
    print(f"\n[plot] Raster: {raster_label}")
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
    positive_ratio: int = 80,
    in_class_ratio: int = 50,
    remove_overlaps: bool = True,
    include_background: bool = True,
) -> gpd.GeoDataFrame:
    """
    Filter merged labels for single-class training with tunable positive/negative ratio.
    
    Strategy:
    - Confirmed positives: focus_class with VerifiedTr == 1 (e.g., confirmed Tile_Outlets)
    - In-class negatives: focus_class with VerifiedTr == 0 (looked like focus_class but aren't)
    - Out-of-class negatives: features from other classes (fundamentally different)
    - Background pool splits between in-class and out-of-class using in_class_ratio
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Merged labels with 'Classname' and 'VerifiedTr' columns
    focus_class : str
        Target class name (e.g., 'Tile_Outlet')
    positive_ratio : int
        Percentage of training data that should be confirmed positives. Default 80.
        Ignored when include_background=False.
        - 80: 80% confirmed focus_class, 20% negatives (in-class + out-of-class combined)
        - 50: 50/50 split between confirmed and negatives
        - 20: 20% confirmed, 80% negatives
    in_class_ratio : int
        Percentage of the negative pool that should be in-class negatives. Default 50.
        Ignored when include_background=False.
        - 50: 50/50 split between in-class and out-of-class negatives
        - 80: 80% in-class (misclassified examples), 20% out-of-class
        - 20: 20% in-class, 80% out-of-class
    remove_overlaps : bool
        If True, remove any non-focus-class features that overlap with focus_class features.
        This prevents conflicting training signals in overlap zones. Default True.
        Ignored when include_background=False (there's no negative pool to filter).
    include_background : bool
        If True (default), builds a negative/"Background" pool sized via positive_ratio/
        in_class_ratio, matching prior behavior. If False, returns ONLY confirmed
        positives (VerifiedTr==1 focus_class rows) — no Background class, no negatives
        at all. positive_ratio/in_class_ratio/remove_overlaps are ignored in this case.
    
    Returns
    -------
    GeoDataFrame
        Filtered labels with only focus_class (and Background, unless
        include_background=False)
    """
    
    gdf = gdf.copy()
    
    # Ensure VerifiedTr column exists (some counties may not have it)
    if "VerifiedTr" not in gdf.columns:
        gdf["VerifiedTr"] = 1  # assume all verified if column missing
    
    # Validate focus_class exists
    if focus_class not in gdf["Classname"].unique():
        _fail(f"Focus class '{focus_class}' not found in labels. Available: {gdf['Classname'].unique()}")
    
    # Separate confirmed positives (focus_class with VerifiedTr == 1)
    confirmed_positive = gdf[(gdf["Classname"] == focus_class) & (gdf["VerifiedTr"] == 1)].copy()

    if not include_background:
        # Positives-only mode: no negative/Background sampling at all.
        confirmed_positive["Classname"] = focus_class
        result = gpd.GeoDataFrame(confirmed_positive, geometry="geometry", crs=gdf.crs)
        _log(f"  include_background=False: {len(result)} confirmed {focus_class} only "
             f"(no negatives/Background class)")
        return result

    # In-class negatives: focus_class with VerifiedTr == 0 (misclassified examples)
    in_class_negative = gdf[(gdf["Classname"] == focus_class) & (gdf["VerifiedTr"] == 0)].copy()
    
    # Out-of-class negatives: other classes
    out_of_class_negative = gdf[(gdf["Classname"] != focus_class)].copy()
    
    _log(f"  Confirmed {focus_class} (VerifiedTr=1): {len(confirmed_positive)} examples")
    _log(f"  In-class negatives ({focus_class}, VerifiedTr=0): {len(in_class_negative)} examples")
    _log(f"  Out-of-class negatives (other classes): {len(out_of_class_negative)} examples")
    
    # Remove overlaps from out-of-class negatives
    n_removed_overlaps = 0
    if remove_overlaps and len(confirmed_positive) > 0:
        # Union all confirmed positive geometries
        focus_union = confirmed_positive.unary_union
        
        # Mark features that DO NOT overlap with focus
        non_overlapping = ~out_of_class_negative.geometry.intersects(focus_union)
        n_removed_overlaps = (~non_overlapping).sum()
        
        out_of_class_negative = out_of_class_negative[non_overlapping].copy()
        
        if n_removed_overlaps > 0:
            _log(f"  Removed {n_removed_overlaps} out-of-class features overlapping confirmed {focus_class}")
    
    # Calculate target negative pool size based on positive_ratio
    n_confirmed = len(confirmed_positive)
    if positive_ratio <= 0 or positive_ratio >= 100:
        _fail(f"positive_ratio must be between 1 and 99, got {positive_ratio}")
    
    # If confirmed are positive_ratio%, then negatives are (100-positive_ratio)%
    # So: n_confirmed / total = positive_ratio / 100
    # Therefore: total = n_confirmed * 100 / positive_ratio
    # And: n_negatives = total - n_confirmed
    n_total = int(n_confirmed * 100 / positive_ratio)
    n_negatives_target = n_total - n_confirmed
    
    # Split negatives target between in-class and out-of-class
    in_class_pct = in_class_ratio / 100.0
    out_of_class_pct = (100 - in_class_ratio) / 100.0
    
    n_in_class_target = int(n_negatives_target * in_class_pct)
    n_out_of_class_target = n_negatives_target - n_in_class_target
    
    # Sample with replacement if needed, reallocating shortfalls between pools
    n_in_class_available = len(in_class_negative)
    n_out_of_class_available = len(out_of_class_negative)
    
    if n_in_class_available == 0 and n_out_of_class_available == 0:
        _fail(f"No negative examples available for {focus_class} training")
    
    # Try to get in_class_target, then reallocate any shortfall to out_of_class_target
    n_in_class_actual = min(n_in_class_target, n_in_class_available)
    n_in_class_shortfall = max(0, n_in_class_target - n_in_class_available)
    n_out_of_class_adjusted = n_out_of_class_target + n_in_class_shortfall
    
    # Sample from each pool
    if n_in_class_available > 0:
        in_class_sample = in_class_negative.sample(
            n=n_in_class_actual,
            replace=(n_in_class_target > n_in_class_available),
            random_state=42
        )
    else:
        in_class_sample = gpd.GeoDataFrame(columns=in_class_negative.columns, crs=gdf.crs)
    
    if n_out_of_class_available > 0:
        n_out_of_class_actual = min(n_out_of_class_adjusted, n_out_of_class_available)
        out_of_class_sample = out_of_class_negative.sample(
            n=n_out_of_class_actual,
            replace=(n_out_of_class_adjusted > n_out_of_class_available),
            random_state=42
        )
    else:
        out_of_class_sample = gpd.GeoDataFrame(columns=out_of_class_negative.columns, crs=gdf.crs)
    
    # Combine negative samples
    background = pd.concat([in_class_sample, out_of_class_sample], ignore_index=True)
    
    # Log actual composition (accounting for reallocations)
    n_background_actual = len(background)
    actual_in_class_pct = (len(in_class_sample) / n_background_actual * 100) if n_background_actual > 0 else 0
    actual_out_of_class_pct = (len(out_of_class_sample) / n_background_actual * 100) if n_background_actual > 0 else 0
    
    _log(f"  Negative pool target: {n_negatives_target} total")
    if n_in_class_shortfall > 0:
        _log(f"    In-class negatives (VerifiedTr=0): {len(in_class_sample)} (requested {n_in_class_target}, only {n_in_class_available} available)")
        _log(f"    Out-of-class negatives (other classes): {len(out_of_class_sample)} (reallocated {n_in_class_shortfall} from in-class shortfall)")
    else:
        _log(f"    In-class negatives (VerifiedTr=0): {len(in_class_sample)} ({actual_in_class_pct:.1f}%)")
        _log(f"    Out-of-class negatives (other classes): {len(out_of_class_sample)} ({actual_out_of_class_pct:.1f}%)")
    
    # Assign classes
    confirmed_positive["Classname"] = focus_class
    background["Classname"] = "Background"
    
    # Combine and return
    result = pd.concat([confirmed_positive, background], ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)
    
    final_ratio = 100 * len(confirmed_positive) / len(result)
    _log(f"  Final: {len(confirmed_positive)} {focus_class} ({final_ratio:.1f}%) + {len(background)} Background ({100-final_ratio:.1f}%) = {len(result)} total")
    
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
    debug_plot_mode: str = "sample",
    single_class: bool = False,
    focus_class: str = None,
    positive_ratio: int = 80,
    in_class_ratio: int = 50,
    include_background: bool = True,
    background_ratio: int = None,
) -> None:
    """
    Prepare a multi-county training dataset:
      - merge shapefiles
      - validate alignment
      - build a training VRT
      - optionally filter for single-class training with tunable positive/negative ratio

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
        Keep only rows where VerifiedTr is 0 or 1 if the column exists (i.e. drop rows
        with any other/missing VerifiedTr value). VerifiedTr==0 rows are NOT dropped
        outright: in single_class mode they feed the in-class-negative pool (see
        _filter_singleclass_labels); in multi-class mode they are relabeled to an
        explicit "Background" class (see below).
    debug_plots : bool
        Show raster/label overlay plots for each county
    debug_plot_mode : str
        Only used when debug_plots is True. Controls how many tiles are plotted per county:
        - "sample" (default): plot a single representative tile per county
        - "all": merge every canonical tile for the county into one mosaic and
          plot it as a single figure (with all labels overlaid)
    single_class : bool
        If True, filter labels for single-class training (requires focus_class)
    focus_class : str
        Target class for single-class training (e.g., 'Tile_Outlet')
    positive_ratio : int
        For single-class mode: percentage of training data that should be confirmed positives.
        - 80 (default): 80% confirmed focus_class, 20% negatives
        - 50: 50/50 split
        - 20: 20% confirmed, 80% negatives
    in_class_ratio : int
        For single-class mode: percentage of the negative pool that should be in-class negatives.
        - 50 (default): 50/50 split between in-class and out-of-class
        - 80: 80% in-class (misclassified), 20% out-of-class
        - 20: 20% in-class, 80% out-of-class
        Ignored when include_background=False.
    include_background : bool
        For single-class mode: if True (default), builds a negative/Background pool
        sized via positive_ratio/in_class_ratio. If False, keeps ONLY confirmed
        positives (VerifiedTr==1 focus_class) — no Background class, no negatives at
        all. positive_ratio/in_class_ratio are ignored in this case.
    background_ratio : int, optional
        For multi-class mode only (single_class=False): caps the relabeled Background
        pool (VerifiedTr==0 rows) so it's at most `background_ratio`% of the final
        combined dataset (confirmed real-class rows + kept Background rows). Excess
        Background rows are randomly downsampled (never upsampled). Default None means
        no cap — all VerifiedTr==0 rows are kept as Background (prior behavior).
        Ignored in single-class mode.
    """

    county_data_dir = Path(county_data_dir)
    out_labels = Path(out_labels)
    out_vrt = Path(out_vrt)

    if not selected_counties:
        _fail("No counties selected")

    if debug_plot_mode not in ("sample", "all"):
        _fail(f"debug_plot_mode must be 'sample' or 'all', got {debug_plot_mode!r}")

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

        tiles = sorted((cdir / "canonical_tiles").glob("*.tif"))
        if not tiles:
            _fail(
                f"No canonical tiles found in {cdir}/canonical_tiles. "
                f"Run ensure_canonical_mosaic_for_counties() for this county first."
            )

        county_to_shp[c] = shp_files[0]
        county_to_tiles[c] = tiles

    # -------------------------------------------------------------------------
    # Target CRS: the pipeline's canonical CRS (every canonical_tiles/ chip is
    # already reprojected + resampled to this CRS by
    # ensure_canonical_mosaic_for_counties(), so there is no need to sample or
    # sanity-check individual tiles here).
    # -------------------------------------------------------------------------

    target_crs = DEFAULT_CANONICAL_CRS
    _crs_config_file = project_root() / "outputs" / ".reference_crs"
    if _crs_config_file.exists():
        target_crs = _crs_config_file.read_text().strip()

    _log(f"Target CRS: {target_crs} (pipeline canonical CRS)")

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
        # NOTE: Do NOT overwrite Classname for VerifiedTr=0. Keep original class labels so that
        # single-class filtering can identify in-class negatives (e.g., Tile_Outlet with VerifiedTr=0)
        if verified_only and "VerifiedTr" in gdf.columns:
            # Keep rows where VerifiedTr is 0 (negative) or 1 (positive)
            gdf = gdf[gdf["VerifiedTr"].isin([0, 1])]
            
            # Log class distribution for this county (split by VerifiedTr)
            n_positives = (gdf["VerifiedTr"] == 1).sum()
            n_negatives = (gdf["VerifiedTr"] == 0).sum()
            _log(f"  {c}: {n_positives} confirmed (VerifiedTr=1), {n_negatives} unconfirmed (VerifiedTr=0)")
        elif verified_only:
            # If no VerifiedTr column, assume all records are verified (VerifiedTr=1)
            gdf["VerifiedTr"] = 1
            _log(f"  {c}: no VerifiedTr column, assuming all {len(gdf)} examples are confirmed")

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
            if debug_plot_mode == "all":
                _plot_overlay(tiles, gdf, title=f"{c} label alignment - all {len(tiles)} tiles")
            else:
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
        
        if include_background:
            _log(f"\nFiltering for single-class training: focus_class='{focus_class}'")
            _log(f"Target ratio: {positive_ratio}% confirmed {focus_class}, {100-positive_ratio}% negatives")
            _log(f"Negative composition: {in_class_ratio}% in-class (VerifiedTr=0), {100-in_class_ratio}% out-of-class")
            _log(f"Removing overlaps: any out-of-class features overlapping confirmed {focus_class} will be excluded")
        else:
            _log(f"\nFiltering for single-class training: focus_class='{focus_class}'")
            _log(f"include_background=False: keeping ONLY confirmed {focus_class} (VerifiedTr=1), no Background class")
        
        merged = _filter_singleclass_labels(
            merged,
            focus_class=focus_class,
            positive_ratio=positive_ratio,
            in_class_ratio=in_class_ratio,
            remove_overlaps=True,
            include_background=include_background,
        )
        
        _log("Single-class filtering complete. Final class distribution:")
        print(merged["Classname"].value_counts())
    else:
        # Multi-class mode: unconfirmed (VerifiedTr=0) rows were kept above (across ALL
        # classes, not just a single focus class) with their original Classname intact,
        # since the earlier verified_only filter deliberately keeps both VerifiedTr 0/1
        # rows. Relabel those unconfirmed rows to an explicit "Background" class here so
        # they're trained as real (correctly-boxed) hard-negative/distractor examples
        # rather than silently leaking in as if they were confirmed instances of their
        # original class. Optionally cap Background's volume (background_ratio) so it
        # doesn't dwarf the real classes.
        if "VerifiedTr" in merged.columns:
            merged_crs = merged.crs
            is_background = merged["VerifiedTr"] == 0
            background_rows = merged[is_background]
            confirmed_rows = merged[~is_background]
            n_available = len(background_rows)

            if background_ratio is not None:
                if not (1 <= background_ratio <= 99):
                    _fail(f"background_ratio must be between 1 and 99, got {background_ratio}")
                n_confirmed = len(confirmed_rows)
                n_target = int(n_confirmed * background_ratio / (100 - background_ratio))
                n_keep = min(n_target, n_available)
                if n_keep < n_available:
                    background_rows = background_rows.sample(n=n_keep, random_state=42)
                _log(f"Multi-class mode: capping Background to ~{background_ratio}% of total "
                     f"({n_keep} of {n_available} unconfirmed rows kept, {n_available - n_keep} dropped)")
            else:
                _log(f"Multi-class mode: keeping all {n_available} unconfirmed (VerifiedTr=0) rows as Background (no cap)")

            background_rows = background_rows.copy()
            background_rows["Classname"] = "Background"
            merged = gpd.GeoDataFrame(
                pd.concat([confirmed_rows, background_rows], ignore_index=True),
                geometry="geometry",
                crs=merged_crs,
            )

        _log("Multi-class mode. Final class distribution:")
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

# scripts/prepare_multicounty_training.py

import os
import sys
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional

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

def _log(msg: str):
    print(f"[prepare] {msg}", flush=True)


def _fail(msg: str):
    raise RuntimeError(f"[prepare][FATAL] {msg}")


def _normalize_class_column(
    gdf: gpd.GeoDataFrame,
    county: str,
) -> gpd.GeoDataFrame:
    class_cols = [c for c in ("Classname", "Class") if c in gdf.columns]

    if len(class_cols) == 0:
        _fail(f"{county}: missing class column (expected 'Classname' or 'Class')")

    if len(class_cols) > 1:
        _fail(f"{county}: both 'Classname' and 'Class' present — ambiguous schema")

    if class_cols[0] == "Class":
        gdf = gdf.rename(columns={"Class": "Classname"})

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
):
    """
    Prepare a multi-county training dataset:
      - merge shapefiles
      - validate alignment
      - build a training VRT

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

    _log(f"Target CRS: {target_crs}")
    _log(f"Reference resolution: {ref_res}")

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

        if verified_only and "VerifiedTr" in gdf.columns:
            gdf = gdf[gdf["VerifiedTr"] == 1]

        if gdf.empty:
            _log(f"WARNING: {c} has zero usable labels")
            poison_counties.append(c)
            continue

        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Validate overlap with tiles
        overlaps = 0
        for t in tiles:
            if _bounds_intersect(_raster_bounds(t), gdf.total_bounds):
                overlaps += 1

        if overlaps == 0:
            _log(f"ERROR: {c} labels do not overlap any tiles")
            poison_counties.append(c)
            continue

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
        _log("Class distribution:")
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


# src/utils/crs_utils.py
# all comments in lowercase; variable names in snake_case

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import rasterio
from shapely.geometry.base import BaseGeometry

PathLike = Union[str, Path]


def read_raster_crs(raster_path: PathLike):
    """
    read and return the raster crs using rasterio.
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"raster not found: {raster_path}")
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"raster has no crs: {raster_path}")
        return src.crs


def fix_invalid_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    attempt to fix invalid geometries with a cheap buffer(0) trick.
    works for common self-intersections in polygons; no-op for valid geoms.
    """
    # avoid importing shapely.make_valid to keep compatibility with shapely<2
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def normalize_labels_crs(
    labels_path: PathLike,
    raster_path: PathLike,
    out_path: Optional[PathLike] = None,
    prefer_gpkg: bool = False,
    fix_invalid: bool = True,
    overwrite: bool = True,
) -> str:
    """
    ensure vector labels are in the same crs as the raster.
    if crs differs, reproject and write to out_path (or derive a sensible default).
    returns the path to the labels file to use downstream (possibly the input path).

    parameters
    ----------
    labels_path : path-like
        path to the training labels (shapefile/gpkg/etc).
    raster_path : path-like
        path to the raster imagery used for training/inference.
    out_path : optional path-like
        explicit output path for reprojected labels. if none, a default is created.
    prefer_gpkg : bool
        if true, write geopackage (.gpkg). otherwise preserve the original extension.
    fix_invalid : bool
        if true, tries to fix invalid geometries before writing.
    overwrite : bool
        if false and out_path exists, will return out_path without reprojecting again.

    returns
    -------
    str : path to the labels file in the raster crs.
    """
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels not found: {labels_path}")

    raster_crs = read_raster_crs(raster_path)

    gdf = gpd.read_file(labels_path)
    if gdf.crs is None:
        raise ValueError(
            f"labels have no crs: {labels_path}. "
            f"ensure the source file has a crs (e.g., .prj for shapefile) or set it explicitly before normalization."
        )

    # if already matching, just return the original path
    if gdf.crs == raster_crs:
        return str(labels_path)

    # derive an output path if not supplied
    if out_path is None:
        parent = labels_path.parent
        stem = labels_path.stem
        if prefer_gpkg:
            out_path = parent / f"{stem}_reproj.gpkg"
        else:
            # keep the original extension (commonly .shp) to avoid breaking downstream code
            out_path = parent / f"{stem}_reproj{labels_path.suffix}"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return str(out_path)

    # reproject
    gdf_reproj = gdf.to_crs(raster_crs)

    # optionally fix invalid geometries
    if fix_invalid:
        try:
            gdf_reproj = fix_invalid_geoms(gdf_reproj)
        except Exception:
            # if anything goes wrong, proceed without fixing
            pass

    # choose driver and optional layer name
    if out_path.suffix.lower() == ".gpkg":
        layer_name = out_path.stem
        gdf_reproj.to_file(out_path, driver="GPKG", layer=layer_name)
    else:
        # for shapefile and others, geopandas picks the driver by extension
        gdf_reproj.to_file(out_path)

    return str(out_path)


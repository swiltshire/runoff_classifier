"""
One-time backfill: write S3 manifests (see write_county_manifest() in
src/utils/prepare_reprojected_tiles.py) for counties that were already fully
processed BEFORE that manifest-writing step existed.

For counties whose raw tiles are still present locally, this just reuses the
normal county_tiles_by_crs() + local rasterio.open() path. For counties whose
raw tiles have already been archived+pruned (e.g. the fully-completed 2967
training set), this reads each archived tile's bounds/CRS directly from S3 via
GDAL's /vsis3/ driver (header-only reads - NOT a full download) instead of
restoring the whole raw tile set locally first.

Going forward, ensure_canonical_mosaic_for_counties() writes/refreshes each
requested county's manifest automatically - this script only needs to be run
once to backfill counties processed before that existed.

Usage:
    python scripts/backfill_county_manifests.py --counties "Benton,Boone,Cass"
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rasterio
from rasterio.session import AWSSession

from utils.indiana_cogs import project_root, safe_name  # noqa: E402
from utils.prepare_reprojected_tiles import (  # noqa: E402
    RAW_ARCHIVE_S3_PREFIX,
    S3_BUCKET,
    _bounds_to_canonical,
    _raster_bounds_native,
    county_tiles_by_crs,
    grid_cells_for_bounds,
    log,
    s3,
    write_county_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill S3 manifests (list of chip keys per county) for already-processed counties."
    )
    parser.add_argument(
        "--counties", type=str, required=True,
        help="comma-separated county names to backfill manifests for",
    )
    return parser.parse_args()


def archived_tile_bounds_by_crs(county_safe: str) -> Dict[str, Tuple[float, float, float, float]]:
    """For a county whose raw tiles have been archived+pruned, read each
    archived tile's bounds/CRS directly from S3 via /vsis3/ (header-only -
    does not download the full file), grouped by native EPSG, and return
    {epsg_code: combined_bounds}."""
    prefix = f"{RAW_ARCHIVE_S3_PREFIX}/{county_safe}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".tif"):
                keys.append(obj["Key"])

    bounds_by_epsg: Dict[str, List[float]] = {}  # epsg -> [minx, miny, maxx, maxy]
    with rasterio.Env(session=AWSSession()):
        for key in keys:
            vsi_path = f"/vsis3/{S3_BUCKET}/{key}"
            with rasterio.open(vsi_path) as ds:
                if not ds.crs:
                    continue
                epsg_code = f"EPSG:{ds.crs.to_epsg()}"
                b = ds.bounds
                if epsg_code not in bounds_by_epsg:
                    bounds_by_epsg[epsg_code] = [b.left, b.bottom, b.right, b.top]
                else:
                    cur = bounds_by_epsg[epsg_code]
                    cur[0] = min(cur[0], b.left)
                    cur[1] = min(cur[1], b.bottom)
                    cur[2] = max(cur[2], b.right)
                    cur[3] = max(cur[3], b.top)

    return {epsg: tuple(b) for epsg, b in bounds_by_epsg.items()}


def main():
    args = parse_args()
    counties = [c.strip() for c in args.counties.split(",") if c.strip()]

    for county in counties:
        county_safe = safe_name(county)
        tiles_dir = project_root() / "data" / "counties" / county_safe / "tiles"
        cells: List[Tuple[str, int, int]] = []

        if tiles_dir.is_dir() and any(tiles_dir.glob("*.tif")):
            log(f"{county}: raw tiles present locally, using county_tiles_by_crs()")
            subgroups = county_tiles_by_crs(county_safe)
            for epsg_code, tiles in subgroups.items():
                native_bounds = _raster_bounds_native(tiles)
                canon_bounds = _bounds_to_canonical(native_bounds, epsg_code, buffer_chips=0.0)
                cells.extend((epsg_code, r, c) for (r, c) in grid_cells_for_bounds(canon_bounds))
        else:
            log(f"{county}: no local raw tiles - reading archived tile bounds from S3 (/vsis3/ header reads)...")
            bounds_by_epsg = archived_tile_bounds_by_crs(county_safe)
            if not bounds_by_epsg:
                log(f"  \u26a0 WARNING: no local OR archived raw tiles found for {county} - skipping")
                continue
            for epsg_code, native_bounds in bounds_by_epsg.items():
                canon_bounds = _bounds_to_canonical(native_bounds, epsg_code, buffer_chips=0.0)
                cells.extend((epsg_code, r, c) for (r, c) in grid_cells_for_bounds(canon_bounds))

        if not cells:
            log(f"  \u26a0 WARNING: no cells computed for {county} - manifest NOT written")
            continue

        key = write_county_manifest(county, county_safe, cells)
        log(f"  \u2713 wrote manifest for {county}: {len(cells)} chip(s) -> s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    main()

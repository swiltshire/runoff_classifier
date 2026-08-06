"""
One-time backfill: write S3 manifests (see write_county_manifest() in
src/utils/prepare_reprojected_tiles.py) for counties that were already fully
processed BEFORE that manifest-writing step existed.

Instead of recomputing grid cells from raw tile bounds, this just lists
whatever is currently sitting in the county's local canonical_tiles/ folder.
_crop_and_cache_chip() names each materialized local file identically to its
S3 key's filename (chip_{epsg_num}_r{row}_c{col}.tif) - see
src/utils/prepare_reprojected_tiles.py - so the local tileset directly tells
us the exact (epsg_code, row, col) triples needed for the manifest, no bounds
math or S3 archive reads required.

Only works for counties whose canonical_tiles/ folder is still present
locally (i.e. hasn't been cleaned up yet). For counties without local
canonical_tiles/, this script logs a warning and skips them.

Going forward, ensure_canonical_mosaic_for_counties() writes/refreshes each
requested county's manifest automatically - this script only needs to be run
once to backfill counties processed before that existed.

Usage:
    python scripts/backfill_county_manifests.py --counties "Benton,Boone,Cass"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.indiana_cogs import project_root, safe_name  # noqa: E402
from utils.prepare_reprojected_tiles import S3_BUCKET, log, write_county_manifest  # noqa: E402

CHIP_FILENAME_RE = re.compile(r"^chip_(\d+)_r(-?\d+)_c(-?\d+)\.tif$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill S3 manifests (list of chip keys per county) for already-processed counties."
    )
    parser.add_argument(
        "--counties", type=str, required=True,
        help="comma-separated county names to backfill manifests for",
    )
    return parser.parse_args()


def cells_from_local_canonical_tiles(county_safe: str) -> List[Tuple[str, int, int]]:
    """Parse (epsg_code, row, col) triples directly out of the filenames
    already present in data/counties/{county_safe}/canonical_tiles/."""
    tiles_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
    if not tiles_dir.is_dir():
        return []

    cells: List[Tuple[str, int, int]] = []
    for path in tiles_dir.glob("*.tif"):
        m = CHIP_FILENAME_RE.match(path.name)
        if not m:
            continue
        epsg_num, row, col = m.groups()
        cells.append((f"EPSG:{epsg_num}", int(row), int(col)))
    return cells


def main():
    args = parse_args()
    counties = [c.strip() for c in args.counties.split(",") if c.strip()]

    for county in counties:
        county_safe = safe_name(county)
        cells = cells_from_local_canonical_tiles(county_safe)

        if not cells:
            log(f"  \u26a0 WARNING: no local canonical_tiles/ found for {county} - manifest NOT written")
            continue

        key = write_county_manifest(county, county_safe, cells)
        log(f"  \u2713 wrote manifest for {county}: {len(cells)} chip(s) -> s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    main()

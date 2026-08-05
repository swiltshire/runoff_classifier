"""
Audit the S3 canonical chip cache for orphaned/stale chip objects that don't
belong to the CURRENT canonical grid.

Why this can happen: chip S3 keys are addressed only by
`{epsg}/chips/chip_{epsg}_r{row}_c{col}.tif` - NOT by chip size or
resolution. If CHIP_SIZE_PX or CANONICAL_RES ever changes (as happened when
CHIP_SIZE_PX went 4096 -> 8192), old chips from a previous grid can:
  (a) coincidentally collide with a new grid cell's (row, col) key, in which
      case ensure_canonical_mosaic_for_counties()'s skip-cache would wrongly
      treat the OLD chip as already-generated for that cell, or
  (b) simply be orphaned clutter that was never cleaned up, inflating chip
      counts and wasting S3 storage without representing valid grid cells.

This script is READ-ONLY / dry-run: it recomputes the exact (row, col)
cells the CURRENT grid needs for the given counties (same math
ensure_canonical_mosaic_for_counties() uses), lists what's actually in S3
per CRS group, and reports/writes out anything that doesn't belong. It
never deletes anything - review the written orphan key list files before
deciding whether/how to delete.

IMPORTANT: only include counties whose raw tiles (data/counties/{county}/tiles/)
are still present locally. A county's needed cells can't be recomputed once
its raw tiles have been archived+pruned (which happens automatically once
its CRS group's canonical chips are fully generated) - restore from the S3
raw tile archive first if you need to audit an already-completed/pruned
group. Do not trust an "orphan" verdict for a CRS group unless ALL of that
group's contributing counties were included in --counties.

Usage:
    python scripts/audit_canonical_chips.py --counties "Benton,Boone,Cass"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Set, Tuple

# add project root and src/ to path (same requirement as
# prepare_canonical_mosaic.py - prepare_reprojected_tiles.py internally does
# `from utils.indiana_cogs import ...`, which requires src/ on sys.path)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.indiana_cogs import safe_name  # noqa: E402
from utils.prepare_reprojected_tiles import (  # noqa: E402
    S3_BUCKET,
    S3_PREFIX,
    _bounds_to_canonical,
    _raster_bounds_native,
    county_tiles_by_crs,
    grid_cells_for_bounds,
    s3,
)

CHIP_KEY_RE = re.compile(r"chip_(\d+)_r(-?\d+)_c(-?\d+)\.tif$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit S3 canonical chip cache for orphaned chips not in the current grid (read-only, never deletes)."
    )
    parser.add_argument(
        "--counties", type=str, required=True,
        help="comma-separated county names whose raw tiles are still present locally",
    )
    parser.add_argument(
        "--out_dir", type=str, default="outputs/chip_audit",
        help="where to write orphan S3 key list files for review",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    counties = [c.strip() for c in args.counties.split(",") if c.strip()]
    counties_safe = [safe_name(c) for c in counties]

    # recompute the CURRENTLY-needed (row, col) cells per CRS group from
    # these counties' local raw tiles (same math ensure_canonical_mosaic_for_counties uses)
    needed: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    skipped_counties = []
    for county, county_safe in zip(counties, counties_safe):
        subgroups = county_tiles_by_crs(county_safe)
        if not subgroups:
            skipped_counties.append(county)
            continue
        for epsg_code, tiles in subgroups.items():
            native_bounds = _raster_bounds_native(tiles)
            canon_bounds = _bounds_to_canonical(native_bounds, epsg_code, buffer_chips=0.0)
            for cell in grid_cells_for_bounds(canon_bounds):
                needed[epsg_code].add(cell)

    if skipped_counties:
        print(
            f"WARNING: no local raw tiles for {skipped_counties} - skipped. "
            "Likely already archived+pruned; their needed cells can't be verified "
            "without restoring from the S3 raw tile archive first. Do NOT trust "
            "an orphan verdict for a CRS group unless ALL its contributing "
            "counties were included in --counties.",
            flush=True,
        )

    if not needed:
        print("No CRS groups to audit (no local raw tiles found for the given counties).")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    for epsg_code, needed_cells in sorted(needed.items()):
        epsg_num = epsg_code.split(":")[-1]
        prefix = f"{S3_PREFIX}/{epsg_num}/chips/"
        paginator = s3.get_paginator("list_objects_v2")

        total = 0
        orphan_keys = []
        orphan_bytes = 0
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                total += 1
                m = CHIP_KEY_RE.search(key)
                if not m:
                    continue
                row, col = int(m.group(2)), int(m.group(3))
                if (row, col) not in needed_cells:
                    orphan_keys.append(key)
                    orphan_bytes += obj["Size"]

        print(
            f"\n{epsg_code}: {total} objects in S3, {len(needed_cells)} cells in current grid, "
            f"{len(orphan_keys)} orphan object(s) ({orphan_bytes / 1e9:.2f} GB)",
            flush=True,
        )

        if orphan_keys:
            out_file = os.path.join(args.out_dir, f"orphans_{epsg_num}.txt")
            with open(out_file, "w") as f:
                f.write("\n".join(orphan_keys))
            print(f"  orphan keys written to {out_file} - review before deleting anything")


if __name__ == "__main__":
    main()

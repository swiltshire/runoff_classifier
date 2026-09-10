"""
CLI wrapper around `ensure_canonical_mosaic_for_counties()` so long-running
canonical chip generation can be launched as a detached background process
(e.g. via `nohup`) instead of inside a Jupyter kernel - immune to notebook
session disconnects / SageMaker Studio forced re-logins killing the kernel.

Usage:
    mkdir -p outputs/logs
    nohup python -u scripts/prepare_canonical_mosaic.py --counties "Benton,Boone,Cass" \
        > outputs/logs/canonical_mosaic_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    (-u matters: with stdout redirected to a file, Python block-buffers plain
    print() calls, so the log can appear frozen for long stretches otherwise.)

    # tail the live log:
    tail -f outputs/logs/canonical_mosaic_<timestamp>.log

    # confirm it's still running:
    ps aux | grep prepare_canonical_mosaic

Do not run this at the same time as an equivalent notebook cell for the same
counties - both would race over the same shared per-CRS-group VRT cache files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# add project root and src/ to path (prepare_reprojected_tiles.py internally
# does `from utils.indiana_cogs import ...`, which requires src/ on sys.path)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.prepare_reprojected_tiles import (  # noqa: E402
    BORDER_BUFFER_CHIPS,
    CHIP_SIZE_PX,
    county_has_manifest,
    ensure_canonical_mosaic_for_counties,
    fetch_county_canonical_chips_from_s3,
)
from utils.indiana_cogs import safe_name  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensure canonical (mosaic-then-reproject) imagery chips exist for the given counties."
    )
    parser.add_argument(
        "--counties", type=str, required=True,
        help="comma-separated county names, e.g. 'Benton,Boone,Cass'",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="force regeneration of all needed chips, even if already cached",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="skip the manifest fast path and run the full mosaic pipeline for every "
             "county, even those with a completion manifest in S3 (implied by --force)",
    )
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--chip_size_px", type=int, default=CHIP_SIZE_PX)
    parser.add_argument("--border_buffer_chips", type=float, default=BORDER_BUFFER_CHIPS)
    return parser.parse_args()


def main():
    args = parse_args()
    counties = [c.strip() for c in args.counties.split(",") if c.strip()]
    if not counties:
        raise SystemExit("--counties produced an empty list; nothing to do")

    print(f"[prepare_canonical_mosaic] starting for {len(counties)} counties: {counties}", flush=True)
    t0 = time.time()

    # Manifest fast path: a county's S3 manifest is only written at the very
    # END of a fully successful ensure pass, so its presence means every chip
    # the county needs is already cached in S3. For those counties we can
    # just fetch the chips per manifest (size-verified) - no raw-tile
    # restore/download at all, which the full pipeline would otherwise need
    # merely to recompute the chip list from raw tile footprints. Counties
    # WITHOUT a manifest (never completed, e.g. an interrupted run) go
    # through the full pipeline as before.
    result = {}
    if args.force or args.full:
        full_counties = counties
    else:
        full_counties = []
        for county in counties:
            county_safe = safe_name(county)
            if county_has_manifest(county_safe):
                print(f"[prepare_canonical_mosaic] {county}: manifest found - fetching chips from S3 "
                      f"(no raw tiles needed)", flush=True)
                result[county_safe] = fetch_county_canonical_chips_from_s3(
                    county, county_safe, verify_sizes=True
                )
            else:
                print(f"[prepare_canonical_mosaic] {county}: no completion manifest - full pipeline", flush=True)
                full_counties.append(county)

    if full_counties:
        result.update(ensure_canonical_mosaic_for_counties(
            counties=full_counties,
            force=args.force,
            max_workers=args.max_workers,
            chip_size_px=args.chip_size_px,
            border_buffer_chips=args.border_buffer_chips,
        ))

    elapsed = time.time() - t0
    total_chips = sum(len(v) for v in result.values())
    print(
        f"[prepare_canonical_mosaic] done in {elapsed / 60:.1f} min. "
        f"{len(result)} counties, {total_chips} total canonical chips.",
        flush=True,
    )


if __name__ == "__main__":
    main()

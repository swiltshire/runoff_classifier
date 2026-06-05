# utils/prepare_reprojected_tiles.py

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import rasterio
import boto3

from utils.indiana_cogs import (
    download_6in_tiles,
    project_root,
    safe_name,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CANONICAL_CRS = "EPSG:2968"          # Indiana West (ft)
CANONICAL_RES = "0.5 0.5"
S3_BUCKET = "sagemaker-gst-stage.sharing"
S3_PREFIX = "serge-wiltshire/runoff-classifier-data/reprojected_imagery"

GDALWARP = "gdalwarp"

# prevent gdal internal oversubscription
os.environ.setdefault("GDAL_NUM_THREADS", "1")

s3 = boto3.client("s3")

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------

def log(msg: str):
    print(f"[canonical] {msg}", flush=True)

def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m:02d}m {s:02d}s"
    return f"{s:02d}s"

# ---------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------

def s3_key_for_tile(county: str, tile_name: str) -> str:
    county = safe_name(county)
    return f"{S3_PREFIX}/{county}/tiles/{tile_name}"

def s3_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

def upload_to_s3(local_path: Path, bucket: str, key: str):
    s3.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={"StorageClass": "STANDARD"},
    )

# ---------------------------------------------------------------------
# Reprojection helpers
# ---------------------------------------------------------------------

def tile_epsg(tile_path: Path) -> str:
    with rasterio.open(tile_path) as ds:
        return f"EPSG:{ds.crs.to_epsg()}" if ds.crs else "UNKNOWN"

def needs_reprojection(tile_path: Path) -> bool:
    return tile_epsg(tile_path) != CANONICAL_CRS

def gdal_reproject(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        GDALWARP,
        "-q",
        "-t_srs", CANONICAL_CRS,
        "-tr", *CANONICAL_RES.split(),
        "-tap",
        "-r", "cubic",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        "-overwrite",
        str(src),
        str(dst),
    ]

    subprocess.check_call(cmd)

# ---------------------------------------------------------------------
# Parallel worker (must be top-level)
# ---------------------------------------------------------------------

def process_one_tile(
    tile: Path,
    county_safe: str,
    *,
    force: bool,
) -> Dict[str, int]:
    tile_name = tile.name
    s3_key = s3_key_for_tile(county_safe, tile_name)

    stats = {
        "skipped": 0,
        "reprojected": 0,
        "uploaded": 0,
    }

    if not force and s3_exists(S3_BUCKET, s3_key):
        stats["skipped"] = 1
        return stats

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        out_tile = tile

        if needs_reprojection(tile):
            out_tile = tmpdir / tile_name
            gdal_reproject(tile, out_tile)
            stats["reprojected"] = 1

        upload_to_s3(out_tile, S3_BUCKET, s3_key)
        stats["uploaded"] = 1

    return stats

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def ensure_canonical_tiles_for_county(
    county: str,
    *,
    force: bool = False,
    max_workers: int = 16,
    cleanup: bool = False,
):
    start = time.time()
    county_safe = safe_name(county)

    log(f"County={county_safe}")

    # 1. ensure raw tiles exist locally
    info = download_6in_tiles(county, max_workers=max_workers)
    local_tiles = list(Path(info["output_dir"]).glob("*.tif"))
    log(f"  Found {len(local_tiles)} tiles locally")

    # CRS diagnostics
    crs_counts = Counter(tile_epsg(t) for t in local_tiles)
    log("  Tile CRS distribution:")
    for crs, n in crs_counts.items():
        log(f"    {crs}: {n}")

    skipped = reprojected = uploaded = 0

    workers = min(max_workers, os.cpu_count() or 1)
    log(f"  Using {workers} parallel workers")

    desc = f"{county_safe} 06in → canonical"

    with tqdm(total=len(local_tiles), unit="tile", desc=desc) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_one_tile,
                    tile,
                    county_safe,
                    force=force,
                )
                for tile in local_tiles
            ]

            for fut in as_completed(futures):
                result = fut.result()

                skipped += result["skipped"]
                reprojected += result["reprojected"]
                uploaded += result["uploaded"]

                pbar.update(1)
                pbar.set_postfix(
                    skip=skipped,
                    reproj=reprojected,
                    upload=uploaded,
                )

    elapsed = fmt_time(time.time() - start)
    log(
        f"Summary {county_safe}: "
        f"total={len(local_tiles)}, "
        f"skipped={skipped}, "
        f"reprojected={reprojected}, "
        f"uploaded={uploaded}, "
        f"elapsed={elapsed}"
    )

    # 2. optionally cleanup local tiles
    if cleanup:
        # import shutil
        import subprocess
        output_dir = Path(info["output_dir"])
        log(f"  Cleaning up local tiles in {output_dir}")
        # shutil.rmtree(output_dir) # this was VERY slow
        subprocess.run(["rm", "-rf", output_dir], check=True) # use os-native deletion
        log(f"  ✓ Deleted {len(local_tiles)} tiles ({output_dir})")

def ensure_canonical_tiles_for_counties(
    counties: List[str],
    *,
    force: bool = False,
    cleanup: bool = False,
):
    log(f"Preparing canonical tiles for {len(counties)} counties (cleanup={cleanup})")
    for county in counties:
        ensure_canonical_tiles_for_county(county, force=force, cleanup=cleanup)
    log("All requested counties processed")

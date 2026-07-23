# utils/prepare_reprojected_tiles.py

import os
import shutil
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
    CANONICAL_CRS as DEFAULT_CANONICAL_CRS,
    set_reference_crs,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Load persisted reference CRS from previous Phase 0 run, or use default
CANONICAL_CRS = DEFAULT_CANONICAL_CRS
_crs_config_file = project_root() / "outputs" / ".reference_crs"
if _crs_config_file.exists():
    CANONICAL_CRS = _crs_config_file.read_text().strip()
    set_reference_crs(CANONICAL_CRS)

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
    except (s3.exceptions.ClientError, Exception):
        # Catch ClientError (404, etc) and connection errors
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

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdalwarp failed: {e}")

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
        "replaced_local": 0,
    }

    if not force and s3_exists(S3_BUCKET, s3_key):
        stats["skipped"] = 1
        return stats

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        out_tile = tile
        was_reprojected = False

        if needs_reprojection(tile):
            out_tile = tmpdir / tile_name
            gdal_reproject(tile, out_tile)
            stats["reprojected"] = 1
            was_reprojected = True

        upload_to_s3(out_tile, S3_BUCKET, s3_key)
        stats["uploaded"] = 1

        # Replace local tile with reprojected version (before temp cleanup)
        if was_reprojected:
            shutil.copy2(out_tile, tile)
            stats["replaced_local"] = 1

    return stats

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def ensure_canonical_tiles_for_county(
    county: str,
    *,
    force: bool = False,
    max_workers: int = 16,
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
    replaced_local = 0

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
                replaced_local += result["replaced_local"]

                pbar.update(1)
                pbar.set_postfix(
                    skip=skipped,
                    reproj=reprojected,
                    upload=uploaded,
                    repl=replaced_local,
                )

    elapsed = fmt_time(time.time() - start)
    log(
        f"Summary {county_safe}: "
        f"total={len(local_tiles)}, "
        f"skipped={skipped}, "
        f"reprojected={reprojected}, "
        f"uploaded={uploaded}, "
        f"replaced_local={replaced_local}, "
        f"elapsed={elapsed}"
    )

def ensure_canonical_tiles_for_counties(
    counties: List[str],
    *,
    force: bool = False,
    skip_counties: List[str] = None,
):
    """Ensure canonical tiles for multiple counties with checkpoint resume.
    
    Maintains a checkpoint file (.canonical_crs_fix_checkpoint) to track processed counties.
    If interrupted, will resume from where it left off on next run.
    
    Args:
        counties: List of counties to process
        force: Force re-check even if already uploaded to S3
        skip_counties: Optional list of counties to mark as already-processed (useful for resuming)
    """
    checkpoint_file = project_root() / "outputs" / ".canonical_crs_fix_checkpoint"
    
    # Load already-processed counties
    processed = set()
    if checkpoint_file.exists():
        processed = set(checkpoint_file.read_text().strip().split("\n"))
    
    # Add manually-specified skip counties to checkpoint
    if skip_counties:
        processed.update(skip_counties)
    
    # Filter to counties still needing processing
    remaining = [c for c in counties if c not in processed]
    
    if not remaining:
        log(f"All {len(counties)} counties already processed (checkpoint found)")
        return
    
    log(f"Preparing canonical tiles for {len(counties)} counties ({len(remaining)} remaining)")
    
    for county in remaining:
        try:
            ensure_canonical_tiles_for_county(county, force=force)
            # Mark as processed
            processed.add(county)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file.write_text("\n".join(sorted(processed)))
        except Exception as e:
            log(f"ERROR processing {county}: {e}")
            raise
    
    log(f"All requested counties processed")

# utils/prepare_reprojected_tiles.py
"""
Prepares canonical (seamlessly reprojected) imagery chips for training and
inference, using a "mosaic-then-reproject" architecture.

Why: the previous approach reprojected each raw tile independently
(`gdalwarp` per tile). Because a lone tile has no neighboring pixels for the
resampler to draw context from at its own edges, and CRS reprojection skews
originally-rectangular tile footprints, the result was triangular NoData
slivers visible at tile boundaries once mosaicked in ArcGIS Pro. `-tap`
(target-aligned-pixels) kept the output grid aligned, but did not fix the
missing edge context.

New architecture:
  1. Raw county tiles (native CRS - possibly several different EPSG codes
     across the training set, and occasionally >1 CRS within a single
     county) are downloaded via `download_6in_tiles()` and kept pristine on
     local disk at `data/counties/{county}/tiles/` - nothing here overwrites
     raw tiles in place anymore. Any raw tiles previously archived to our
     own S3 bucket (see step 6) are restored first via
     `restore_raw_tiles_from_archive()`, so a re-added county in an
     already-processed CRS group doesn't need to re-fetch from Indiana's
     ArcGIS REST service.
  2. ALL locally-available raw tiles (across every county ever downloaded,
     not just the counties requested in a given call) are grouped by native
     EPSG code, so every warp has the maximum possible spatial context.
  3. Native-CRS groups are processed ONE AT A TIME (not all at once), so at
     most one group's raw tiles are ever resident on local disk
     simultaneously. For each group: a mosaic VRT is built from every tile
     in that group (`gdalbuildvrt`), then warped ONCE to the canonical
     CRS/resolution as a lazy "warped VRT" (`gdalwarp -of VRT`). Because the
     resampler always sees the full mosaic, there is no tile-edge starvation.
  4. A fixed-size, tap-aligned "canonical chip" grid - anchored at the
     global origin (0, 0) and independent of any tile's original boundary -
     is used to crop real GeoTIFF chips out of that group's warped VRT
     (`gdal_translate -projwin`: pure extraction, no further resampling).
     Chips are cached in S3 + a local shared cache keyed by
     (epsg, row, col) - NOT by county or original tile name - so the same
     physical chip is reused for training and inference regardless of which
     counties happen to be requested in a given call.
  5. Each chip is materialized immediately (inside the same worker that
     crops/uploads it) as a hardlink (falling back to a copy) into every
     requesting county's `canonical_tiles/` folder, then the shared local
     staging copy is deleted right away. This keeps peak local disk usage
     from the chip cache bounded to roughly "chips currently in flight"
     rather than accumulating a second full local copy of the entire
     (potentially multi-TB) canonical dataset. `canonical_tiles/` still
     gives the rest of the pipeline (training VRT assembly, inference,
     notebook widgets) a familiar per-county folder layout without needing
     to know about the shared cache.
  6. Raw tiles for every requested county are kept resident on local disk
     for the ENTIRE call - never pruned mid-run - so every native-CRS
     group's mosaic is always built from its complete, final raw-tile set
     from the very start. This is what makes border invalidation
     unnecessary to reason about across separate calls: there is no
     scenario where a group gets mosaicked from a partial/context-starved
     subset of its tiles. Instead, local disk is bounded from the OTHER
     side: as soon as a group's chips are generated and durably uploaded to
     S3 (inside `_crop_and_cache_chip`), that group's local
     `canonical_tiles/` copies are deleted immediately (they're safe to
     lose - already durable in S3). Once every group has been processed,
     ALL raw tiles used this run are backed up to our own S3 bucket
     (`RAW_ARCHIVE_S3_PREFIX`), verified, then deleted locally in one
     deferred pass (`archive_and_prune_raw_tiles()`) - and finally, every
     requested county's `canonical_tiles/` is repopulated by fetching its
     chips back from S3 via its manifest
     (`fetch_county_canonical_chips_from_s3()`). So at any given moment,
     local disk holds at most: all requested raw tiles, plus one CRS
     group's canonical chips - never the full multi-group canonical total
     mid-run, and never raw tiles for only a subset of a group.

Border invalidation: when a genuinely new county is added whose raw tiles
fall into a CRS group that already has cached chips, any existing chip
within `BORDER_BUFFER_CHIPS` chip-widths of the new tiles' bounding box is
force-regenerated from the now-larger native mosaic, since it may have been
produced without the new neighboring data as resampling context. This keeps
incremental runs fast (only new + border-adjacent chips are (re)computed)
without silently leaving stale slivers at the seam between old and
newly-added counties.
"""

import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import boto3
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from tqdm import tqdm

from utils.indiana_cogs import (
    download_6in_tiles,
    load_training_imagery_years,
    normalize_county_key,
    project_root,
    safe_name,
    CANONICAL_CRS as DEFAULT_CANONICAL_CRS,
    set_reference_crs,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Load persisted reference CRS from a previous Phase 0 run, or use default
CANONICAL_CRS = DEFAULT_CANONICAL_CRS
_crs_config_file = project_root() / "outputs" / ".reference_crs"
if _crs_config_file.exists():
    CANONICAL_CRS = _crs_config_file.read_text().strip()
    set_reference_crs(CANONICAL_CRS)

CANONICAL_RES = 0.5  # feet/pixel (6-inch imagery)
CHIP_SIZE_PX = 8192  # ~4096 x 4096 ft per chip at 0.5 ft resolution
BORDER_BUFFER_CHIPS = 1.0  # force-regen radius (in chips) around newly-added counties' tiles

S3_BUCKET = "sagemaker-gst-stage.sharing"
S3_PREFIX = "serge-wiltshire/runoff-classifier-data/canonical_mosaic"
RAW_ARCHIVE_S3_PREFIX = "serge-wiltshire/runoff-classifier-data/raw_tile_archive"

GDALBUILDVRT = "gdalbuildvrt"
GDALWARP = "gdalwarp"
GDAL_TRANSLATE = "gdal_translate"

# prevent gdal internal oversubscription (we parallelize at the process level instead)
os.environ.setdefault("GDAL_NUM_THREADS", "1")

s3 = boto3.client("s3")

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------

def log(msg: str):
    print(f"[mosaic] {msg}", flush=True)

def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m:02d}m {s:02d}s"
    return f"{s:02d}s"

# ---------------------------------------------------------------------
# S3 / local cache helpers
# ---------------------------------------------------------------------

def chip_filename(epsg_code: str, row: int, col: int) -> str:
    epsg_num = epsg_code.split(":")[-1]
    return f"chip_{epsg_num}_r{row}_c{col}.tif"

def chip_s3_key(epsg_code: str, row: int, col: int) -> str:
    epsg_num = epsg_code.split(":")[-1]
    return f"{S3_PREFIX}/{epsg_num}/chips/{chip_filename(epsg_code, row, col)}"

def chip_local_cache_path(epsg_code: str, row: int, col: int) -> Path:
    epsg_num = epsg_code.split(":")[-1]
    return project_root() / "data" / "_canonical_chip_cache" / epsg_num / chip_filename(epsg_code, row, col)

def s3_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def upload_to_s3(local_path: Path, bucket: str, key: str):
    s3.upload_file(str(local_path), bucket, key)

def county_manifest_s3_key(county_safe: str) -> str:
    return f"{S3_PREFIX}/manifests/{county_safe}.json"

def write_county_manifest(county: str, county_safe: str, cells: List[Tuple[str, int, int]]) -> str:
    """Upload a small JSON manifest listing every canonical chip S3 key a
    single county needs (`{epsg_code, row, col}` triples -> full S3 keys via
    chip_s3_key()). Lets a collaborator download just one county's chips
    directly (e.g. via a PowerShell + AWS CLI script) without needing to
    know anything about the native-CRS folder layout or replicate any of
    the grid/CRS math in this file. Returns the S3 key the manifest was
    written to."""
    keys = sorted({chip_s3_key(epsg_code, row, col) for (epsg_code, row, col) in cells})
    manifest = {
        "county": county,
        "bucket": S3_BUCKET,
        "chip_size_px": CHIP_SIZE_PX,
        "canonical_crs": CANONICAL_CRS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "keys": keys,
    }
    key = county_manifest_s3_key(county_safe)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return key

def download_from_s3(local_path: Path, bucket: str, key: str):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))

def fetch_county_canonical_chips_from_s3(county: str, county_safe: str) -> List[Path]:
    """Download every canonical chip listed in a county's S3 manifest
    (written by write_county_manifest()) into its local canonical_tiles/
    folder, skipping any chip already present locally with the right name.
    Python port of scripts/fetch_canonical.ps1's logic, for use on
    SageMaker/Linux where the pipeline itself runs. Used internally by
    ensure_canonical_mosaic_for_counties() to repopulate canonical_tiles/
    after generation (chips are pruned locally per-group during generation
    to bound disk usage), and can also be called standalone to restore a
    county's local chips later without re-running the whole sweep."""
    manifest_key = county_manifest_s3_key(county_safe)
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
    except Exception as e:
        log(f"  ✗ {county}: no manifest found in S3 ({manifest_key}): {e}")
        return []
    manifest = json.loads(obj["Body"].read())
    keys = manifest.get("keys", [])

    county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
    county_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for key in keys:
        local_path = county_dir / Path(key).name
        if not local_path.exists():
            download_from_s3(local_path, S3_BUCKET, key)
        paths.append(local_path)
    return sorted(paths)

# ---------------------------------------------------------------------
# Raw tile S3 archive (backup-then-delete lifecycle, so raw tiles and
# canonical output never both have to be fully resident on local disk at
# the same time)
# ---------------------------------------------------------------------

def raw_tile_s3_key(county_safe: str, filename: str) -> str:
    return f"{RAW_ARCHIVE_S3_PREFIX}/{county_safe}/{filename}"

def restore_raw_tiles_from_archive(county_safe: str) -> int:
    """Download any raw tiles already archived in S3 for this county into its
    local tiles/ folder. Lets an incremental re-add of a county whose native
    CRS group was previously processed (and its raw tiles pruned) restore
    from our own S3 archive instead of re-fetching from Indiana's ArcGIS REST
    service. download_6in_tiles()'s own download_one() already skips valid
    existing local files by size/header, so this is a pure win - only the
    lightweight attribute/metadata query still hits Indiana, not the bulk
    tile bytes."""
    prefix = f"{RAW_ARCHIVE_S3_PREFIX}/{county_safe}/"
    dest_dir = project_root() / "data" / "counties" / county_safe / "tiles"
    dest_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]
            if not filename or "/" in filename:
                continue
            local_path = dest_dir / filename
            if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                continue
            s3.download_file(S3_BUCKET, key, str(local_path))
            restored += 1
    return restored

def archive_and_prune_raw_tiles(tile_paths: List[Path]) -> None:
    """Back up raw tiles to our own S3 bucket, verify the upload, then delete
    the local copy - so a native-CRS group's raw tiles don't have to coexist
    locally with the canonical output generated from them."""
    verified: List[Path] = []
    uploaded = already_archived = 0
    with tqdm(total=len(tile_paths), unit="tile", desc="  archiving raw tiles") as pbar:
        for tile_path in tile_paths:
            county_safe = tile_path.parent.parent.name
            key = raw_tile_s3_key(county_safe, tile_path.name)
            local_size = tile_path.stat().st_size

            if s3_exists(S3_BUCKET, key):
                already_archived += 1
            else:
                upload_to_s3(tile_path, S3_BUCKET, key)
                uploaded += 1

            try:
                head = s3.head_object(Bucket=S3_BUCKET, Key=key)
                if head["ContentLength"] != local_size:
                    log(f"  \u26a0 WARNING: archive size mismatch for {tile_path.name} - not deleting local copy")
                    pbar.update(1)
                    continue
            except Exception as e:
                log(f"  \u26a0 WARNING: could not verify archive for {tile_path.name} ({e}) - not deleting local copy")
                pbar.update(1)
                continue

            verified.append(tile_path)
            pbar.update(1)
            pbar.set_postfix(uploaded=uploaded, cached=already_archived)

    for tile_path in verified:
        try:
            tile_path.unlink()
        except OSError:
            pass

    log(f"  archived + pruned {len(verified)}/{len(tile_paths)} raw tiles")

# ---------------------------------------------------------------------
# Native tile CRS grouping (with a small on-disk cache per county tiles/
# folder, since scanning every tile's header gets slow once many counties
# have accumulated locally)
# ---------------------------------------------------------------------

def tile_epsg(tile_path: Path) -> str:
    with rasterio.open(tile_path) as ds:
        return f"EPSG:{ds.crs.to_epsg()}" if ds.crs else "UNKNOWN"

def _tile_epsg_map(tiles_dir: Path) -> Dict[str, str]:
    """Return {filename: epsg} for every *.tif in tiles_dir, using (and refreshing) an on-disk cache."""
    cache_path = tiles_dir / ".tile_crs_cache.json"
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    tif_paths = list(tiles_dir.glob("*.tif"))
    changed = False
    for p in tif_paths:
        if p.name not in cache:
            cache[p.name] = tile_epsg(p)
            changed = True

    # drop stale entries for files that no longer exist
    present_names = {p.name for p in tif_paths}
    stale = set(cache) - present_names
    if stale:
        for name in stale:
            cache.pop(name, None)
        changed = True

    if changed:
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass

    return cache

def local_raw_tiles_by_crs() -> Dict[str, List[Path]]:
    """Scan every county currently present under data/counties/*/tiles and group ALL raw tiles by native EPSG."""
    counties_dir = project_root() / "data" / "counties"
    groups: Dict[str, List[Path]] = defaultdict(list)
    if not counties_dir.is_dir():
        return groups
    for county_dir in sorted(counties_dir.iterdir()):
        tiles_dir = county_dir / "tiles"
        if not tiles_dir.is_dir():
            continue
        for name, epsg in _tile_epsg_map(tiles_dir).items():
            groups[epsg].append(tiles_dir / name)
    return groups

def county_tiles_by_crs(county_safe: str) -> Dict[str, List[Path]]:
    """Group just one county's own raw tiles by native EPSG (a county can span >1 CRS)."""
    tiles_dir = project_root() / "data" / "counties" / county_safe / "tiles"
    groups: Dict[str, List[Path]] = defaultdict(list)
    if not tiles_dir.is_dir():
        return groups
    for name, epsg in _tile_epsg_map(tiles_dir).items():
        groups[epsg].append(tiles_dir / name)
    return groups

# ---------------------------------------------------------------------
# Grid / bounds math
# ---------------------------------------------------------------------

def _tap_align(minx: float, miny: float, maxx: float, maxy: float, res: float) -> Tuple[float, float, float, float]:
    """Snap a bbox outward to the global tap grid (multiples of res, anchored at the origin) - same convention as gdalwarp's -tap."""
    ax = math.floor(minx / res) * res
    ay = math.floor(miny / res) * res
    bx = math.ceil(maxx / res) * res
    by = math.ceil(maxy / res) * res
    return ax, ay, bx, by

def _raster_bounds_native(tiles: List[Path]) -> Tuple[float, float, float, float]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for p in tiles:
        with rasterio.open(p) as ds:
            b = ds.bounds
            minx = min(minx, b.left)
            miny = min(miny, b.bottom)
            maxx = max(maxx, b.right)
            maxy = max(maxy, b.top)
    return minx, miny, maxx, maxy

def _bounds_to_canonical(
    bounds: Tuple[float, float, float, float], src_epsg: str, buffer_chips: float = 0.0
) -> Tuple[float, float, float, float]:
    """Reproject a native-CRS bbox into CANONICAL_CRS (densified for curvature), buffer outward by `buffer_chips` chip-widths, then tap-align."""
    minx, miny, maxx, maxy = transform_bounds(src_epsg, CANONICAL_CRS, *bounds, densify_pts=21)
    chip_units = CHIP_SIZE_PX * CANONICAL_RES
    buf = buffer_chips * chip_units
    return _tap_align(minx - buf, miny - buf, maxx + buf, maxy + buf, CANONICAL_RES)

def grid_cells_for_bounds(bounds: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
    """List (row, col) grid cells (in the GLOBAL, origin-anchored canonical grid) overlapping bounds."""
    minx, miny, maxx, maxy = bounds
    chip_units = CHIP_SIZE_PX * CANONICAL_RES
    col_start = math.floor(minx / chip_units)
    col_end = math.floor((maxx - 1e-6) / chip_units)
    row_start = math.floor(miny / chip_units)
    row_end = math.floor((maxy - 1e-6) / chip_units)
    return [(r, c) for r in range(row_start, row_end + 1) for c in range(col_start, col_end + 1)]

def _needed_cells_for_tiles(tiles: List[Path], epsg_code: str, buffer_chips: float = 0.0) -> Set[Tuple[int, int]]:
    """Canonical grid cells actually covered by `tiles` - the UNION of each
    tile's own footprint, NOT the group's overall bounding rectangle.

    Using a single overall bbox (via _raster_bounds_native) is wrong whenever
    a native-CRS group's tiles are sparse/scattered relative to their own
    bounding rectangle: the bbox can span area that a DIFFERENT native-CRS
    group's tiles actually cover, causing this group to also claim (and
    generate a genuinely-blank chip for) a canonical cell that's really
    covered by the other group. Computing per-tile means each group only
    ever claims cells its own tiles actually touch. Same per-tile file-header
    I/O cost as _raster_bounds_native (which already opens every tile)."""
    cells: Set[Tuple[int, int]] = set()
    for p in tiles:
        with rasterio.open(p) as ds:
            b = ds.bounds
        canon_bounds = _bounds_to_canonical((b.left, b.bottom, b.right, b.top), epsg_code, buffer_chips=buffer_chips)
        cells.update(grid_cells_for_bounds(canon_bounds))
    return cells

def chip_bounds(row: int, col: int) -> Tuple[float, float, float, float]:
    chip_units = CHIP_SIZE_PX * CANONICAL_RES
    minx = col * chip_units
    miny = row * chip_units
    return minx, miny, minx + chip_units, miny + chip_units

# ---------------------------------------------------------------------
# GDAL subprocess helpers (native mosaic + warped VRT per CRS group)
# ---------------------------------------------------------------------

def _mosaic_workdir() -> Path:
    d = project_root() / "outputs" / "_native_mosaic_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def build_native_vrt(epsg_code: str, tile_paths: List[Path]) -> Path:
    workdir = _mosaic_workdir()
    epsg_num = epsg_code.split(":")[-1]
    list_file = workdir / f"native_{epsg_num}_filelist.txt"
    list_file.write_text("\n".join(str(p) for p in tile_paths))
    out_vrt = workdir / f"native_{epsg_num}.vrt"
    if out_vrt.exists():
        out_vrt.unlink()
    # -srcnodata/-vrtnodata 0: these raw tiles carry no intrinsic NoDataValue
    # or mask/alpha band of their own, so without this, gdalbuildvrt's default
    # overlap compositing ("last listed source wins, full opaque paint") would
    # let a later-listed tile's 0-valued edge/padding pixels silently
    # overwrite real imagery from an earlier-listed tile in the same overlap
    # region - it has no way to know 0 means "no data" here otherwise.
    # Declaring 0 as nodata lets gdalbuildvrt see through those pixels to
    # whichever source actually has real data underneath, instead of just
    # picking whichever tile happens to be listed last (tile order here is
    # arbitrary directory-listing order, not quality-ranked). 0 is also the
    # same sentinel value is_fully_blank() already treats as "no imagery"
    # everywhere else in this module, so this is consistent, not a new
    # convention - and true all-band (R,G,B,IR all exactly 0) pixels don't
    # occur in real photographed imagery.
    subprocess.check_call([
        GDALBUILDVRT, "-q",
        "-srcnodata", "0", "-vrtnodata", "0",
        "-input_file_list", str(list_file), str(out_vrt),
    ])
    return out_vrt

def build_warped_vrt(native_vrt: Path, epsg_code: str) -> Path:
    workdir = _mosaic_workdir()
    epsg_num = epsg_code.split(":")[-1]
    out_vrt = workdir / f"warped_{epsg_num}.vrt"
    if out_vrt.exists():
        out_vrt.unlink()
    cmd = [
        GDALWARP, "-q", "-of", "VRT",
        "-t_srs", CANONICAL_CRS,
        "-tr", str(CANONICAL_RES), str(CANONICAL_RES),
        "-tap", "-r", "cubic", "-overwrite",
        # keep the same 0-as-nodata declaration through the reprojection -
        # explicit rather than relying on gdalwarp's implicit "inherit
        # nodata from source" default, so the warped VRT (and every chip
        # gdal_translate'd from it) reliably carries a real NoDataValue tag.
        "-srcnodata", "0", "-dstnodata", "0",
        str(native_vrt), str(out_vrt),
    ]
    subprocess.check_call(cmd)
    return out_vrt

# ---------------------------------------------------------------------
# Blank-chip detection + logging
#
# The chip cache is keyed only by (epsg, row, col) and reused indefinitely
# (see module docstring). If a raw tile silently fails to download (see
# indiana_cogs.download_6in_tiles's failed-download warning above) or a
# CRS group's native mosaic otherwise has a transient gap, the chip cropped
# from that gap is all-NoData and gets cached in S3 forever - nothing else
# in the pipeline ever revisits it. These helpers make that condition
# visible at generation time and queryable later via
# `rescan_and_fix_blank_chips()`.
# ---------------------------------------------------------------------

BLANK_CHIP_LOG_PATH = None  # set lazily via _blank_chip_log_path()


def _blank_chip_log_path() -> Path:
    p = project_root() / "outputs" / "blank_chip_log.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def is_fully_blank(path: Path) -> bool:
    """True if every pixel in `path` is NoData (or exactly zero, when no
    NoData value is set) across all bands. Public - also used directly by
    notebook diagnostics (e.g. the "Validate inference imagery" cell) to
    flag 100%-blank canonical tiles."""
    with rasterio.open(path) as src:
        arr = src.read()
        nodata_val = src.nodata
        if nodata_val is not None:
            return bool(np.all(arr == nodata_val))
        return bool(np.all(arr == 0))


def _log_blank_chip(epsg_code: str, row: int, col: int, s3_key: str) -> None:
    log_path = _blank_chip_log_path()
    entries = []
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text())
        except (json.JSONDecodeError, OSError):
            entries = []
    entries.append({
        "epsg": epsg_code,
        "row": row,
        "col": col,
        "s3_key": s3_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    log_path.write_text(json.dumps(entries, indent=2))

# ---------------------------------------------------------------------
# Per-chip crop worker (top-level function so it is picklable for
# ProcessPoolExecutor)
# ---------------------------------------------------------------------

def _crop_and_cache_chip(job: Dict) -> Dict[str, int]:
    epsg_code = job["epsg"]
    row = job["row"]
    col = job["col"]
    warped_vrt_path = job["warped_vrt_path"]
    force = job["force"]
    counties_needing_this_cell: List[str] = job["counties"]

    stats = {"skipped": 0, "generated": 0, "downloaded": 0, "retried_blank": 0}
    s3_key = chip_s3_key(epsg_code, row, col)
    local_path = chip_local_cache_path(epsg_code, row, col)

    use_cache = not force and s3_exists(S3_BUCKET, s3_key)
    if use_cache:
        if not local_path.exists():
            download_from_s3(local_path, S3_BUCKET, s3_key)
            stats["downloaded"] = 1

        # Never trust a cached hit blindly - the chip cache is keyed only by
        # (epsg, row, col), NEVER by county, and is NOT invalidated by
        # re-downloading raw tiles or wiping a county's own local/S3 data.
        # If this exact cell was ever cached blank by a bug that's since been
        # fixed elsewhere in the pipeline (stale mosaic gap, wrong
        # needed-cells grouping, truncated write, etc.), it would otherwise
        # be silently re-served forever. Re-cropping from the CURRENT warped
        # mosaic is cheap relative to a full run and harmless even if the
        # cell is a genuine coverage gap (it will just come out blank again
        # and get re-logged below) - so always re-derive rather than trust a
        # blank cache hit.
        if is_fully_blank(local_path):
            use_cache = False
            stats["retried_blank"] = 1

    if use_cache:
        stats["skipped"] = 1
    else:
        minx, miny, maxx, maxy = chip_bounds(row, col)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp path in the same directory and atomically rename
        # into place once gdal_translate has fully succeeded and the result
        # has been read back cleanly. This guarantees a worker that gets
        # killed/OOM-killed/interrupted (e.g. during a large parallel
        # force-rebuild) or hits a disk-full condition mid-write can never
        # leave a truncated/corrupted GeoTIFF sitting at `local_path` - the
        # rename is the only thing that can make a bad file appear at the
        # final path, and os.replace() is atomic on both POSIX and Windows.
        tmp_path = local_path.with_name(local_path.name + f".tmp{os.getpid()}")
        try:
            cmd = [
                GDAL_TRANSLATE,
                "-of", "GTiff",  # tmp_path's extension isn't ".tif" (pid suffix
                                 # appended after it), so it can't be auto-detected
                "-projwin", str(minx), str(maxy), str(maxx), str(miny),
                "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
                str(warped_vrt_path), str(tmp_path),
            ]
            # Don't use -q / check_call: on failure we need gdal_translate's
            # stderr (e.g. "Computed -srcwin falls outside raster extent") to
            # diagnose *why* - a bare non-zero exit code alone isn't
            # actionable. Still print stdout/stderr on success too (gdal
            # rarely writes anything without -q, so this is normally silent).
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"gdal_translate failed for chip {s3_key} (exit {proc.returncode}):\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stderr: {proc.stderr.strip()}\n"
                    f"  stdout: {proc.stdout.strip()}"
                )

            # Verify the freshly-written file actually reads back cleanly
            # before trusting it - catches truncated/corrupted output from a
            # gdal_translate that reported success but wrote a bad file
            # (e.g. a delayed disk-full error only surfaced at close()).
            with rasterio.open(tmp_path) as _verify_src:
                _verify_src.read()

            os.replace(tmp_path, local_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        stats["generated"] = 1

        # A freshly-cropped chip that comes out fully blank is suspicious (a
        # gap in the native mosaic - e.g. from a silently-failed raw tile
        # download) rather than expected, and will otherwise get cached in
        # S3 forever with no way to tell later. Log it so it can be found
        # and retried via rescan_and_fix_blank_chips() once raw coverage
        # might have improved, without blocking/slowing this run.
        if is_fully_blank(local_path):
            reason = "still blank after re-cropping a stale cached blank" if stats["retried_blank"] else "generated fully blank/NoData"
            print(f"  \u26a0 WARNING: chip {s3_key} {reason} - caching anyway, "
                  f"logged to {_blank_chip_log_path()} for later rescan")
            _log_blank_chip(epsg_code, row, col, s3_key)

        upload_to_s3(local_path, S3_BUCKET, s3_key)

    # Materialize into every requesting county's canonical_tiles/ folder as a
    # hardlink (falling back to a copy across filesystems) immediately, then
    # drop the shared staging copy - S3 is the durable store; local disk
    # should only transiently hold each chip long enough to link it out, not
    # accumulate a second full local copy of the entire (potentially
    # multi-TB) canonical dataset on top of the raw tiles. This bounds peak
    # local disk usage from the chip cache to roughly "chips currently in
    # flight" instead of "every chip ever generated".
    for county_safe in counties_needing_this_cell:
        county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
        county_dir.mkdir(parents=True, exist_ok=True)
        dst = county_dir / local_path.name
        if dst.exists():
            dst.unlink()
        try:
            os.link(local_path, dst)
        except OSError:
            shutil.copy2(local_path, dst)

    try:
        local_path.unlink()
    except OSError:
        pass

    return stats

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def ensure_canonical_mosaic_for_counties(
    counties: List[str],
    *,
    force: bool = False,
    max_workers: int = 16,
    chip_size_px: int = CHIP_SIZE_PX,
    border_buffer_chips: float = BORDER_BUFFER_CHIPS,
) -> Dict[str, List[Path]]:
    """
    Ensure canonical (seamlessly reprojected) imagery chips exist for
    `counties`, using mosaic-then-reproject instead of per-tile independent
    reprojection. Works the same whether `counties` is the full training
    set or a single new county being pulled in for inference - in both
    cases every relevant native-CRS group is mosaicked from ALL locally
    available raw tiles (not just those belonging to `counties`) before
    warping, so tile-edge context is never starved.

    Each county's raw-tile download is pinned to the imagery year recorded
    in `data/training_county_imagery_years.csv` (matching the year its
    training labels were verified against), when that county has a CSV
    entry. Counties without one (typically inference-only counties with no
    labels) fall back to auto-detecting the newest/most-complete year, with
    a warning logged so the gap is visible rather than silent.

    Returns {county_safe_name: [local canonical_tiles chip paths]}.
    """
    global CHIP_SIZE_PX, BORDER_BUFFER_CHIPS
    CHIP_SIZE_PX = chip_size_px
    BORDER_BUFFER_CHIPS = border_buffer_chips

    start = time.time()
    counties_safe = [safe_name(c.strip()) for c in counties]

    # Pin each county's imagery download to the year its training labels
    # were verified against (data/training_county_imagery_years.csv), so we
    # never silently auto-detect a different (e.g. newer) year than the
    # labels expect. Counties with no CSV entry (e.g. inference-only
    # counties with no labels) fall back to auto-detect, with a loud warning
    # so gaps are visible instead of silent.
    imagery_years: Dict[str, int] = {}
    imagery_years_csv = project_root() / "data" / "training_county_imagery_years.csv"
    if imagery_years_csv.exists():
        imagery_years = load_training_imagery_years(imagery_years_csv)

    # 1. ensure raw tiles exist locally for every requested county, tracking
    #    which ones are genuinely new (had no local raw tiles before this
    #    call) so we know where border-invalidation applies.
    new_counties_safe: List[str] = []
    for county, county_safe in zip(counties, counties_safe):
        tiles_dir = project_root() / "data" / "counties" / county_safe / "tiles"
        existed_before = tiles_dir.is_dir() and any(tiles_dir.glob("*.tif"))
        if not existed_before:
            new_counties_safe.append(county_safe)
            # restore any previously-archived raw tiles from our own S3
            # bucket first (fast/reliable) before falling back to Indiana's
            # ArcGIS REST service for anything still missing.
            restored = restore_raw_tiles_from_archive(county_safe)
            if restored:
                log(f"  {county}: restored {restored} raw tiles from S3 archive")

        pinned_year = imagery_years.get(normalize_county_key(county))
        if pinned_year is None:
            log(f"  ⚠ WARNING: no pinned imagery year for {county} - falling back to auto-detect")

        try:
            download_result = download_6in_tiles(county, max_workers=max_workers, imagery_year=pinned_year)
        except Exception as e:
            log(f"  ✗ {county}: failed to download raw tiles: {e}")
            raise

        if download_result.get("failed", 0) > 0:
            log(
                f"  ⚠ WARNING: {county}: {download_result['failed']} raw tile download(s) failed "
                f"({download_result['year']} imagery) - the native mosaic for this area will have a "
                f"gap where those tiles would have been, and any canonical chip cropped from that gap "
                f"will be cached as blank/NoData. Re-run this county's download (or "
                f"rescan_and_fix_blank_chips([...]) afterward) to repair it."
            )

    # 2. group ALL locally-available raw tiles (every county ever
    #    downloaded) by native CRS, so every warp has maximum context.
    all_groups = local_raw_tiles_by_crs()

    # 3. compute each requested county's own required chip coverage, and
    #    (for genuinely new counties) the border-buffered bbox whose
    #    existing cached chips must be force-regenerated.
    county_needed_cells: Dict[str, List[Tuple[str, int, int]]] = {}
    relevant_epsgs: Set[str] = set()
    force_cells: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)

    for county_safe in counties_safe:
        subgroups = county_tiles_by_crs(county_safe)
        needed: Set[Tuple[str, int, int]] = set()
        for epsg_code, tiles in subgroups.items():
            relevant_epsgs.add(epsg_code)
            for (r, c) in _needed_cells_for_tiles(tiles, epsg_code, buffer_chips=0.0):
                needed.add((epsg_code, r, c))
        county_needed_cells[county_safe] = sorted(needed)

    for county_safe in new_counties_safe:
        subgroups = county_tiles_by_crs(county_safe)
        for epsg_code, tiles in subgroups.items():
            for cell in _needed_cells_for_tiles(tiles, epsg_code, buffer_chips=BORDER_BUFFER_CHIPS):
                force_cells[epsg_code].add(cell)

    if not relevant_epsgs:
        log("No raw tiles found for the requested counties; nothing to do.")
        return {c: [] for c in counties_safe}

    log(f"Requested counties span {len(relevant_epsgs)} native CRS group(s): {sorted(relevant_epsgs)}")

    # 4. clear stale canonical_tiles/*.tif upfront (once, before any group's
    #    jobs run) - workers materialize fresh hardlinks per-chip as their
    #    group's batch completes below, rather than in one big pass at the
    #    end, so this cleanup can't race against in-flight workers.
    for county_safe in counties_safe:
        county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
        county_dir.mkdir(parents=True, exist_ok=True)
        for existing in county_dir.glob("*.tif"):
            existing.unlink()

    # 5. assemble the full chip job list (union of all requested counties'
    #    needed cells), applying force where border-invalidation says so,
    #    and map each cell back to the county/counties that need it so
    #    workers can materialize hardlinks directly.
    all_needed: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    for cells in county_needed_cells.values():
        for (epsg_code, r, c) in cells:
            all_needed[epsg_code].add((r, c))

    cell_to_counties: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)
    for county_safe, cells in county_needed_cells.items():
        for cell in cells:
            cell_to_counties[cell].append(county_safe)

    # 6. process ONE native-CRS group at a time, end to end: build its
    #    mosaic, crop+cache+materialize every chip it needs. Raw tiles for
    #    every requested county are already resident on local disk from
    #    step 1 and stay resident for the whole call (pruning is deferred
    #    until every group is done - see below), so every group's mosaic is
    #    always built from its complete, final raw-tile set. To bound local
    #    disk from the canonical-output side instead, each group's local
    #    canonical_tiles/ copies are deleted immediately once that group's
    #    chips are durably uploaded to S3 - canonical_tiles/ is repopulated
    #    for every requested county via a single S3 fetch pass at the end
    #    (step 8), so this is safe and invisible to callers.
    generated = skipped = downloaded = retried_blank = 0
    workers = max(1, min(max_workers, os.cpu_count() or 1))
    processed_groups: List[Tuple[str, List[Path]]] = []

    for epsg_code in sorted(relevant_epsgs):
        group_tiles = all_groups.get(epsg_code, [])
        if not group_tiles:
            continue

        log(f"  {epsg_code}: building native mosaic from {len(group_tiles)} tiles...")
        native_vrt = build_native_vrt(epsg_code, group_tiles)
        warped_vrt = build_warped_vrt(native_vrt, epsg_code)

        cells = all_needed.get(epsg_code, set())
        jobs = []
        for (r, c) in cells:
            is_forced = force or (r, c) in force_cells.get(epsg_code, set())
            jobs.append({
                "epsg": epsg_code,
                "row": r,
                "col": c,
                "warped_vrt_path": warped_vrt,
                "force": is_forced,
                "counties": cell_to_counties.get((epsg_code, r, c), []),
            })

        log(f"  {epsg_code}: chips needed: {len(jobs)} (force={force}, border-forced={sum(1 for j in jobs if j['force'])})")

        with tqdm(total=len(jobs), unit="chip", desc=f"canonical chips {epsg_code}") as pbar:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_crop_and_cache_chip, job) for job in jobs]
                for fut in as_completed(futures):
                    result = fut.result()
                    generated += result["generated"]
                    skipped += result["skipped"]
                    downloaded += result["downloaded"]
                    retried_blank += result["retried_blank"]
                    pbar.update(1)
                    pbar.set_postfix(gen=generated, skip=skipped, dl=downloaded, retried=retried_blank)

        # this group's chips are all generated/materialized and already
        # durable in S3 (uploaded inside _crop_and_cache_chip). Prune the
        # local canonical_tiles/ copies for every county this group
        # affected right away, to bound peak local disk to roughly one
        # group's canonical footprint at a time - they'll be repopulated
        # from S3 for every requested county in one pass once the whole
        # loop finishes (step 8). Raw-tile pruning for this group is
        # deferred (see after the loop): every requested county's raw
        # tiles must stay resident until ALL groups are processed, so no
        # group is ever mosaicked from a partial raw-tile set.
        affected_counties = {c for job in jobs for c in job["counties"]}
        epsg_num = epsg_code.split(":")[-1]
        for county_safe in affected_counties:
            county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
            for stale in county_dir.glob(f"chip_{epsg_num}_*.tif"):
                stale.unlink()

        processed_groups.append((epsg_code, group_tiles))

    # 6b. every relevant native-CRS group has now been fully processed with
    #     its complete raw-tile set - archive+prune ALL of this run's raw
    #     tiles in one deferred pass.
    for epsg_code, group_tiles in processed_groups:
        archive_and_prune_raw_tiles(group_tiles)

    # 7. write/refresh each requested county's S3 manifest (list of exact
    #    chip S3 keys it needs) so a collaborator - or the fetch step right
    #    below - can pull just that county's chips directly, without
    #    needing to know the native-CRS folder layout or replicate any
    #    grid/CRS math locally.
    for county, county_safe in zip(counties, counties_safe):
        cells = county_needed_cells.get(county_safe, [])
        if cells:
            write_county_manifest(county, county_safe, cells)

    # 8. every chip generated this run was pruned from local disk right
    #    after its native-CRS group finished (step 6), so canonical_tiles/
    #    is currently empty for every requested county. Repopulate it now
    #    by fetching each county's chips back from S3 via its manifest.
    result_paths: Dict[str, List[Path]] = {}
    for county, county_safe in zip(counties, counties_safe):
        cells = county_needed_cells.get(county_safe, [])
        result_paths[county_safe] = fetch_county_canonical_chips_from_s3(county, county_safe) if cells else []

    elapsed = fmt_time(time.time() - start)
    log(
        f"Done: {generated} generated, {skipped} skipped, {downloaded} "
        f"downloaded from S3, {retried_blank} stale-blank cache hit(s) re-cropped, "
        f"elapsed={elapsed}"
    )

    return result_paths


# ---------------------------------------------------------------------
# Blank-chip rescan/repair utility
# ---------------------------------------------------------------------

def rescan_and_fix_blank_chips(counties: List[str], max_workers: int = 16) -> Dict[str, Dict[str, List[str]]]:
    """
    Scan `counties`' currently-materialized canonical_tiles/*.tif for chips
    that are fully blank/NoData at full resolution, and - if any are found
    for a county - force a full regeneration of that county's chips
    (`ensure_canonical_mosaic_for_counties(counties=[county], force=True)`),
    bypassing the stale S3 cache so each chip is re-cropped from the
    CURRENT native raw-tile mosaic. Re-checks afterward and reports which
    previously-blank chips came back with real content vs. are still blank
    (the latter likely reflects a genuine source-imagery gap rather than a
    pipeline caching issue).

    This is the recommended remediation for chips that were cached blank
    due to a since-resolved gap in raw tile coverage (e.g. a transient
    download failure - see the loud warning now emitted by
    `download_6in_tiles()`/`ensure_canonical_mosaic_for_counties()`).

    Returns {county_safe: {"fixed": [chip names], "still_blank": [chip names]}}.
    """
    summary: Dict[str, Dict[str, List[str]]] = {}
    workers = max(1, min(max_workers, os.cpu_count() or 1))

    for county in counties:
        county_safe = safe_name(county.strip())
        county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
        tile_paths = sorted(county_dir.glob("*.tif"))
        if not tile_paths:
            log(f"{county}: no canonical tiles found at {county_dir} - skipping")
            continue

        log(f"{county}: scanning {len(tile_paths)} chip(s) at full resolution for blank/NoData content "
            f"({workers} parallel workers - each chip is a full read, so this can take a while for a "
            f"large county)...")
        blank_before = []
        with tqdm(total=len(tile_paths), unit="chip", desc=f"{county}: scanning for blanks") as pbar:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_path = {executor.submit(is_fully_blank, p): p for p in tile_paths}
                for fut in as_completed(future_to_path):
                    p = future_to_path[fut]
                    if fut.result():
                        blank_before.append(p.name)
                    pbar.update(1)
                    pbar.set_postfix(blank=len(blank_before))

        if not blank_before:
            log(f"{county}: {len(tile_paths)} chips checked, none blank - nothing to fix")
            summary[county_safe] = {"fixed": [], "still_blank": []}
            continue

        log(f"{county}: {len(blank_before)}/{len(tile_paths)} chip(s) currently blank - "
            f"force-regenerating this county's chips from the current native mosaic...")
        for name in blank_before:
            log(f"    - {name}")

        ensure_canonical_mosaic_for_counties(counties=[county], force=True, max_workers=max_workers)

        log(f"{county}: re-checking the {len(blank_before)} previously-blank chip(s)...")
        fixed, still_blank = [], []
        for name in blank_before:
            p = county_dir / name
            if p.exists() and not is_fully_blank(p):
                fixed.append(name)
            else:
                still_blank.append(name)

        log(f"{county}: {len(fixed)} chip(s) fixed, {len(still_blank)} still blank "
            f"(likely a genuine source-imagery gap rather than a caching issue)")
        summary[county_safe] = {"fixed": fixed, "still_blank": still_blank}

    return summary

CHIP_FILENAME_RE = re.compile(r"^chip_(\d+)_r(-?\d+)_c(-?\d+)\.tif$")

def find_phantom_duplicate_chips(counties: List[str], max_workers: int = 16) -> Dict[str, List[Dict]]:
    """
    Scan `counties`' materialized canonical_tiles/*.tif for the multi-CRS-
    group phantom-duplicate bug (see ensure_canonical_mosaic_for_counties():
    a native-CRS group's needed cells used to be computed from its overall
    bounding rectangle rather than its tiles' actual footprint, so a sparse
    secondary CRS group could wrongly claim - and generate a genuinely-blank
    chip for - a canonical (row, col) cell that a DIFFERENT CRS group
    actually covers).

    For each canonical (row, col) cell that has files from more than one
    epsg prefix, classifies the group:
      - exactly one non-blank + the rest blank -> the blank one(s) are
        phantom duplicates, flagged for cleanup.
      - all blank, or more than one non-blank -> NOT flagged (a genuine
        source gap, or a legitimate real overlap at a true CRS-zone
        boundary, respectively) - just reported for visibility.

    This only reads already-materialized canonical_tiles/*.tif files (no raw
    tiles needed), so it works even for counties whose raw tiles have
    already been archived+pruned.

    The (potentially large) set of full-resolution blank-checks - one per
    file that's part of a same-cell duplicate group, across every county -
    is done in parallel (ProcessPoolExecutor, same pattern as
    rescan_and_fix_blank_chips) with a single shared tqdm progress bar,
    rather than one blank-check at a time per county.

    Returns {county_safe: [{"row", "col", "epsg_code", "filename",
    "local_path"} for every phantom chip found]}.
    """
    workers = max(1, min(max_workers, os.cpu_count() or 1))

    # pass 1 (cheap - just filename parsing, no raster I/O): find every
    # county's duplicate (row, col) cells and collect the full set of
    # candidate files across ALL counties that actually need a blank check.
    county_dupe_cells: Dict[str, Dict[Tuple[int, int], List[Path]]] = {}
    candidate_paths: List[Path] = []
    for county in counties:
        county_safe = safe_name(county.strip())
        county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
        tile_paths = sorted(county_dir.glob("*.tif"))
        if not tile_paths:
            continue

        by_cell: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
        for p in tile_paths:
            m = CHIP_FILENAME_RE.match(p.name)
            if not m:
                continue
            row, col = int(m.group(2)), int(m.group(3))
            by_cell[(row, col)].append(p)

        dupe_cells = {cell: paths for cell, paths in by_cell.items() if len(paths) > 1}
        if not dupe_cells:
            continue

        county_dupe_cells[county_safe] = dupe_cells
        for paths in dupe_cells.values():
            candidate_paths.extend(paths)

    if not candidate_paths:
        return {}

    # pass 2 (expensive - full-resolution raster reads): blank-check every
    # candidate file once, in parallel, with one shared progress bar.
    log(f"Blank-checking {len(candidate_paths)} chip(s) across {len(county_dupe_cells)} "
        f"county folder(s) with duplicate cells ({workers} parallel workers)...")
    blank_by_path: Dict[Path, bool] = {}
    with tqdm(total=len(candidate_paths), unit="chip", desc="blank-checking duplicate chips") as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {executor.submit(is_fully_blank, p): p for p in candidate_paths}
            for fut in as_completed(future_to_path):
                p = future_to_path[fut]
                blank_by_path[p] = fut.result()
                pbar.update(1)

    # pass 3 (cheap): classify each county's dupe cells using the blank-check results.
    report: Dict[str, List[Dict]] = {}
    for county_safe, dupe_cells in county_dupe_cells.items():
        phantoms: List[Dict] = []
        all_blank_cells = 0
        multi_real_cells = 0
        for (row, col), paths in dupe_cells.items():
            non_blank = [p for p in paths if not blank_by_path[p]]
            if len(non_blank) == 1:
                for p in paths:
                    if blank_by_path[p]:
                        m = CHIP_FILENAME_RE.match(p.name)
                        epsg_num = m.group(1)
                        phantoms.append({
                            "row": row,
                            "col": col,
                            "epsg_code": f"EPSG:{epsg_num}",
                            "filename": p.name,
                            "local_path": p,
                        })
            elif len(non_blank) == 0:
                all_blank_cells += 1
            else:
                multi_real_cells += 1

        if phantoms or all_blank_cells or multi_real_cells:
            log(f"{county_safe}: {len(dupe_cells)} canonical cell(s) with >1 epsg-prefixed chip - "
                f"{len(phantoms)} confirmed phantom (blank duplicate), "
                f"{all_blank_cells} all-blank (genuine gap, not a duplicate issue), "
                f"{multi_real_cells} multiple-real-overlap (legitimate CRS-zone boundary, harmless)")
        if phantoms:
            report[county_safe] = phantoms

    return report

def delete_phantom_chips(phantom_report: Dict[str, List[Dict]], dry_run: bool = True) -> Dict[str, int]:
    """
    Delete phantom blank duplicate chips identified by
    find_phantom_duplicate_chips() - removes the local canonical_tiles/*.tif
    file and its S3 chip object, then rewrites that county's manifest
    excluding the deleted cells.

    Defaults to dry_run=True (report only, deletes nothing) - pass
    dry_run=False explicitly to actually delete, since this removes S3 data
    across every county in `phantom_report`.

    Returns {county_safe: number of phantom chips deleted (0 if dry_run)}.
    """
    deleted_counts: Dict[str, int] = {}
    for county_safe, phantoms in phantom_report.items():
        if not phantoms:
            continue

        if dry_run:
            log(f"[DRY RUN] {county_safe}: would delete {len(phantoms)} phantom chip(s): "
                f"{[p['filename'] for p in phantoms]}")
            deleted_counts[county_safe] = 0
            continue

        county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
        deleted = 0
        for p in phantoms:
            local_path: Path = p["local_path"]
            epsg_code, row, col = p["epsg_code"], p["row"], p["col"]
            key = chip_s3_key(epsg_code, row, col)
            try:
                if local_path.exists():
                    local_path.unlink()
                s3.delete_object(Bucket=S3_BUCKET, Key=key)
                deleted += 1
            except Exception as e:
                log(f"  ✗ {county_safe}: failed to delete phantom chip {p['filename']}: {e}")
        log(f"{county_safe}: deleted {deleted}/{len(phantoms)} phantom chip(s)")
        deleted_counts[county_safe] = deleted

        # rewrite this county's manifest excluding the deleted cells
        deleted_cells = {(p["epsg_code"], p["row"], p["col"]) for p in phantoms}
        remaining_tiles = sorted(county_dir.glob("*.tif")) if county_dir.is_dir() else []
        remaining_cells: List[Tuple[str, int, int]] = []
        for tp in remaining_tiles:
            m = CHIP_FILENAME_RE.match(tp.name)
            if not m:
                continue
            epsg_num, row, col = m.group(1), int(m.group(2)), int(m.group(3))
            cell = (f"EPSG:{epsg_num}", row, col)
            if cell not in deleted_cells:
                remaining_cells.append(cell)
        write_county_manifest(county_safe, county_safe, remaining_cells)

    return deleted_counts

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
  6. Once a native-CRS group's chips are fully generated and materialized,
     that group's raw tiles are no longer needed locally for this run: they
     are backed up to our own S3 bucket (`RAW_ARCHIVE_S3_PREFIX`), verified,
     then deleted locally (`archive_and_prune_raw_tiles()`) before moving on
     to the next group. This means raw tiles and canonical output never
     both have to be fully resident on local disk at the same time - only
     one CRS group's raw tiles plus the canonical output generated so far.

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
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import boto3
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

def download_from_s3(local_path: Path, bucket: str, key: str):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))

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
    for tile_path in tile_paths:
        county_safe = tile_path.parent.parent.name
        key = raw_tile_s3_key(county_safe, tile_path.name)
        local_size = tile_path.stat().st_size

        if not s3_exists(S3_BUCKET, key):
            upload_to_s3(tile_path, S3_BUCKET, key)

        try:
            head = s3.head_object(Bucket=S3_BUCKET, Key=key)
            if head["ContentLength"] != local_size:
                log(f"  \u26a0 WARNING: archive size mismatch for {tile_path.name} - not deleting local copy")
                continue
        except Exception as e:
            log(f"  \u26a0 WARNING: could not verify archive for {tile_path.name} ({e}) - not deleting local copy")
            continue

        verified.append(tile_path)

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
    subprocess.check_call([GDALBUILDVRT, "-q", "-input_file_list", str(list_file), str(out_vrt)])
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
        str(native_vrt), str(out_vrt),
    ]
    subprocess.check_call(cmd)
    return out_vrt

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

    stats = {"skipped": 0, "generated": 0, "downloaded": 0}
    s3_key = chip_s3_key(epsg_code, row, col)
    local_path = chip_local_cache_path(epsg_code, row, col)

    if not force and s3_exists(S3_BUCKET, s3_key):
        if not local_path.exists():
            download_from_s3(local_path, S3_BUCKET, s3_key)
            stats["downloaded"] = 1
        stats["skipped"] = 1
    else:
        minx, miny, maxx, maxy = chip_bounds(row, col)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            GDAL_TRANSLATE, "-q",
            "-projwin", str(minx), str(maxy), str(maxx), str(miny),
            "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
            str(warped_vrt_path), str(local_path),
        ]
        subprocess.check_call(cmd)
        stats["generated"] = 1
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
            download_6in_tiles(county, max_workers=max_workers, imagery_year=pinned_year)
        except Exception as e:
            log(f"  ✗ {county}: failed to download raw tiles: {e}")
            raise

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
            native_bounds = _raster_bounds_native(tiles)
            canon_bounds = _bounds_to_canonical(native_bounds, epsg_code, buffer_chips=0.0)
            for (r, c) in grid_cells_for_bounds(canon_bounds):
                needed.add((epsg_code, r, c))
        county_needed_cells[county_safe] = sorted(needed)

    for county_safe in new_counties_safe:
        subgroups = county_tiles_by_crs(county_safe)
        for epsg_code, tiles in subgroups.items():
            native_bounds = _raster_bounds_native(tiles)
            canon_bounds = _bounds_to_canonical(native_bounds, epsg_code, buffer_chips=BORDER_BUFFER_CHIPS)
            for cell in grid_cells_for_bounds(canon_bounds):
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
    #    mosaic, crop+cache+materialize every chip it needs, then archive
    #    that group's raw tiles to S3 and delete them locally before moving
    #    to the next group. This keeps at most one group's raw tiles on
    #    local disk at once instead of the full multi-group total.
    generated = skipped = downloaded = 0
    workers = max(1, min(max_workers, os.cpu_count() or 1))

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
                    pbar.update(1)
                    pbar.set_postfix(gen=generated, skip=skipped, dl=downloaded)

        # this group's chips are all generated/materialized - its raw tiles
        # are no longer needed locally for this run. Back them up to S3 and
        # free the space before starting the next group.
        archive_and_prune_raw_tiles(group_tiles)

    # 7. result_paths - workers already materialized hardlinks per-chip as
    #    each group's jobs completed above, so just glob what's there.
    result_paths: Dict[str, List[Path]] = {}
    for county_safe in counties_safe:
        county_dir = project_root() / "data" / "counties" / county_safe / "canonical_tiles"
        result_paths[county_safe] = sorted(county_dir.glob("*.tif"))

    elapsed = fmt_time(time.time() - start)
    log(
        f"Done: {generated} generated, {skipped} skipped, {downloaded} "
        f"downloaded from S3, elapsed={elapsed}"
    )

    return result_paths

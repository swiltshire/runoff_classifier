# utils/prepare_reprojected_tiles.py
"""
Prepares canonical (seamlessly reprojected) imagery chips for training and
inference, per state, using a "mosaic-then-reproject" architecture.

Every public function takes a `state` (e.g. "IN"); all paths, S3 prefixes, the
canonical CRS and the raw imagery source come from `utils.states`. Nothing is
shared across states: county folders, raw tiles, chips, manifests and caches
are all state-scoped (many county names exist in more than one state).

Why mosaic-then-reproject: reprojecting each raw tile independently leaves the
resampler with no neighboring pixels at tile edges and skews tile footprints,
producing triangular NoData slivers at tile boundaries.

Pipeline (`ensure_canonical_mosaic_for_counties(state, counties)`):
  1. Raw county tiles (native CRS - possibly several EPSG codes, occasionally
     >1 within a county) are restored from the state's S3 raw-tile archive
     and/or downloaded via the state's imagery source into
     `data/{STATE}/counties/{county}/tiles/`, kept pristine.
  2. ALL of the state's locally-available raw tiles are grouped by native
     EPSG code, so every warp has the maximum possible spatial context.
  3. Native-CRS groups are processed one at a time: mosaic VRT
     (`gdalbuildvrt`), warped ONCE to the state's canonical CRS/resolution as
     a lazy VRT (`gdalwarp -of VRT`) - no tile-edge starvation.
  4. A fixed-size, tap-aligned chip grid anchored at the origin of the
     canonical CRS is cropped out of the warped VRT (`gdal_translate
     -projwin`: pure extraction). Chips are cached in S3 keyed by
     (epsg, row, col), so the same physical chip is reused across counties.
  5. Each chip is hardlinked (or copied) into every requesting county's
     `canonical_tiles/` immediately and the staging copy deleted, bounding
     local disk to roughly the chips in flight.
  6. Raw tiles for every requested county stay resident for the whole call
     (so no group is mosaicked from a partial tile set). Each group's local
     canonical copies are pruned once durably in S3; after all groups are
     done, raw tiles are archived to S3, verified, and deleted locally.
  7. A per-county manifest listing that county's chip keys is written to S3,
     then every requested county's `canonical_tiles/` is repopulated from S3.

Border invalidation: when a county with no local raw tiles is added to a CRS
group that already has cached chips, chips within `BORDER_BUFFER_CHIPS`
chip-widths of its tiles are force-regenerated from the now-larger mosaic.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import boto3
import numpy as np
import rasterio
from botocore.exceptions import ClientError
from rasterio.warp import transform_bounds
from tqdm import tqdm

from utils import states
from utils.common import load_imagery_years_csv, normalize_county_key, safe_name
from utils.imagery_sources import download_raw_tiles

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CANONICAL_RES = states.CANONICAL_RES  # feet/pixel (6-inch imagery)
CHIP_SIZE_PX = 8192  # ~4096 x 4096 ft per chip at 0.5 ft resolution
BORDER_BUFFER_CHIPS = 1.0  # force-regen radius (in chips) around newly-added counties' tiles

S3_BUCKET = states.S3_BUCKET

GDALBUILDVRT = "gdalbuildvrt"
GDALWARP = "gdalwarp"
GDAL_TRANSLATE = "gdal_translate"

# prevent gdal internal oversubscription (we parallelize at the process level instead)
os.environ.setdefault("GDAL_NUM_THREADS", "1")

_S3_CLIENT = {"pid": None, "client": None}


def _s3():
    """boto3 client owned by the current process (clients are not fork-safe)."""
    if _S3_CLIENT["pid"] != os.getpid():
        _S3_CLIENT["client"] = boto3.client("s3")
        _S3_CLIENT["pid"] = os.getpid()
    return _S3_CLIENT["client"]


@dataclass(frozen=True)
class ChipContext:
    """Immutable, picklable per-state settings passed to every worker."""
    state: str
    canonical_crs: str
    chip_size_px: int
    canonical_prefix: str
    counties_dir: Path
    chip_cache_dir: Path
    mosaic_workdir: Path


def chip_context(state, chip_size_px: int = CHIP_SIZE_PX) -> ChipContext:
    cfg = states.get_state(state)
    return ChipContext(
        state=cfg.code,
        canonical_crs=states.require_canonical_crs(cfg),
        chip_size_px=chip_size_px,
        canonical_prefix=states.s3_canonical_prefix(cfg),
        counties_dir=states.counties_dir(cfg),
        chip_cache_dir=states.chip_cache_dir(cfg),
        mosaic_workdir=states.native_mosaic_cache_dir(cfg),
    )

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

def chip_s3_key(ctx: ChipContext, epsg_code: str, row: int, col: int) -> str:
    epsg_num = epsg_code.split(":")[-1]
    return f"{ctx.canonical_prefix}/{epsg_num}/chips/{chip_filename(epsg_code, row, col)}"

def chip_local_cache_path(ctx: ChipContext, epsg_code: str, row: int, col: int) -> Path:
    epsg_num = epsg_code.split(":")[-1]
    return ctx.chip_cache_dir / epsg_num / chip_filename(epsg_code, row, col)

def s3_exists(bucket: str, key: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def upload_to_s3(local_path: Path, bucket: str, key: str):
    _s3().upload_file(str(local_path), bucket, key)

def download_from_s3(local_path: Path, bucket: str, key: str):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(bucket, key, str(local_path))

def write_county_manifest(ctx: ChipContext, county: str, cells: List[Tuple[str, int, int]]) -> str:
    """Upload a small JSON manifest listing every canonical chip S3 key a
    single county needs. Lets the fast path (and collaborators, via
    scripts/fetch_canonical.ps1) fetch one county's chips without knowing the
    native-CRS folder layout or any grid math. Returns the manifest key."""
    keys = sorted({chip_s3_key(ctx, epsg_code, row, col) for (epsg_code, row, col) in cells})
    manifest = {
        "state": ctx.state,
        "county": county,
        "bucket": S3_BUCKET,
        "chip_size_px": ctx.chip_size_px,
        "canonical_crs": ctx.canonical_crs,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "keys": keys,
    }
    key = states.s3_manifest_key(ctx.state, county)
    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return key

def validate_manifest(state, county: str, manifest: Dict) -> List[str]:
    """Check a manifest belongs to `state`/`county` and only references chips
    under the state's canonical prefix; returns its keys. Raises ValueError
    otherwise (e.g. a manifest from another state or a stale prefix)."""
    cfg = states.get_state(state)
    problems = []
    if manifest.get("state") != cfg.code:
        problems.append(f"state={manifest.get('state')!r} (expected {cfg.code!r})")
    if normalize_county_key(manifest.get("county", "")) != normalize_county_key(county):
        problems.append(f"county={manifest.get('county')!r} (expected {county!r})")
    if manifest.get("bucket") != S3_BUCKET:
        problems.append(f"bucket={manifest.get('bucket')!r}")
    if cfg.canonical_crs and manifest.get("canonical_crs") != cfg.canonical_crs:
        problems.append(f"canonical_crs={manifest.get('canonical_crs')!r} (expected {cfg.canonical_crs!r})")
    keys = manifest.get("keys", [])
    prefix = states.s3_canonical_prefix(cfg) + "/"
    outside = [k for k in keys if not k.startswith(prefix)]
    if outside:
        problems.append(f"{len(outside)} key(s) outside {prefix} (e.g. {outside[0]})")
    if len(set(keys)) != len(keys):
        problems.append("duplicate keys")
    if problems:
        raise ValueError(f"invalid canonical manifest for {cfg.code}/{county}: " + "; ".join(problems))
    return keys

class ManifestChipsMissingError(RuntimeError):
    """A county's S3 manifest lists chips that no longer exist in S3.
    Callers should fall back to the full ensure pipeline for the county,
    which regenerates missing chips and rewrites a consistent manifest."""

    def __init__(self, county: str, missing_keys: List[str]):
        self.county = county
        self.missing_keys = missing_keys
        super().__init__(
            f"{county}: {len(missing_keys)} manifest-listed chip(s) missing from S3 "
            f"(first: {missing_keys[0]})"
        )

def _is_s3_404(e: Exception) -> bool:
    return isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") in (
        "404", "NoSuchKey", "NotFound",
    )

def fetch_county_canonical_chips_from_s3(
    state, county: str, *, verify_sizes: bool = False
) -> List[Path]:
    """Download every canonical chip listed in a county's S3 manifest into
    its local canonical_tiles/ folder, skipping chips already present
    locally. Used internally by ensure_canonical_mosaic_for_counties() to
    repopulate canonical_tiles/ after generation, and standalone as the
    manifest fast path (scripts/prepare_canonical_mosaic.py).

    With verify_sizes=True, chips already present locally are also checked
    against the S3 object's ContentLength and re-downloaded on mismatch
    (catches truncated files from an interrupted earlier download)."""
    cfg = states.get_state(state)
    manifest_key = states.s3_manifest_key(cfg, county)
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=manifest_key)
    except Exception as e:
        log(f"  ✗ {cfg.code}/{county}: no manifest found in S3 ({manifest_key}): {e}")
        return []
    keys = validate_manifest(cfg, county, json.loads(obj["Body"].read()))

    county_dir = states.canonical_tiles_dir(cfg, county)
    county_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    downloaded = 0
    repaired = 0
    missing: List[str] = []
    with tqdm(total=len(keys), unit="chip", desc=f"  fetching {cfg.code}/{county} from S3") as pbar:
        for key in keys:
            local_path = county_dir / Path(key).name
            if not local_path.exists():
                try:
                    download_from_s3(local_path, S3_BUCKET, key)
                    downloaded += 1
                except Exception as e:
                    if not _is_s3_404(e):
                        raise
                    missing.append(key)
                    pbar.update(1)
                    continue
            elif verify_sizes:
                try:
                    remote_size = _s3().head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"]
                except Exception as e:
                    if _is_s3_404(e):
                        # object deleted from S3 after the manifest was written -
                        # the local copy may be equally stale, treat as missing
                        missing.append(key)
                        pbar.update(1)
                        continue
                    remote_size = None
                if remote_size is not None and local_path.stat().st_size != remote_size:
                    download_from_s3(local_path, S3_BUCKET, key)
                    repaired += 1
            paths.append(local_path)
            pbar.update(1)
    if missing:
        log(
            f"  \u26a0 {cfg.code}/{county}: {len(missing)} of {len(keys)} manifest-listed chip(s) "
            f"missing from S3 (e.g. {missing[0]}) - manifest is stale, county needs "
            f"the full ensure pipeline to regenerate them and rewrite the manifest"
        )
        raise ManifestChipsMissingError(county, missing)
    log(
        f"  {cfg.code}/{county}: {len(paths)} chip(s) ready ({downloaded} downloaded, "
        f"{repaired} repaired, {len(paths) - downloaded - repaired} already local)"
    )
    return sorted(paths)

def county_has_manifest(state, county: str) -> bool:
    """True if a canonical-chip manifest exists in S3 for this county.
    Manifests are only written at the END of a fully successful ensure pass,
    so presence means every chip it lists was cached to S3 first."""
    return s3_exists(S3_BUCKET, states.s3_manifest_key(state, county))

# ---------------------------------------------------------------------
# Raw tile S3 archive (backup-then-delete lifecycle, so raw tiles and
# canonical output never both have to be fully resident on local disk)
# ---------------------------------------------------------------------

def raw_tile_s3_key(state, county_safe: str, filename: str) -> str:
    return f"{states.s3_raw_archive_prefix(state)}/{county_safe}/{filename}"

def restore_raw_tiles_from_archive(state, county: str) -> int:
    """Download any raw tiles already archived in S3 for this county into its
    local tiles/ folder, so a re-added county doesn't need to re-fetch from
    the state's imagery source (whose downloader then skips valid existing
    files)."""
    county_safe = safe_name(county.strip())
    prefix = f"{states.s3_raw_archive_prefix(state)}/{county_safe}/"
    dest_dir = states.raw_tiles_dir(state, county)
    dest_dir.mkdir(parents=True, exist_ok=True)

    archived: List[Tuple[str, str, int]] = []
    paginator = _s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]
            if not filename or "/" in filename:
                continue
            archived.append((key, filename, obj["Size"]))

    if not archived:
        return 0

    restored = 0
    with tqdm(total=len(archived), unit="tile", desc=f"  restoring {county_safe} raw tiles from S3") as pbar:
        for key, filename, size in archived:
            local_path = dest_dir / filename
            if not (local_path.exists() and local_path.stat().st_size == size):
                _s3().download_file(S3_BUCKET, key, str(local_path))
                restored += 1
            pbar.update(1)
    return restored

def archive_and_prune_raw_tiles(state, tile_paths: List[Path]) -> None:
    """Back up raw tiles to the state's S3 raw archive, verify the upload,
    then delete the local copy."""
    verified: List[Path] = []
    uploaded = already_archived = 0
    with tqdm(total=len(tile_paths), unit="tile", desc="  archiving raw tiles") as pbar:
        for tile_path in tile_paths:
            county_safe = tile_path.parent.parent.name
            key = raw_tile_s3_key(state, county_safe, tile_path.name)
            local_size = tile_path.stat().st_size

            if s3_exists(S3_BUCKET, key):
                already_archived += 1
            else:
                upload_to_s3(tile_path, S3_BUCKET, key)
                uploaded += 1

            try:
                head = _s3().head_object(Bucket=S3_BUCKET, Key=key)
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
# folder, since scanning every tile's header gets slow)
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

def local_raw_tiles_by_crs(state) -> Dict[str, List[Path]]:
    """Group ALL of one state's local raw tiles (every county under
    data/{STATE}/counties/*/tiles) by native EPSG. Never crosses states."""
    root = states.counties_dir(state)
    groups: Dict[str, List[Path]] = defaultdict(list)
    if not root.is_dir():
        return groups
    for county_dir in sorted(root.iterdir()):
        tiles_dir = county_dir / "tiles"
        if not tiles_dir.is_dir():
            continue
        for name, epsg in _tile_epsg_map(tiles_dir).items():
            groups[epsg].append(tiles_dir / name)
    return groups

def county_tiles_by_crs(state, county: str) -> Dict[str, List[Path]]:
    """Group just one county's own raw tiles by native EPSG (a county can span >1 CRS)."""
    tiles_dir = states.raw_tiles_dir(state, county)
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

def _bounds_to_canonical(
    ctx: ChipContext, bounds: Tuple[float, float, float, float], src_epsg: str, buffer_chips: float = 0.0
) -> Tuple[float, float, float, float]:
    """Reproject a native-CRS bbox into the canonical CRS (densified for curvature), buffer outward by `buffer_chips` chip-widths, then tap-align."""
    minx, miny, maxx, maxy = transform_bounds(src_epsg, ctx.canonical_crs, *bounds, densify_pts=21)
    chip_units = ctx.chip_size_px * CANONICAL_RES
    buf = buffer_chips * chip_units
    return _tap_align(minx - buf, miny - buf, maxx + buf, maxy + buf, CANONICAL_RES)

def grid_cells_for_bounds(ctx: ChipContext, bounds: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
    """List (row, col) grid cells (in the GLOBAL, origin-anchored canonical grid) overlapping bounds."""
    minx, miny, maxx, maxy = bounds
    chip_units = ctx.chip_size_px * CANONICAL_RES
    col_start = math.floor(minx / chip_units)
    col_end = math.floor((maxx - 1e-6) / chip_units)
    row_start = math.floor(miny / chip_units)
    row_end = math.floor((maxy - 1e-6) / chip_units)
    return [(r, c) for r in range(row_start, row_end + 1) for c in range(col_start, col_end + 1)]

def _needed_cells_for_tiles(
    ctx: ChipContext, tiles: List[Path], epsg_code: str, buffer_chips: float = 0.0
) -> Set[Tuple[int, int]]:
    """Canonical grid cells actually covered by `tiles` - the UNION of each
    tile's own footprint, NOT the group's overall bounding rectangle.

    Using a single overall bbox is wrong whenever a native-CRS group's tiles
    are sparse/scattered relative to their own bounding rectangle: the bbox
    can span area that a DIFFERENT native-CRS group's tiles actually cover,
    causing this group to also claim (and generate a genuinely-blank chip
    for) a canonical cell that's really covered by the other group.
    Computing per-tile means each group only ever claims cells its own tiles
    actually touch."""
    cells: Set[Tuple[int, int]] = set()
    for p in tiles:
        with rasterio.open(p) as ds:
            b = ds.bounds
        canon_bounds = _bounds_to_canonical(ctx, (b.left, b.bottom, b.right, b.top), epsg_code, buffer_chips=buffer_chips)
        cells.update(grid_cells_for_bounds(ctx, canon_bounds))
    return cells

def chip_bounds(ctx: ChipContext, row: int, col: int) -> Tuple[float, float, float, float]:
    chip_units = ctx.chip_size_px * CANONICAL_RES
    minx = col * chip_units
    miny = row * chip_units
    return minx, miny, minx + chip_units, miny + chip_units

# ---------------------------------------------------------------------
# GDAL subprocess helpers (native mosaic + warped VRT per CRS group)
# ---------------------------------------------------------------------

def build_native_vrt(ctx: ChipContext, epsg_code: str, tile_paths: List[Path]) -> Path:
    workdir = ctx.mosaic_workdir
    workdir.mkdir(parents=True, exist_ok=True)
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
    # region. Declaring 0 as nodata lets gdalbuildvrt see through those
    # pixels to whichever source actually has real data underneath. 0 is
    # also the sentinel is_fully_blank() treats as "no imagery", and true
    # all-band (R,G,B,IR all exactly 0) pixels don't occur in real imagery.
    subprocess.check_call([
        GDALBUILDVRT, "-q",
        "-srcnodata", "0", "-vrtnodata", "0",
        "-input_file_list", str(list_file), str(out_vrt),
    ])
    return out_vrt

def build_warped_vrt(ctx: ChipContext, native_vrt: Path, epsg_code: str) -> Path:
    epsg_num = epsg_code.split(":")[-1]
    out_vrt = ctx.mosaic_workdir / f"warped_{epsg_num}.vrt"
    if out_vrt.exists():
        out_vrt.unlink()
    cmd = [
        GDALWARP, "-q", "-of", "VRT",
        "-t_srs", ctx.canonical_crs,
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
# Blank-chip detection
#
# The chip cache is keyed only by (epsg, row, col) and reused indefinitely.
# If a raw tile fails to download or a CRS group's native mosaic otherwise
# has a transient gap, the chip cropped from that gap is all-NoData. Cache
# hits are therefore re-checked for blankness and re-cropped from the
# current mosaic rather than trusted blindly.
# ---------------------------------------------------------------------

def is_fully_blank(path: Path) -> bool:
    """True if every pixel in `path` is NoData (or exactly zero, when no
    NoData value is set) across all bands."""
    with rasterio.open(path) as src:
        arr = src.read()
        nodata_val = src.nodata
        if nodata_val is not None:
            return bool(np.all(arr == nodata_val))
        return bool(np.all(arr == 0))

# ---------------------------------------------------------------------
# Per-chip crop worker (top-level function so it is picklable for
# ProcessPoolExecutor)
# ---------------------------------------------------------------------

def _crop_and_cache_chip(job: Dict) -> Dict[str, int]:
    ctx: ChipContext = job["ctx"]
    epsg_code = job["epsg"]
    row = job["row"]
    col = job["col"]
    warped_vrt_path = job["warped_vrt_path"]
    force = job["force"]
    counties_needing_this_cell: List[str] = job["counties"]

    stats = {"skipped": 0, "generated": 0, "downloaded": 0, "retried_blank": 0}
    s3_key = chip_s3_key(ctx, epsg_code, row, col)
    local_path = chip_local_cache_path(ctx, epsg_code, row, col)

    use_cache = not force and s3_exists(S3_BUCKET, s3_key)
    if use_cache:
        if not local_path.exists():
            download_from_s3(local_path, S3_BUCKET, s3_key)
            stats["downloaded"] = 1

        # Never trust a cached hit blindly - the chip cache is keyed only by
        # (epsg, row, col), NEVER by county, and is NOT invalidated by
        # re-downloading raw tiles. If this exact cell was ever cached blank
        # (stale mosaic gap, truncated write, etc.), it would otherwise be
        # re-served forever. Re-cropping from the CURRENT warped mosaic is
        # cheap and harmless even for a genuine coverage gap.
        if is_fully_blank(local_path):
            use_cache = False
            stats["retried_blank"] = 1

    if use_cache:
        stats["skipped"] = 1
    else:
        minx, miny, maxx, maxy = chip_bounds(ctx, row, col)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp path in the same directory and atomically rename
        # into place once gdal_translate has fully succeeded and the result
        # has been read back cleanly, so a killed/interrupted worker or a
        # disk-full condition can never leave a truncated GeoTIFF at
        # `local_path`.
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
            # Not -q / check_call: on failure we need gdal_translate's stderr
            # to diagnose *why*.
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"gdal_translate failed for chip {s3_key} (exit {proc.returncode}):\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stderr: {proc.stderr.strip()}\n"
                    f"  stdout: {proc.stdout.strip()}"
                )

            # Verify the freshly-written file reads back cleanly before
            # trusting it (e.g. a delayed disk-full error only surfaced at close()).
            with rasterio.open(tmp_path) as _verify_src:
                _verify_src.read()

            os.replace(tmp_path, local_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        stats["generated"] = 1

        # A freshly-cropped chip that comes out fully blank is suspicious (a
        # gap in the native mosaic - e.g. from a failed raw tile download)
        # rather than expected. Warn without blocking this run; blank cache
        # hits are re-cropped automatically on the next run.
        if is_fully_blank(local_path):
            reason = "still blank after re-cropping a stale cached blank" if stats["retried_blank"] else "generated fully blank/NoData"
            print(f"  \u26a0 WARNING: chip {s3_key} {reason} - caching anyway")

        upload_to_s3(local_path, S3_BUCKET, s3_key)

    # Materialize into every requesting county's canonical_tiles/ folder as a
    # hardlink (falling back to a copy across filesystems) immediately, then
    # drop the shared staging copy - S3 is the durable store.
    for county_safe in counties_needing_this_cell:
        county_dir = ctx.counties_dir / county_safe / "canonical_tiles"
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

@contextmanager
def _state_chip_lock(ctx: ChipContext):
    """Exclusive per-state lock so two chip-prep runs for the same state can't
    race over the shared per-CRS VRTs / chip cache (Linux; no-op elsewhere)."""
    ctx.mosaic_workdir.mkdir(parents=True, exist_ok=True)
    lock_path = ctx.mosaic_workdir / ".chip_prep.lock"
    with open(lock_path, "w") as fh:
        try:
            import fcntl
        except ImportError:
            yield
            return
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"another canonical chip run for {ctx.state} is in progress (lock: {lock_path})"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def ensure_canonical_mosaic_for_counties(
    state,
    counties: List[str],
    *,
    force: bool = False,
    max_workers: int = 16,
    chip_size_px: int = CHIP_SIZE_PX,
    border_buffer_chips: float = BORDER_BUFFER_CHIPS,
    max_failed_raw_downloads: int = 20,
) -> Dict[str, List[Path]]:
    """
    Ensure canonical (seamlessly reprojected) imagery chips exist for
    `counties` of `state`. Every relevant native-CRS group is mosaicked from
    ALL of the state's locally available raw tiles (not just `counties`)
    before warping, so tile-edge context is never starved.

    Each county's raw-tile download is pinned to the imagery year in
    `data/{STATE}/training_county_imagery_years.csv` when the county has an
    entry (matching the year its training labels were verified against);
    otherwise the imagery source auto-detects, with a warning.

    Returns {county_safe_name: [local canonical_tiles chip paths]}.
    """
    ctx = chip_context(state, chip_size_px)
    with _state_chip_lock(ctx):
        return _ensure_canonical_mosaic(
            ctx, counties,
            force=force,
            max_workers=max_workers,
            border_buffer_chips=border_buffer_chips,
            max_failed_raw_downloads=max_failed_raw_downloads,
        )


def _ensure_canonical_mosaic(
    ctx: ChipContext,
    counties: List[str],
    *,
    force: bool,
    max_workers: int,
    border_buffer_chips: float,
    max_failed_raw_downloads: int,
) -> Dict[str, List[Path]]:
    start = time.time()
    st = ctx.state
    unknown = [c for c in counties if normalize_county_key(c) not in
               {normalize_county_key(x) for x in states.get_state(st).counties}]
    if unknown:
        raise ValueError(f"{st}: not counties of this state: {unknown}")
    counties_safe = [safe_name(c.strip()) for c in counties]

    imagery_years: Dict[str, int] = {}
    years_csv = states.imagery_years_csv(st)
    if years_csv.exists():
        imagery_years = load_imagery_years_csv(years_csv)

    # 1. ensure raw tiles exist locally for every requested county, tracking
    #    which ones are genuinely new (had no local raw tiles before this
    #    call) so we know where border-invalidation applies.
    new_counties: List[str] = []
    for county in counties:
        tiles_dir = states.raw_tiles_dir(st, county)
        existed_before = tiles_dir.is_dir() and any(tiles_dir.glob("*.tif"))
        if not existed_before:
            new_counties.append(county)
            # restore previously-archived raw tiles from our own S3 bucket
            # first (fast/reliable) before falling back to the state's
            # imagery source for anything still missing.
            restored = restore_raw_tiles_from_archive(st, county)
            if restored:
                log(f"  {st}/{county}: restored {restored} raw tiles from S3 archive")

        pinned_year = imagery_years.get(normalize_county_key(county))
        if pinned_year is None:
            log(f"  ⚠ WARNING: no pinned imagery year for {st}/{county} - falling back to auto-detect")

        try:
            download_result = download_raw_tiles(
                st, county, tiles_dir, max_workers=max_workers, imagery_year=pinned_year
            )
        except Exception as e:
            log(f"  ✗ {st}/{county}: failed to download raw tiles: {e}")
            raise

        if download_result.get("failed", 0) > max_failed_raw_downloads:
            # A handful of failures can be a genuine upstream gap, but MASS
            # failure means something is systemically wrong (disk full,
            # upstream outage, throttling). Building a mosaic from a badly
            # incomplete tile set would compute an undersized needed-cells
            # list and cache potentially hundreds of blank chips to S3.
            # Abort resumably instead - re-running repairs the download.
            raise RuntimeError(
                f"{st}/{county}: {download_result['failed']} raw tile download(s) failed "
                f"(> max_failed_raw_downloads={max_failed_raw_downloads}) - aborting before "
                f"building a gappy mosaic that would poison the S3 chip cache with blanks. "
                f"Likely cause: disk full or upstream outage. Fix and re-run to resume."
            )
        if download_result.get("failed", 0) > 0:
            log(
                f"  ⚠ WARNING: {st}/{county}: {download_result['failed']} raw tile download(s) failed "
                f"({download_result['year']} imagery) - the native mosaic for this area will have a "
                f"gap where those tiles would have been, and any canonical chip cropped from that gap "
                f"will be blank/NoData. Re-run this county through "
                f"ensure_canonical_mosaic_for_counties() to retry the download and re-crop blank chips."
            )

    # 2. group ALL of the state's locally-available raw tiles by native CRS,
    #    so every warp has maximum context.
    all_groups = local_raw_tiles_by_crs(st)

    # 3. compute each requested county's own required chip coverage, and
    #    (for genuinely new counties) the border-buffered cells whose
    #    existing cached chips must be force-regenerated.
    county_needed_cells: Dict[str, List[Tuple[str, int, int]]] = {}
    relevant_epsgs: Set[str] = set()
    force_cells: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)

    for county, county_safe in zip(counties, counties_safe):
        needed: Set[Tuple[str, int, int]] = set()
        for epsg_code, tiles in county_tiles_by_crs(st, county).items():
            relevant_epsgs.add(epsg_code)
            for (r, c) in _needed_cells_for_tiles(ctx, tiles, epsg_code, buffer_chips=0.0):
                needed.add((epsg_code, r, c))
        county_needed_cells[county_safe] = sorted(needed)

    for county in new_counties:
        for epsg_code, tiles in county_tiles_by_crs(st, county).items():
            for cell in _needed_cells_for_tiles(ctx, tiles, epsg_code, buffer_chips=border_buffer_chips):
                force_cells[epsg_code].add(cell)

    if not relevant_epsgs:
        log("No raw tiles found for the requested counties; nothing to do.")
        return {c: [] for c in counties_safe}

    log(f"{st}: requested counties span {len(relevant_epsgs)} native CRS group(s): {sorted(relevant_epsgs)}")

    # 4. clear stale canonical_tiles/*.tif upfront (once, before any group's
    #    jobs run), so this cleanup can't race in-flight workers.
    for county in counties:
        county_dir = states.canonical_tiles_dir(st, county)
        county_dir.mkdir(parents=True, exist_ok=True)
        for existing in county_dir.glob("*.tif"):
            existing.unlink()

    # 5. assemble the full chip job list (union of all requested counties'
    #    needed cells), applying force where border-invalidation says so,
    #    and map each cell back to the counties that need it.
    all_needed: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    for cells in county_needed_cells.values():
        for (epsg_code, r, c) in cells:
            all_needed[epsg_code].add((r, c))

    cell_to_counties: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)
    for county_safe, cells in county_needed_cells.items():
        for cell in cells:
            cell_to_counties[cell].append(county_safe)

    # 6. process ONE native-CRS group at a time, end to end. Raw tiles stay
    #    resident for the whole call; each group's local canonical copies
    #    are deleted once durably in S3 and repopulated at the end (step 8).
    generated = skipped = downloaded = retried_blank = 0
    workers = max(1, min(max_workers, os.cpu_count() or 1))
    processed_groups: List[Tuple[str, List[Path]]] = []

    for epsg_code in sorted(relevant_epsgs):
        group_tiles = all_groups.get(epsg_code, [])
        if not group_tiles:
            continue

        log(f"  {epsg_code}: building native mosaic from {len(group_tiles)} tiles...")
        native_vrt = build_native_vrt(ctx, epsg_code, group_tiles)
        warped_vrt = build_warped_vrt(ctx, native_vrt, epsg_code)

        cells = all_needed.get(epsg_code, set())
        jobs = []
        for (r, c) in cells:
            is_forced = force or (r, c) in force_cells.get(epsg_code, set())
            jobs.append({
                "ctx": ctx,
                "epsg": epsg_code,
                "row": r,
                "col": c,
                "warped_vrt_path": warped_vrt,
                "force": is_forced,
                "counties": cell_to_counties.get((epsg_code, r, c), []),
            })

        log(f"  {epsg_code}: chips needed: {len(jobs)} (force={force}, border-forced={sum(1 for j in jobs if j['force'])})")

        with tqdm(total=len(jobs), unit="chip", desc=f"canonical chips {st} {epsg_code}") as pbar:
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

        affected_counties = {c for job in jobs for c in job["counties"]}
        epsg_num = epsg_code.split(":")[-1]
        for county_safe in affected_counties:
            county_dir = ctx.counties_dir / county_safe / "canonical_tiles"
            for stale in county_dir.glob(f"chip_{epsg_num}_*.tif"):
                stale.unlink()

        processed_groups.append((epsg_code, group_tiles))

    # 6b. every relevant group has been processed with its complete raw-tile
    #     set - archive+prune ALL of this run's raw tiles in one pass.
    for epsg_code, group_tiles in processed_groups:
        archive_and_prune_raw_tiles(st, group_tiles)

    # 7. write/refresh each requested county's S3 manifest.
    log(f"Writing S3 manifests for {len(counties)} counties...")
    for county, county_safe in zip(counties, counties_safe):
        cells = county_needed_cells.get(county_safe, [])
        if cells:
            write_county_manifest(ctx, county, cells)

    # 8. repopulate canonical_tiles/ for every requested county from S3.
    log(f"Fetching canonical chips back from S3 for {len(counties)} counties...")
    result_paths: Dict[str, List[Path]] = {}
    for county, county_safe in zip(counties, counties_safe):
        cells = county_needed_cells.get(county_safe, [])
        result_paths[county_safe] = fetch_county_canonical_chips_from_s3(st, county) if cells else []

    elapsed = fmt_time(time.time() - start)
    log(
        f"Done: {generated} generated, {skipped} skipped, {downloaded} "
        f"downloaded from S3, {retried_blank} stale-blank cache hit(s) re-cropped, "
        f"elapsed={elapsed}"
    )

    return result_paths

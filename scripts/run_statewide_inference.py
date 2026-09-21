"""
Statewide inference driver: run chip-prep + inference + cleanup over every
requested Indiana county in small disk-bounded batches, resumably and without
manual intervention. Designed to be launched once via nohup and left alone:

    mkdir -p outputs/logs
    cd /home/sagemaker-user/runoff_classifier
    nohup python -u scripts/run_statewide_inference.py \
        > outputs/logs/statewide_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    tail -f outputs/logs/statewide_*.log
    ps aux | grep run_statewide_inference

Per batch (default 4 counties) it:
  1. skips counties that already have a final merged detections gpkg
     (resume marker - re-running this script never redoes finished work);
  2. ensures canonical chips: manifest fast path when the county completed a
     prior ensure run (chips fetched straight from S3, no raw tiles), full
     mosaic pipeline otherwise;
  3. runs 4-GPU torch.distributed inference county by county (same arguments
     as the pipeline notebook's inference cell);
  4. deletes each county's local canonical_tiles/ right after its inference
     succeeds (chips remain durably in S3 + manifest, restorable in minutes),
     so local disk stays bounded to roughly one batch instead of growing
     with every county processed.

Raw tiles are archived+pruned by ensure_canonical_mosaic_for_counties()
itself. NOTE: that pruning pass covers every locally-resident raw tile in
each processed CRS group - including counties from LATER batches whose raw
tiles happen to be on disk already. Those get restored from our own S3
archive when their batch comes up, which costs some S3 traffic but keeps
peak disk bounded and the run fully hands-off.

A disk preflight aborts (resumably) before starting a batch if free space
has dropped below --min_free_gb, rather than dying mid-gdal_translate.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.county_boundaries import get_county_boundary_path  # noqa: E402
from utils.indiana_cogs import safe_name  # noqa: E402
from utils.prepare_reprojected_tiles import (  # noqa: E402
    county_has_manifest,
    ensure_canonical_mosaic_for_counties,
    fetch_county_canonical_chips_from_s3,
)

# Keep in sync with the pipeline notebook ("All 92 Indiana counties" cell).
ALL_INDIANA_COUNTIES = sorted([
    'Adams', 'Allen', 'Bartholomew', 'Benton', 'Blackford', 'Boone', 'Brown',
    'Carroll', 'Cass', 'Clark', 'Clay', 'Clinton', 'Crawford', 'Daviess',
    'Dearborn', 'Decatur', 'DeKalb', 'Delaware', 'Dubois', 'Elkhart', 'Fayette',
    'Floyd', 'Fountain', 'Franklin', 'Fulton', 'Gibson', 'Grant', 'Greene',
    'Hamilton', 'Hancock', 'Harrison', 'Hendricks', 'Henry', 'Howard', 'Huntington',
    'Jackson', 'Jasper', 'Jay', 'Jefferson', 'Jennings', 'Johnson', 'Knox',
    'Kosciusko', 'LaGrange', 'Lake', 'LaPorte', 'Lawrence', 'Madison', 'Marion',
    'Marshall', 'Martin', 'Miami', 'Monroe', 'Montgomery', 'Morgan', 'Newton',
    'Noble', 'Ohio', 'Orange', 'Owen', 'Parke', 'Perry', 'Pike', 'Porter',
    'Posey', 'Pulaski', 'Putnam', 'Randolph', 'Ripley', 'Rush', 'Scott',
    'Shelby', 'Spencer', 'Starke', 'Steuben', 'St. Joseph', 'Sullivan', 'Switzerland',
    'Tippecanoe', 'Tipton', 'Union', 'Vanderburgh', 'Vermillion', 'Vigo', 'Wabash',
    'Warren', 'Warrick', 'Washington', 'Wayne', 'Wells', 'White', 'Whitley'
])

# Keep in sync with the notebook's inference config cell.
TRAIN_OUT_DIR = PROJECT_ROOT / "outputs" / "train_multicounty"
MODEL_PATH = TRAIN_OUT_DIR / "model_final.pth"
NHD_MASK_PATH = PROJECT_ROOT / "data" / "NHDmask_Indiana" / "NHDfinalMaskIndiana.shp"
CLASS_AREA_CSV = PROJECT_ROOT / "data" / "feature_size_threshholds.csv"
INFER_CONFIG = {
    "task": "instance_seg",
    "tile_size": 512,
    "stride": 256,
    "infer_batch": 5,
    "score_thresh": 0.75,
    "normalize": "imagenet",
    "nms_iou_thresh": 0.3,
    "final_box_iou": 0.3,
    "min_cover_frac": 0.01,
    "county_min_cover_frac": 0.5,
}


def log(msg: str) -> None:
    print(f"[statewide {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def county_is_done(county: str) -> bool:
    """A county is done when a final MERGED detections gpkg exists (per-rank
    partials `..._rankN.gpkg` and diagnostic exports don't count)."""
    pattern = str(TRAIN_OUT_DIR / f"inferences_{county}" / f"detections_{county}_*.gpkg")
    for p in glob.glob(pattern):
        if not re.search(r"_rank\d+", p) and not p.endswith("_pixel_diag.gpkg"):
            return True
    return False


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def ensure_chips_for_batch(batch: list[str], max_workers: int) -> None:
    """Manifest fast path per county where possible; full mosaic pipeline for
    the rest (same split as scripts/prepare_canonical_mosaic.py)."""
    full_counties = []
    for county in batch:
        county_safe = safe_name(county)
        if county_has_manifest(county_safe):
            log(f"{county}: manifest found - fetching chips from S3 (no raw tiles needed)")
            fetch_county_canonical_chips_from_s3(county, county_safe, verify_sizes=True)
        else:
            full_counties.append(county)
    if full_counties:
        log(f"full chip pipeline for: {full_counties}")
        ensure_canonical_mosaic_for_counties(counties=full_counties, max_workers=max_workers)


def run_inference(county: str, nproc: int) -> Path:
    """Run 4-GPU DDP inference for one county; returns the output gpkg path.
    Raises on nonzero exit or missing output."""
    raster_dir = PROJECT_ROOT / "data" / "counties" / safe_name(county) / "canonical_tiles"
    if not glob.glob(str(raster_dir / "*.tif")):
        raise RuntimeError(f"{county}: no canonical tiles at {raster_dir} - ensure step failed?")

    out_dir = TRAIN_OUT_DIR / f"inferences_{county}"
    out_dir.mkdir(parents=True, exist_ok=True)
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_gpkg = out_dir / f"detections_{county}_{datestamp}.gpkg"

    # Resolve the county's TIGER boundary in THIS single process before the
    # multi-rank launch, so ranks can't race the download/extraction.
    county_boundary_path = get_county_boundary_path(county)
    log(f"{county}: boundary {county_boundary_path}")

    cmd = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
        str(PROJECT_ROOT / "scripts" / "inference.py"),
        "--task", INFER_CONFIG["task"],
        "--device", "cuda",
        "--raster_path", str(raster_dir) + os.sep,
        "--checkpoint", str(MODEL_PATH),
        "--out_vector", str(output_gpkg),
        "--tile_size", str(INFER_CONFIG["tile_size"]),
        "--stride", str(INFER_CONFIG["stride"]),
        "--infer_batch", str(INFER_CONFIG["infer_batch"]),
        "--score_thresh", str(INFER_CONFIG["score_thresh"]),
        "--normalize", INFER_CONFIG["normalize"],
        "--nms_iou_thresh", str(INFER_CONFIG["nms_iou_thresh"]),
        "--final_box_iou", str(INFER_CONFIG["final_box_iou"]),
        "--mask_path", str(NHD_MASK_PATH),
        "--min_cover_frac", str(INFER_CONFIG["min_cover_frac"]),
        "--county_mask_path", str(county_boundary_path),
        "--county_min_cover_frac", str(INFER_CONFIG["county_min_cover_frac"]),
        "--class_area_csv", str(CLASS_AREA_CSV),
    ]
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(nproc)),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    log(f"{county}: launching inference -> {output_gpkg}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{county}: inference exited with code {proc.returncode}")
    if not output_gpkg.exists():
        raise RuntimeError(f"{county}: inference exited 0 but final output missing: {output_gpkg}")
    log(f"{county}: inference done in {(time.time() - t0) / 60:.1f} min")
    return output_gpkg


def cleanup_county_chips(county: str) -> None:
    """Reclaim a county's local disk after inference.

    - Mask-cache artifacts ({County}_mosaic.vrt + *_mosaic_mask_*.npy/.json)
      are pure derived caches written by inference.py into the county dir at
      full mosaic resolution (~40-80 GB *per mask*, and there are two per
      county: NHD + county boundary). Nothing else ever deletes them, so an
      unattended statewide run silently accumulates terabytes. Always delete.
    - canonical_tiles chips are deleted only if the county's S3 manifest
      exists (i.e. they are durably restorable).
    """
    county_safe = safe_name(county)
    county_dir = PROJECT_ROOT / "data" / "counties" / county_safe

    freed = 0
    for f in list(county_dir.glob("*_mosaic_mask_*")) + list(county_dir.glob("*_mosaic.vrt")):
        if f.is_file():
            freed += f.stat().st_size
            f.unlink()
    if freed:
        log(f"{county}: mask-cache artifacts deleted ({freed / 1e9:.0f} GB reclaimed)")

    if not county_has_manifest(county_safe):
        log(f"{county}: ⚠ no S3 manifest - keeping local canonical_tiles as the only copy")
        return
    chip_dir = county_dir / "canonical_tiles"
    if chip_dir.is_dir():
        shutil.rmtree(chip_dir)
        log(f"{county}: local canonical_tiles deleted (restorable from S3 manifest)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counties", type=str, default="",
        help="comma-separated county names; default = all 92 Indiana counties",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="comma-separated county names to exclude",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="counties per chip-prep batch; bounds peak local disk (default 4)",
    )
    parser.add_argument(
        "--min_free_gb", type=float, default=1200.0,
        help="abort (resumably) before a batch if free disk is below this. A 4-county "
             "full-pipeline batch can peak well past 1 TB (raw tiles + mosaic chips), "
             "so keep this generous (default 1200)",
    )
    parser.add_argument("--nproc_per_node", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument(
        "--keep_chips", action="store_true",
        help="don't delete counties' local canonical_tiles after inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    requested = ([c.strip() for c in args.counties.split(",") if c.strip()]
                 if args.counties else list(ALL_INDIANA_COUNTIES))
    skip = {c.strip() for c in args.skip.split(",") if c.strip()}
    requested = [c for c in requested if c not in skip]

    if not MODEL_PATH.exists():
        raise SystemExit(f"model checkpoint not found: {MODEL_PATH}")

    done = [c for c in requested if county_is_done(c)]
    todo = [c for c in requested if not county_is_done(c)]
    log(f"{len(requested)} counties requested: {len(done)} already done, {len(todo)} to run")
    if done:
        log(f"already done (skipping): {done}")
    if not todo:
        log("nothing to do")
        return

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    failures: list[str] = []
    t_start = time.time()

    for bi, batch in enumerate(batches, 1):
        log("=" * 70)
        log(f"BATCH {bi}/{len(batches)}: {batch}")
        log("=" * 70)

        headroom = free_gb(PROJECT_ROOT)
        log(f"disk preflight: {headroom:.0f} GB free")
        if headroom < args.min_free_gb:
            raise SystemExit(
                f"aborting before batch {bi}: only {headroom:.0f} GB free "
                f"(< --min_free_gb {args.min_free_gb:.0f}). Free space and re-run "
                f"this same command - completed counties are skipped automatically."
            )

        ensure_chips_for_batch(batch, max_workers=args.max_workers)

        for county in batch:
            try:
                run_inference(county, nproc=args.nproc_per_node)
            except Exception as e:
                # keep going - one county's failure shouldn't waste the rest of
                # an unattended overnight run.
                log(f"\u2717 {county} FAILED: {e}")
                failures.append(county)
                # Still reclaim the county's local chips if they're durably
                # restorable from S3 - otherwise a string of failures (e.g. a
                # persistent GPU issue) silently accumulates hundreds of GB of
                # kept chips and turns into a disk-full cascade for the
                # remaining batches. The retry re-fetches via the manifest
                # fast path in minutes.
                if not args.keep_chips:
                    cleanup_county_chips(county)
                continue
            if not args.keep_chips:
                cleanup_county_chips(county)

    elapsed_h = (time.time() - t_start) / 3600
    log("=" * 70)
    if failures:
        log(f"FINISHED WITH FAILURES in {elapsed_h:.1f} h - failed counties: {failures}")
        log("re-run this same command to retry them (completed counties are skipped).")
        raise SystemExit(1)
    log(f"ALL {len(todo)} COUNTIES COMPLETE in {elapsed_h:.1f} h")


if __name__ == "__main__":
    main()

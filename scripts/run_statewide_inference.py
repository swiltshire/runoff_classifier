"""
Statewide inference driver: run chip-prep + inference + cleanup over every
requested county of ONE state in small disk-bounded batches, resumably and
without manual intervention. Designed to be launched once via nohup and left
alone:

    cd /home/sagemaker-user/runoff_classifier
    mkdir -p outputs/IN/logs
    nohup python -u scripts/run_statewide_inference.py --state IN \
        > outputs/IN/logs/statewide_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    tail -f outputs/IN/logs/statewide_*.log
    ps aux | grep run_statewide_inference

Per batch (default 4 counties) it:
  1. skips counties that already have a final merged detections gpkg in the
     output dir (resume marker - re-running never redoes finished work);
  2. ensures canonical chips: manifest fast path when the county completed a
     prior ensure run (chips fetched straight from S3, no raw tiles), full
     mosaic pipeline otherwise;
  3. runs multi-GPU torch.distributed inference county by county with the
     shared states.DEFAULT_INFER_CONFIG (same as the state pipeline notebooks);
  4. deletes each county's local canonical_tiles/ and mask-cache artifacts
     right after its inference (chips remain durably in S3 + manifest), so
     local disk stays bounded to roughly one batch.

Raw tiles are archived+pruned by ensure_canonical_mosaic_for_counties()
itself. That pruning covers every locally-resident raw tile of the state in
each processed CRS group - including counties from LATER batches whose raw
tiles happen to be on disk already; those get restored from the S3 archive
when their batch comes up.

A disk preflight aborts (resumably) before starting a batch if free space
has dropped below --min_free_gb, rather than dying mid-gdal_translate.
"""

from __future__ import annotations

import argparse
import glob
import os
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

from utils import states  # noqa: E402
from utils.common import normalize_county_key, safe_name  # noqa: E402
from utils.county_boundaries import get_county_boundary_path  # noqa: E402
from utils.prepare_reprojected_tiles import (  # noqa: E402
    ManifestChipsMissingError,
    county_has_manifest,
    ensure_canonical_mosaic_for_counties,
    fetch_county_canonical_chips_from_s3,
)


def log(msg: str) -> None:
    print(f"[statewide {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def county_out_dir(out_dir: Path, county: str) -> Path:
    return out_dir / safe_name(county.strip())


def county_is_done(out_dir: Path, county: str) -> bool:
    """A county is done when a final MERGED detections gpkg
    (`detections_{county}_{YYYYmmdd_HHMMSS}.gpkg`) exists in its output dir;
    per-rank partials (`..._rankN.gpkg`) don't count."""
    for p in glob.glob(str(county_out_dir(out_dir, county) / "detections_*.gpkg")):
        if states.final_detections_county(os.path.basename(p)) == county:
            return True
    return False


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def ensure_chips_for_batch(state: str, batch: list[str], max_workers: int) -> None:
    """Manifest fast path per county where possible; full mosaic pipeline for
    the rest (same split as scripts/prepare_canonical_mosaic.py)."""
    full_counties = []
    for county in batch:
        if county_has_manifest(state, county):
            log(f"{county}: manifest found - fetching chips from S3 (no raw tiles needed)")
            try:
                fetch_county_canonical_chips_from_s3(state, county, verify_sizes=True)
            except ManifestChipsMissingError as e:
                log(f"{county}: stale manifest ({len(e.missing_keys)} chip(s) gone from S3) "
                    f"- falling back to full chip pipeline")
                full_counties.append(county)
        else:
            full_counties.append(county)
    if full_counties:
        log(f"full chip pipeline for: {full_counties}")
        ensure_canonical_mosaic_for_counties(state, counties=full_counties, max_workers=max_workers)


def run_inference(state: str, county: str, model_path: Path, out_dir: Path, nproc: int) -> Path:
    """Run multi-GPU DDP inference for one county; returns the output gpkg path.
    Raises on nonzero exit or missing output."""
    cfg = states.get_state(state)
    infer_config = states.DEFAULT_INFER_CONFIG
    raster_dir = states.canonical_tiles_dir(cfg, county)
    if not glob.glob(str(raster_dir / "*.tif")):
        raise RuntimeError(f"{county}: no canonical tiles at {raster_dir} - ensure step failed?")

    county_dir = county_out_dir(out_dir, county)
    county_dir.mkdir(parents=True, exist_ok=True)
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_gpkg = county_dir / f"detections_{county}_{datestamp}.gpkg"

    # Resolve the county's TIGER boundary in THIS single process before the
    # multi-rank launch, so ranks can't race the download/extraction.
    county_boundary_path = get_county_boundary_path(county, state_fips=cfg.fips)
    log(f"{county}: boundary {county_boundary_path}")

    cmd = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
        str(PROJECT_ROOT / "scripts" / "inference.py"),
        "--task", infer_config["task"],
        "--device", "cuda",
        "--raster_path", str(raster_dir) + os.sep,
        "--checkpoint", str(model_path),
        "--out_vector", str(output_gpkg),
        "--tile_size", str(infer_config["tile_size"]),
        "--stride", str(infer_config["stride"]),
        "--infer_batch", str(infer_config["infer_batch"]),
        "--score_thresh", str(infer_config["score_thresh"]),
        "--normalize", infer_config["normalize"],
        "--nms_iou_thresh", str(infer_config["nms_iou_thresh"]),
        "--final_box_iou", str(infer_config["final_box_iou"]),
        "--mask_path", str(states.nhd_mask_path(cfg)),
        "--min_cover_frac", str(infer_config["min_cover_frac"]),
        "--county_mask_path", str(county_boundary_path),
        "--county_min_cover_frac", str(infer_config["county_min_cover_frac"]),
        "--class_area_csv", str(states.feature_size_thresholds_path()),
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


def cleanup_county_chips(state: str, county: str) -> None:
    """Reclaim a county's local disk after inference.

    - Mask-cache artifacts ({County}_mosaic.vrt + *_mosaic_mask_*.npy/.json)
      are pure derived caches written by inference.py into the county dir at
      full mosaic resolution (~40-80 GB *per mask*, two per county: NHD +
      county boundary). Always delete.
    - canonical_tiles chips are deleted only if the county's S3 manifest
      exists (i.e. they are durably restorable).
    """
    county_dir = states.county_dir(state, county)

    freed = 0
    for f in list(county_dir.glob("*_mosaic_mask_*")) + list(county_dir.glob("*_mosaic.vrt")):
        if f.is_file():
            freed += f.stat().st_size
            f.unlink()
    if freed:
        log(f"{county}: mask-cache artifacts deleted ({freed / 1e9:.0f} GB reclaimed)")

    if not county_has_manifest(state, county):
        log(f"{county}: ⚠ no S3 manifest - keeping local canonical_tiles as the only copy")
        return
    chip_dir = county_dir / "canonical_tiles"
    if chip_dir.is_dir():
        shutil.rmtree(chip_dir)
        log(f"{county}: local canonical_tiles deleted (restorable from S3 manifest)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state", type=str, required=True, choices=sorted(states.STATES))
    parser.add_argument(
        "--counties", type=str, default="",
        help="comma-separated county names; default = every county of --state",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="comma-separated county names to exclude",
    )
    parser.add_argument("--model", type=str, default=states.DEFAULT_MODEL,
                        help="model name under outputs/models/ (default %(default)s)")
    parser.add_argument(
        "--out_dir", type=str, default="",
        help="detections output root (default outputs/{STATE}/inference/{model}); "
             "use a separate dir for regression/test runs",
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
    cfg = states.get_state(args.state)
    state = cfg.code

    known = {normalize_county_key(c): c for c in cfg.counties}
    requested_raw = ([c.strip() for c in args.counties.split(",") if c.strip()]
                     if args.counties else list(cfg.counties))
    unknown = [c for c in requested_raw if normalize_county_key(c) not in known]
    if unknown:
        raise SystemExit(f"not {state} counties: {unknown}")
    requested = [known[normalize_county_key(c)] for c in requested_raw]
    skip = {normalize_county_key(c) for c in args.skip.split(",") if c.strip()}
    requested = [c for c in requested if normalize_county_key(c) not in skip]

    model_path = states.model_dir(args.model) / "model_final.pth"
    if not model_path.exists():
        raise SystemExit(f"model checkpoint not found: {model_path}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else states.inference_dir(state, args.model)
    for required in (states.nhd_mask_path(state), states.feature_size_thresholds_path()):
        if not required.exists():
            raise SystemExit(f"required input missing: {required}")

    done = [c for c in requested if county_is_done(out_dir, c)]
    todo = [c for c in requested if not county_is_done(out_dir, c)]
    log(f"{state}: {len(requested)} counties requested: {len(done)} already done, {len(todo)} to run")
    log(f"model={model_path} out_dir={out_dir}")
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

        ensure_chips_for_batch(state, batch, max_workers=args.max_workers)

        for county in batch:
            try:
                run_inference(state, county, model_path, out_dir, nproc=args.nproc_per_node)
            except Exception as e:
                # keep going - one county's failure shouldn't waste the rest of
                # an unattended overnight run.
                log(f"\u2717 {county} FAILED: {e}")
                failures.append(county)
                # Still reclaim the county's local chips if they're durably
                # restorable from S3 - otherwise a string of failures silently
                # accumulates hundreds of GB of kept chips. The retry re-fetches
                # via the manifest fast path in minutes.
                if not args.keep_chips:
                    cleanup_county_chips(state, county)
                continue
            if not args.keep_chips:
                cleanup_county_chips(state, county)

    elapsed_h = (time.time() - t_start) / 3600
    log("=" * 70)
    if failures:
        log(f"FINISHED WITH FAILURES in {elapsed_h:.1f} h - failed counties: {failures}")
        log("re-run this same command to retry them (completed counties are skipped).")
        raise SystemExit(1)
    log(f"ALL {len(todo)} COUNTIES COMPLETE in {elapsed_h:.1f} h")


if __name__ == "__main__":
    main()

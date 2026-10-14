# Runoff Classifier

Object detection / instance segmentation model (torchvision Faster R-CNN / Mask R-CNN) that identifies agricultural runoff features in Indiana 6-inch orthoimagery.

The pipeline runs on SageMaker and has four stages:

1. **Raw tiles** - download each county's native-CRS 6-inch imagery tiles from Indiana's ArcGIS tile-footprint service.
2. **Canonical tiles** - mosaic and reproject them into seamless, fixed-grid chips in a single canonical CRS (cached in S3).
3. **Training** - merge multi-county labels and train a distributed (4-GPU DDP) model.
4. **Inference** - run the model per county, filtered to the NHD waterway mask, county boundary, and feature size thresholds.

All stages are orchestrated in [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb). The long-running stages also have CLI scripts that can be run detached with `nohup`.

---

## Quick Start

### Environment setup (one-time)

Open `notebooks/pipeline.ipynb` and run the **VRT ENV BOOTSTRAP** cell. It creates the `vrt` conda environment (GDAL, geopandas, rasterio, PyTorch CUDA, etc.) and registers it as the **Python (vrt)** Jupyter kernel. Restart Jupyter and select that kernel.

### Notebook walkthrough

| Section | What it does |
|---|---|
| **Setup** | Resolves project paths and loads the persisted reference CRS (`outputs/.reference_crs`). Defines the training counties and the list of all 92 Indiana counties. |
| **Imagery year pinning and reference CRS** | Downloads `data/training_county_imagery_years.csv` (the imagery year each county's labels were verified against). Sets the pipeline's reference CRS by sampling one tile from Benton County and persists it. |
| **Fetch and Process Input Data** | Syncs training labels from S3 and selects counties. Runs `ensure_canonical_mosaic_for_counties()` to turn raw tiles into canonical chips. |
| **Setup and Run Model Training** | Selects training counties and fetches the class list. Optionally configures single-class mode. Merges labels and builds the training VRT, with optional validation. Writes `classes.json`, sets hyperparameters, and trains either detached (printed `nohup` command) or in the notebook. |
| **Inference on Trained Model** | Downloads the NHD mask and size thresholds, then selects counties. Prints a `nohup` command to verify/download those counties' canonical chips, then runs inference county by county. |
| **Post-inference** | Cleans up local tiles for non-training counties. Uploads final detections to S3. Builds a per-county/per-class detection count CSV and plots statewide canonical-chip coverage from the S3 manifests. |

---

## Project Structure

```
runoff_classifier/
├── notebooks/
│   └── pipeline.ipynb                   # Main workflow notebook
├── scripts/
│   ├── prepare_canonical_mosaic.py      # CLI: raw tiles -> canonical chips (detached/nohup)
│   ├── train.py                         # DDP training
│   ├── validate_training_data.py        # Optional training-data checks
│   ├── inference.py                     # DDP inference for one county / raster
│   ├── run_statewide_inference.py       # Hands-off batched chip-prep + inference + cleanup
│   └── fetch_canonical.ps1              # Download one county's canonical chips (PowerShell + AWS CLI)
├── src/
│   ├── models/
│   │   └── model.py                     # Faster R-CNN & Mask R-CNN builders
│   ├── data/
│   │   ├── dataset.py                   # Training dataset (windows over a VRT)
│   │   └── sampling.py                  # Stratified / distributed weighted samplers
│   └── utils/
│       ├── indiana_cogs.py              # Indiana tile-footprint service: year selection, raw tile download, reference CRS
│       ├── prepare_reprojected_tiles.py # Mosaic-then-reproject canonical chip pipeline (S3 cache + manifests)
│       ├── prepare_multicounty_training.py  # Merge multi-county labels + training VRT
│       ├── make_vrt.py                  # VRT mosaic builder
│       ├── crs_utils.py                 # Label CRS normalization
│       ├── tiling.py                    # Training/inference window generation
│       ├── fast_mask.py                 # Rasterized AOI mask caching + window/box filtering
│       └── county_boundaries.py         # TIGER/Line county boundary download/extraction
├── data/                                # (gitignored)
│   ├── counties/{county}/
│   │   ├── *.shp                        # Training labels
│   │   ├── tiles/                       # Raw native-CRS tiles (transient; archived to S3)
│   │   └── canonical_tiles/             # Canonical chips (used for training/inference)
│   ├── NHDmask_Indiana/                 # NHD waterway AOI mask
│   ├── feature_size_threshholds.csv     # Per-class area thresholds
│   ├── training_county_imagery_years.csv
│   ├── multi_county_labels.gpkg         # Merged training labels (generated)
│   └── multi_county_training_mosaic.vrt # Training VRT (generated)
└── outputs/                             # (gitignored)
    ├── .reference_crs                   # Persisted canonical CRS
    ├── _native_mosaic_cache/            # Per-CRS native + warped VRTs (generated)
    ├── logs/                            # nohup logs
    └── train_multicounty/
        ├── model_epoch_N.pth, model_final.pth, classes.json
        └── inferences_{county}/detections_{county}_{timestamp}.gpkg
```

---

## Stage 1-2: Raw Tiles -> Canonical Tiles

Indiana orthoimagery spans several native state-plane CRSs (sometimes more than one within a county). Training and inference need seamless imagery in a single CRS. `ensure_canonical_mosaic_for_counties()` in `src/utils/prepare_reprojected_tiles.py` implements a **mosaic-then-reproject** pipeline:

1. **Raw download**: each county's raw tiles are restored from our S3 raw-tile archive if available, then completed from Indiana's service via `download_6in_tiles()`. Downloads are pinned to the county's year in `training_county_imagery_years.csv`. Counties without a CSV entry auto-detect the newest complete year and log a warning. The run aborts if more than `max_failed_raw_downloads` tiles fail.
2. **Group by CRS**: all locally available raw tiles are grouped by native EPSG code.
3. **Mosaic + warp**: each group is mosaicked once (`gdalbuildvrt`, 0 = nodata) and warped once to the canonical CRS at 0.5 ft (`gdalwarp -of VRT -tap -r cubic`). This gives the resampler full neighbor context, so there are no edge slivers.
4. **Crop chips**: 8192 px chips are cropped from the warped VRT with `gdal_translate -projwin` on a fixed, origin-anchored grid. Chips are cached in S3, keyed by `(epsg, row, col)`, and reused across counties. Cache hits that are fully blank are re-cropped instead of trusted.
5. **Border invalidation**: when a new county joins a CRS group that already has chips, chips within one chip-width of its tiles are force-regenerated.
6. **Disk bounding**: each group's local chips are pruned after upload. Raw tiles are archived to S3 and deleted once all groups are done.
7. **Manifests**: a per-county S3 manifest (`canonical_mosaic/manifests/{county}.json`) lists that county's chip keys. `canonical_tiles/` is then repopulated from S3.

### Running it

- **Notebook** (training counties): the *Download imagery tiles...* cell calls `ensure_canonical_mosaic_for_counties()` directly.
- **Detached** (recommended for long runs):
  ```bash
  mkdir -p outputs/logs
  nohup python -u scripts/prepare_canonical_mosaic.py --counties "Benton,Boone,Cass" --max_workers 16 \
      > outputs/logs/canonical_mosaic_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  ```
  Counties that already have a manifest take a **fast path**: their chips are fetched straight from S3 and size-verified, with no raw tiles needed. A stale manifest (listed chips missing from S3) falls back to the full pipeline. Use `--full` to always run the full pipeline, or `--force` to regenerate every chip.
- **Collaborators** (Windows, no GDAL): `scripts/fetch_canonical.ps1 -County Benton` downloads a county's chips via its manifest using only the AWS CLI.

The reference CRS defaults to `EPSG:2968`. The notebook's reference CRS cell sets it via `quick_reference_crs_from_county()` and persists it to `outputs/.reference_crs`. The chip pipeline and label prep read that file. With mosaic-then-reproject, this choice only affects efficiency, not correctness.

---

## Stage 3: Training

### Data preparation

```python
from utils.prepare_multicounty_training import prepare_multicounty_training

prepare_multicounty_training(
    county_data_dir=county_data_dir,
    selected_counties=[...],
    out_labels="../data/multi_county_labels.gpkg",
    out_vrt="../data/multi_county_training_mosaic.vrt",
    verified_only=True,
    single_class=False,           # True -> train one focus class
    focus_class=None,
    positive_ratio=80,            # single-class only
    in_class_ratio=50,            # single-class only
    include_background=True,      # single-class only
    background_ratio=50,          # multi-class: cap Background at this % of the dataset
)
```

- Inputs: `data/counties/{county}/*.shp` + `canonical_tiles/*.tif`. Class column variants (`Classname`, `classname`, `Class`) are normalized, and labels are reprojected to the reference CRS.
- **Multi-class mode** (default): 5 classes plus an explicit `Background` class. Background is made of unconfirmed `VerifiedTr=0` rows, capped by `background_ratio`.
- **Single-class mode**: confirmed `focus_class` positives, plus a tunable mix of in-class (`VerifiedTr=0`) and out-of-class negatives as `Background`. Out-of-class features overlapping positives are removed. Set `include_background=False` for positives only.

### Validation (optional)

`scripts/validate_training_data.py::validate_training_data()` checks file/CRS validity, class presence and balance, geometry integrity, and per-county balance (flags counties that are >95% one class). The notebook cell also prints per-class feature sizes in pixels to guide `tile_size` choices.

### Training run

The notebook prints a single-line `nohup` command to paste into a terminal, which survives kernel/session timeouts. Alternatively, it can run training in-kernel. Equivalent command:

```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/train.py \
    --task instance_seg --device cuda \
    --raster_path data/multi_county_training_mosaic.vrt \
    --labels_path data/multi_county_labels.gpkg \
    --out_dir outputs/train_multicounty \
    --classes_json outputs/classes_w_background.json \
    --tile_size 512 --epochs 20 --batch_size 2 --grad_accum 1 --lr 0.0002 \
    --num_workers 8 --prefetch_factor 4 --normalize imagenet \
    --max_per_class 100000 --stratified_sampling
```

- Training windows are label-centered (`tiling.make_label_centered_training_windows`). `--max_per_class` is intentionally left effectively uncapped.
- `--stratified_sampling` uses inverse-frequency weighted sampling (`StratifiedWeightedSampler` / `DistributedWeightedSampler`).
- Outputs: `model_epoch_N.pth`, `model_final.pth`, and `classes.json` in `--out_dir`.

---

## Stage 4: Inference

### Notebook (selected counties)

1. Download the NHD mask and `feature_size_threshholds.csv`.
2. Select counties. For non-training counties, the notebook prints a `nohup scripts/prepare_canonical_mosaic.py ...` command to verify/download their chips. Wait for its `done` log line before continuing.
3. Review `infer_config` and run the inference cell. For each county, it resolves the TIGER/Line county boundary (`get_county_boundary_path()`) and launches 4-GPU inference.

```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/inference.py \
    --task instance_seg --device cuda \
    --raster_path data/counties/Benton/canonical_tiles/ \
    --checkpoint outputs/train_multicounty/model_final.pth \
    --out_vector outputs/train_multicounty/inferences_Benton/detections_Benton_<timestamp>.gpkg \
    --tile_size 512 --stride 256 --infer_batch 5 --score_thresh 0.75 --normalize imagenet \
    --nms_iou_thresh 0.3 --final_box_iou 0.3 \
    --mask_path data/NHDmask_Indiana/NHDfinalMaskIndiana.shp --min_cover_frac 0.01 \
    --county_mask_path <county boundary geojson> --county_min_cover_frac 0.5 \
    --class_area_csv data/feature_size_threshholds.csv
```

Processing:
- Windows outside the NHD AOI mask are skipped.
- Each rank runs per-class NMS, then filters boxes against the full-resolution AOI mask (`MASK_DOWNSAMPLE = 1`; coarser masks drop small objects such as Tile_Outlet). Boxes must also be majority-inside the county boundary.
- Rank 0 merges the rank outputs, runs a final cross-rank box NMS, applies per-class area thresholds, and drops `Background`.

The rasterized masks are cached per county as `*_mosaic_mask_*` files next to the county's `{County}_mosaic.vrt`.

Output: `detections_{county}_{timestamp}.gpkg` with columns `classname`, `score`, `geometry`. Per-rank partials (`..._rankN.gpkg`) are intermediate.

### Statewide (hands-off)

```bash
mkdir -p outputs/logs
nohup python -u scripts/run_statewide_inference.py \
    > outputs/logs/statewide_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The script processes counties in disk-bounded batches (`--batch_size`, default 4). For each batch it ensures chips (manifest fast path where possible), runs inference, and deletes each county's local chips and mask-cache artifacts. Counties with a final detections gpkg are skipped, so re-running resumes. Options: `--counties`, `--skip`, `--min_free_gb`, `--keep_chips`, `--nproc_per_node`. Its `INFER_CONFIG` should be kept in sync with the notebook's.

### Post-inference

- **Upload**: the latest final gpkg per county goes to `s3://.../inference/{MODEL_NAME}/`.
- **Summary**: per-county/per-class counts plus a STATEWIDE row are written to `outputs/{MODEL_NAME}/detection_counts_summary.csv`.
- **Coverage map**: plots canonical-chip footprints from all S3 manifests over Indiana counties.

---

## Classes

```
Bank_Erosion, Spillway, Culvert_Structure, Tile_Inlet, Tile_Outlet, Background
```

In multi-class mode, `Background` is an explicit 6th training class (`outputs/classes_w_background.json`). In single-class mode, the classes are `[focus_class, 'Background']`, or just `[focus_class]` when Background is disabled. `Background` detections are always dropped from inference output.

---

## Adding a County

1. Put its labels at `data/counties/{county}/*.shp` (or in the S3 training prefix synced by the notebook).
2. Add a `County,Data Year` row to `training_county_imagery_years.csv` if its labels were verified against a specific imagery year.
3. Generate canonical chips (notebook fetch step or `scripts/prepare_canonical_mosaic.py`).
4. Select it in the training widget and re-run `prepare_multicounty_training()`.

---

## Troubleshooting

- **Blank chips after a run**: a warning means raw tiles were missing for that area, typically from failed downloads. Re-run the county through `scripts/prepare_canonical_mosaic.py --full`. Failed downloads are retried, and blank cached chips are re-cropped automatically.
- **CRS mismatch**: labels are reprojected to `outputs/.reference_crs`. Make sure chips were generated after the current reference CRS was set.
- **GPU OOM**: reduce `--batch_size` (training) or `--infer_batch` (inference).
- **Slow training**: increase `--num_workers` / `--prefetch_factor`, and monitor with `watch -n1 nvidia-smi`.
- **Disk pressure**: use the notebook's local-tile cleanup cells, or run statewide inference, which cleans up after each county.

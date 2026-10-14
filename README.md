# Runoff Classifier

Object detection / instance segmentation model (torchvision Faster R-CNN / Mask R-CNN) that identifies agricultural runoff features in 6-inch orthoimagery, organized **by state**.

- **One shared model** is trained once (currently on labeled Indiana counties) and stored in S3.
- **Each state** has its own data and inference pipeline: raw tiles -> canonical chips -> inference -> detections. States share code but never data.
  - **Indiana (IN)** is fully supported.
  - **Illinois (IL)** is scaffolded. Its imagery source and canonical CRS are placeholders (see [Adding / finishing a state](#adding--finishing-a-state)).

Everything state-specific comes from one place, [src/utils/states.py](src/utils/states.py): county lists, FIPS, canonical CRS, imagery source, and every local path and S3 key. No function has a default state, because many county names (Lake, Clark, Jackson, Marion, ... 43 in IN/IL alone) exist in more than one state.

The pipeline runs on SageMaker. Long-running stages have CLI scripts meant to be run detached with `nohup`.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| [setup_env.ipynb](notebooks/setup_env.ipynb) | One-time: create the `vrt` conda env (GDAL, geopandas, rasterio, PyTorch CUDA) and register the **Python (vrt)** kernel. |
| [train.ipynb](notebooks/train.ipynb) | Train the shared model from labeled counties of `TRAIN_STATE` and upload it to `s3://.../models/{MODEL_NAME}/`. |
| [pipeline_IN.ipynb](notebooks/pipeline_IN.ipynb) | Indiana inference pipeline. |
| [pipeline_IL.ipynb](notebooks/pipeline_IL.ipynb) | Illinois inference pipeline (identical to IN except `STATE`; diverge as IL needs). |

### train.ipynb
1. **Setup**: `TRAIN_STATE`, `MODEL_NAME`, and the default training counties.
2. **Fetch training inputs**: the imagery-year pins (`{STATE}/training_county_imagery_years.csv`) and labels (`{STATE}/training_labels/{county}/`) from S3.
3. **Canonical chips** for the training counties (`ensure_canonical_mosaic_for_counties`).
4. **Prepare training data**: optional single-class configuration, merge labels and build the training VRT (`prepare_multicounty_training`), optional validation, then write `classes.json`.
5. **Train**: a printed `nohup` command (recommended) or in-notebook. Writes to `outputs/models/{MODEL_NAME}/`.
6. **Upload the model** to S3 (`model_store.upload_model`).

### pipeline_{STATE}.ipynb
1. **Setup**: `STATE`, `MODEL_NAME`.
2. **Inputs**: model (downloaded from S3 if missing), the state's NHD mask, shared size thresholds, and the state's imagery-year pins (optional).
3. **Select counties**, then copy the printed `nohup scripts/prepare_canonical_mosaic.py --state {STATE} ...` command to verify/download their chips.
4. **Run inference** county by county. Uses `states.DEFAULT_INFER_CONFIG` and the county's TIGER boundary for the state's FIPS.
5. **Post-inference**: local tile cleanup (counties with training labels are never offered), detections upload to `s3://.../{STATE}/inference/{MODEL_NAME}/`, a per-county/per-class count CSV, and a statewide chip-coverage map.

---

## Layout

Local and S3 layouts mirror each other (state-first). All paths come from `utils.states`.

```
data/                                           s3://sagemaker-gst-stage.sharing/serge-wiltshire/runoff-classifier-data/
├── shared/                                     ├── shared/
│   ├── feature_size_threshholds.csv            │   ├── feature_size_threshholds.csv
│   └── county_boundaries/{fips}/...  (TIGER)   │   └── classes_w_background.json
├── training/                                   ├── models/{model}/model_final.pth, classes.json
│   ├── multi_county_labels.gpkg                ├── IN/
│   ├── multi_county_training_mosaic.vrt        │   ├── training_county_imagery_years.csv
│   └── classes_w_background.json               │   ├── training_labels/{county}/*.shp
└── {STATE}/                                    │   ├── nhd_mask/nhd_mask_in.*
    ├── training_county_imagery_years.csv       │   ├── canonical_mosaic/{epsg}/chips/chip_{epsg}_r{row}_c{col}.tif
    ├── nhd_mask/nhd_mask_{state}.*             │   ├── canonical_mosaic/manifests/{county}.json
    ├── _canonical_chip_cache/                  │   ├── raw_tile_archive/{county}/*.tif
    └── counties/{county}/                      │   └── inference/{model}/detections_{county}_{ts}.gpkg
        ├── *.shp            (training labels)  └── IL/ ...
        ├── tiles/           (raw, transient)
        ├── canonical_tiles/ (canonical chips)
        └── {County}_mosaic.vrt + AOI mask caches
outputs/
├── models/{model}/              model_epoch_N.pth, model_final.pth, classes.json, train_windows.json, logs
└── {STATE}/
    ├── _native_mosaic_cache/    per-CRS native + warped VRTs
    ├── logs/                    nohup logs
    └── inference/{model}/{county}/detections_{county}_{timestamp}.gpkg, detection_counts_summary.csv
```

### Code
```
src/utils/
├── states.py                      # StateConfig registry (IN, IL) + every path / S3 key + DEFAULT_INFER_CONFIG
├── common.py                      # project_root, safe_name, normalize_county_key, HTTP session, imagery-year CSV loader
├── imagery_sources.py             # per-state raw tile fetcher dispatch (IN -> indiana_cogs; IL -> placeholder)
├── indiana_cogs.py                # Indiana ArcGIS tile-footprint service: year selection + 6-inch tile download
├── prepare_reprojected_tiles.py   # state-aware mosaic-then-reproject canonical chip pipeline, manifests
├── prepare_multicounty_training.py
├── county_boundaries.py           # TIGER/Line county boundaries (state FIPS required)
├── model_store.py                 # upload / download shared model artifacts
├── fast_mask.py, make_vrt.py, tiling.py, crs_utils.py
src/data/ (dataset, samplers), src/models/model.py
scripts/
├── prepare_canonical_mosaic.py    # --state: raw tiles -> canonical chips (manifest fast path)
├── run_statewide_inference.py     # --state: hands-off batched chip-prep + inference + cleanup
├── train.py, inference.py         # DDP training / inference (path-agnostic)
├── validate_training_data.py
├── fetch_canonical.ps1            # -State -County: download a county's chips (PowerShell + AWS CLI)
└── migrate_state_layout.py        # one-time migration to this layout (see below)
```

---

## Canonical chips (raw tiles -> canonical tiles)

States use different native CRSs (often several per state), but training and inference need seamless imagery in one CRS per state. `ensure_canonical_mosaic_for_counties(state, counties)` implements **mosaic-then-reproject**:

1. **Raw tiles**:
   - Each county's tiles are restored from `{STATE}/raw_tile_archive/` if archived, then completed from the state's imagery source (`imagery_sources.download_raw_tiles`).
   - Downloads are pinned to the year in `{STATE}/training_county_imagery_years.csv` when a county has an entry.
   - The run aborts if more than `max_failed_raw_downloads` tiles fail.
2. **Group by CRS**: all of the **state's** local raw tiles are grouped by native EPSG. The grouping never crosses states.
3. **Mosaic + warp**:
   - Each group is mosaicked once (`gdalbuildvrt`, 0 = nodata).
   - It is then warped once to the state's canonical CRS at 0.5 ft (`gdalwarp -of VRT -tap -r cubic`). This removes tile-edge slivers.
4. **Crop chips**:
   - 8192 px chips on a fixed origin-anchored grid are cropped with `gdal_translate -projwin`.
   - They are cached in S3 by `(epsg, row, col)`.
   - Fully blank cache hits are re-cropped rather than trusted.
5. **Border invalidation**: when a county with no local raw tiles joins a CRS group that already has chips, chips within one chip-width of its tiles are force-regenerated.
6. **Disk bounding**:
   - Each group's local chips are pruned once uploaded.
   - Raw tiles are archived to S3 and deleted after all groups are done.
7. **Manifests**:
   - `{STATE}/canonical_mosaic/manifests/{county}.json` records `state`, `county`, `canonical_crs`, `chip_size_px` and the chip `keys`.
   - Consumers validate the manifest before downloading (`validate_manifest`): state, county, CRS, and that every key is under the state's prefix.

**Safety:**
- A per-state lock stops two chip runs for the same state from overlapping.
- Each worker gets an immutable `ChipContext` and its own S3 client.
- The state's canonical CRS must be a projected CRS in feet, because chips are 0.5 ft and size thresholds are in sq ft (`states.require_canonical_crs`).

Running it detached (recommended):

```bash
mkdir -p outputs/IN/logs
nohup python -u scripts/prepare_canonical_mosaic.py --state IN --counties "Benton,Boone,Cass" --max_workers 16 \
    > outputs/IN/logs/canonical_mosaic_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

- Counties with a manifest use the **fast path**: their chips are fetched from S3 and size-verified, with no raw tiles needed.
- A stale manifest falls back to the full pipeline. `--full` forces the full pipeline; `--force` regenerates every chip.
- Collaborators (Windows, no GDAL) can run `scripts/fetch_canonical.ps1 -State IN -County Benton`.

---

## Training

The `train.ipynb` flow is wrapped around:

```python
prepare_multicounty_training(
    "IN", selected_counties=[...],
    out_labels=str(states.training_data_dir() / "multi_county_labels.gpkg"),
    out_vrt=str(states.training_data_dir() / "multi_county_training_mosaic.vrt"),
    verified_only=True, single_class=False, background_ratio=50, ...)
```

- Labels come from `data/{STATE}/counties/{county}/*.shp`. Class column variants are normalized, and labels are reprojected to the state's canonical CRS.
- **Multi-class** (default): the 5 classes, plus an explicit `Background` class built from unconfirmed `VerifiedTr=0` rows (capped by `background_ratio`).
- **Single-class**: confirmed `focus_class` positives, plus a tunable mix of in-class and out-of-class negatives as `Background` (or positives only).
- **Validation** (optional, `scripts/validate_training_data.py`) checks file/CRS validity, class balance, geometry, and per-county balance. It also prints per-class feature pixel sizes.
- **Training command** (the notebook prints it with absolute paths):
  ```bash
  python -m torch.distributed.run --nproc_per_node=4 scripts/train.py --task instance_seg --device cuda \
      --raster_path data/training/multi_county_training_mosaic.vrt --labels_path data/training/multi_county_labels.gpkg \
      --out_dir outputs/models/train_multicounty --classes_json data/training/classes_w_background.json \
      --tile_size 512 --epochs 20 --batch_size 2 --grad_accum 1 --lr 0.0002 --num_workers 8 --prefetch_factor 4 \
      --normalize imagenet --max_per_class 100000 --stratified_sampling
  ```
- Then upload with `model_store.upload_model("train_multicounty")` (last notebook cell). State pipelines download the model with `ensure_model_local()`.

---

## Inference

Per county (notebook or statewide driver):

```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/inference.py --task instance_seg --device cuda \
    --raster_path data/IN/counties/Benton/canonical_tiles/ --checkpoint outputs/models/train_multicounty/model_final.pth \
    --out_vector outputs/IN/inference/train_multicounty/Benton/detections_Benton_<timestamp>.gpkg \
    --tile_size 512 --stride 256 --infer_batch 5 --score_thresh 0.75 --normalize imagenet \
    --nms_iou_thresh 0.3 --final_box_iou 0.3 \
    --mask_path data/IN/nhd_mask/nhd_mask_in.shp --min_cover_frac 0.01 \
    --county_mask_path data/shared/county_boundaries/18/Benton.geojson --county_min_cover_frac 0.5 \
    --class_area_csv data/shared/feature_size_threshholds.csv
```

Processing:
- Windows outside the NHD AOI mask are skipped.
- Each rank runs per-class NMS, then filters boxes against the full-resolution AOI mask (`MASK_DOWNSAMPLE = 1`). Boxes must also be majority-inside the county boundary. The boundary comes from `get_county_boundary_path(county, state_fips=...)`.
- Rank 0 merges the rank outputs, runs a final cross-rank box NMS, applies per-class area thresholds, and drops `Background`.
- Output columns: `classname`, `score`, `geometry`.

### Statewide (hands-off)

```bash
mkdir -p outputs/IN/logs
nohup python -u scripts/run_statewide_inference.py --state IN \
    > outputs/IN/logs/statewide_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

- Runs in disk-bounded batches. For each batch: ensure chips (fast path where possible), run inference, then delete local chips and mask caches.
- Resumable: counties with a final detections gpkg are skipped.
- Options: `--counties`, `--skip`, `--model`, `--out_dir` (e.g. a separate dir for test runs), `--batch_size`, `--min_free_gb`, `--keep_chips`, `--nproc_per_node`.
- Inference parameters come from `states.DEFAULT_INFER_CONFIG`, shared with the notebooks.

---

## Classes

`Bank_Erosion, Spillway, Culvert_Structure, Tile_Inlet, Tile_Outlet, Background`
- **Multi-class mode:** `Background` is an explicit 6th class (`data/training/classes_w_background.json`).
- **Single-class mode:** `[focus_class, 'Background']`, or `[focus_class]` without Background.
- `Background` detections are always dropped from inference output.

---

## Adding / finishing a state

1. **Register it:** add a `StateConfig` to `STATES` in `src/utils/states.py` with:
   - `code`, `name`, `fips`
   - `counties` (Census names; the folder name is `safe_name(county)`)
   - `canonical_crs`: a projected CRS in **feet** covering the whole state
   - `imagery_source`
2. **Imagery source:** implement a fetcher in `src/utils/imagery_sources.py` and register it in `_FETCHERS`. It must write GeoTIFF tiles into `dest_dir` and return `{"year": ..., "failed": n}`. Alternatively, pre-stage raw tiles in `{STATE}/raw_tile_archive/{county}/` on S3.
3. **Inputs on S3:**
   - NHD mask: `{STATE}/nhd_mask/nhd_mask_{state}.*` (all shapefile sidecars).
   - Optional imagery-year pins: `{STATE}/training_county_imagery_years.csv` (`County,Data Year`).
4. **Notebook:** copy `notebooks/pipeline_IN.ipynb` to `pipeline_{STATE}.ipynb` and set `STATE`.

**Illinois TODO:**
- Choose a `canonical_crs` (e.g. a State Plane ftUS zone).
- Implement the IL imagery fetcher.
- Upload `IL/nhd_mask/nhd_mask_il.*`.

Until the first two are done, IL chip prep stops with a clear `NotImplementedError` and never touches the Indiana service.

---

## One-time migration to the multi-state layout (SageMaker)

`scripts/migrate_state_layout.py` moves the legacy Indiana-only layout (S3 and local disk) into the layout above. Every phase is journaled in `outputs/_migration/` and is safe to re-run. The branch `pre-state-reorg` keeps the last single-state version of the code.

1. **Quiesce:** stop every chip/training/inference job and notebook kernel. Nothing may write to the old layout until cutover.
2. **S3**, from the project root on SageMaker. The copy is about 27 TB server-side and takes hours, so run it in tmux or with nohup.
   ```bash
   python scripts/migrate_state_layout.py s3-inventory      # freeze source list; aborts on unmapped keys / collisions / existing destinations
   python scripts/migrate_state_layout.py s3-copy           # If-Match pinned copies, CRC-verified where available; rerun to resume
   python scripts/migrate_state_layout.py s3-verify         # every planned object present with correct size
   python scripts/migrate_state_layout.py s3-manifests      # rewrite manifests LAST (strict key mapping) -> IN/canonical_mosaic/manifests/
   ```
3. **Local:**
   ```bash
   python scripts/migrate_state_layout.py local-apply --dry-run
   python scripts/migrate_state_layout.py local-apply
   python scripts/migrate_state_layout.py local-cleanup          # dry run: training VRT, native mosaic VRTs, stale AOI mask caches, .reference_crs
   python scripts/migrate_state_layout.py local-cleanup --apply
   python scripts/migrate_state_layout.py model-upload           # outputs/models/train_multicounty -> s3 models/
   ```
4. **Acceptance:**
   - Re-run IN inference for one or two small counties into a test dir, then compare against their existing detections (tolerance-matched per class):
     ```bash
     python scripts/run_statewide_inference.py --state IN --counties Ohio,Union --out_dir outputs/IN/_acceptance --keep_chips
     python scripts/migrate_state_layout.py compare-detections \
         outputs/IN/inference/train_multicounty/Ohio/detections_Ohio_<old_ts>.gpkg \
         outputs/IN/_acceptance/Ohio/detections_Ohio_<new_ts>.gpkg
     ```
   - Run the IN coverage map (92/92 manifests).
   - Check that `prepare_multicounty_training` gives the same label count.
   - Check that IL setup runs and IL chip prep fails with the "not configured" error.
5. **Cutover:** only after acceptance passes:
   ```bash
   python scripts/migrate_state_layout.py s3-delete-old --confirm-acceptance
   ```
   This refuses to run if any legacy object changed since the inventory. A bucket admin must then purge the noncurrent versions under the legacy prefixes (`canonical_mosaic/`, `raw_tile_archive/`, `training/`, `inference/`, `nhd_masks/`, root files). Until then they are billed at Standard storage.

**Rollback:**
- **Before cutover:** `git checkout pre-state-reorg` plus `python scripts/migrate_state_layout.py local-reverse`. The legacy S3 objects are untouched.
- **After the admin purge:** a reverse S3 copy is required (`IN/...` back to the legacy keys; manifests back to unprefixed keys).

---

## Troubleshooting

- **"canonical CRS is not configured" / "no raw imagery source":** the state isn't finished. See [Adding / finishing a state](#adding--finishing-a-state).
- **"invalid canonical manifest":** the manifest belongs to another state, still points at legacy keys, or the CRS changed. Re-run the county with `prepare_canonical_mosaic.py --state ... --full`.
- **"another canonical chip run for {STATE} is in progress":** wait for it, or confirm the other process has exited.
- **Blank chips after a run:** raw tiles were missing for that area. Re-run with `--full`; failed downloads are retried and blank cached chips are re-cropped.
- **GPU OOM:** reduce `--batch_size` (training) or `--infer_batch` (inference).
- **Slow training:** increase `--num_workers` / `--prefetch_factor`, and watch `nvidia-smi`.

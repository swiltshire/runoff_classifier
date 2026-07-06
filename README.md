# Runoff Classifier

Object detection model to identify agricultural runoff sources using remote sensing imagery.

## Quick Start

### Environment Setup

Create the conda environment (one-time):

```bash
cd notebooks/
jupyter notebook pipeline.ipynb
```

Run cell 1 (VRT ENV BOOTSTRAP) to create the `vrt` conda environment and register it as a Jupyter kernel.

### Typical Workflow

1. **Fetch data** — Sync training data and imagery from S3
2. **Prepare training data** — Merge multi-county labels and build VRT mosaic
3. **Validate** — Check data quality before training
4. **Train** — Run distributed training on 4 GPUs
5. **Infer** — Generate predictions on target county

All steps are orchestrated in `notebooks/pipeline.ipynb`.

---

## Project Structure

```
runoff_classifier/
├── notebooks/
│   └── pipeline.ipynb           # Main workflow notebook
├── scripts/
│   ├── train.py                 # DDP-based training script
│   ├── inference.py             # DDP-based inference script
│   └── validate_training_data.py # Data validation & diagnostics
├── src/
│   ├── models/
│   │   └── model.py             # Faster R-CNN & Mask R-CNN builders
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset class
│   │   └── sampling.py          # Stratified sampling for balanced training
│   └── utils/
│       ├── prepare_multicounty_training.py  # Merge & validate labels
│       ├── tiling.py            # Training window generation
│       ├── make_vrt.py          # VRT mosaic builder
│       ├── crs_utils.py         # CRS normalization helpers
│       ├── fast_mask.py         # Mask-based filtering
│       ├── prepare_reprojected_tiles.py    # S3 imagery prep
│       └── indiana_cogs.py      # Indiana COG helpers
├── data/
│   ├── counties/                # County-specific data (shapefiles + tiles)
│   ├── multi_county_labels.gpkg # Merged training labels (generated)
│   └── multi_county_training_mosaic.vrt  # VRT mosaic (generated)
└── outputs/
    ├── train_multicounty/       # Training outputs (models, logs)
    ├── inferences_*/            # Per-county inference results
    └── classes.json             # Class names (auto-generated)
```

---

## Key Features

### Multi-County Training
- Consolidates labels from multiple counties into unified training dataset
- Handles schema inconsistencies (different column names across counties)
- Supports negative examples (Background class from VerifiedTr==0)
- Per-county class balancing via `StratifiedWeightedSampler`

### Data Validation
Single consolidated validation step checks:
1. File and CRS validity
2. Class presence and balance
3. Geometry integrity and spatial overlap
4. Per-county balance (flags data poisoning)

Run validation in notebook after `prepare_multicounty_training()` — exits with error if issues found.

### Distributed Training
- DDP (DistributedDataParallel) on up to 4 GPUs
- Stratified sampling ensures balanced class representation
- ImageNet normalization (configurable)
- Checkpoint saving per epoch

### Inference
- Distributed inference with per-rank NMS
- Spatial filter (NHD mask) and size thresholds
- Outputs GeoPackage with predictions

---

## Class Definitions

```python
classes = [
    'Bank_Erosion',          # Eroded riverbanks
    'Spillway',              # Water management structures
    'Culvert_Structure',     # Culvert installations
    'Tile_Inlet',            # Tile drainage inlets
    'Tile_Outlet'            # Tile drainage outlets
]
```

Background (class 0) is implicit in Faster R-CNN / Mask R-CNN.

---

## Workflow Details

### 1. Data Preparation

```python
from utils.prepare_multicounty_training import prepare_multicounty_training

prepare_multicounty_training(
    county_data_dir='data/counties',
    selected_counties=['Benton', 'Boone', 'Brown', ...],
    out_labels='data/multi_county_labels.gpkg',
    out_vrt='data/multi_county_training_mosaic.vrt',
    verified_only=True,
    debug_plots=False
)
```

**Inputs:**
- County directories with structure: `{county}/tiles/*.tif` + `{county}/*.shp`
- Shapefiles may use 'Classname', 'classname', or 'Class' columns (auto-normalized)

**Outputs:**
- GeoPackage with merged labels (all counties + consistent CRS)
- VRT mosaic of all tiles

### 2. Data Validation

```python
from scripts.validate_training_data import validate_training_data

is_valid, errors, warnings, summary = validate_training_data(
    raster_path='data/multi_county_training_mosaic.vrt',
    labels_path='data/multi_county_labels.gpkg',
    expected_classes=['Bank_Erosion', 'Spillway', 'Culvert_Structure', 'Tile_Inlet', 'Tile_Outlet'],
    verbose=True
)

if not is_valid:
    print(f"Fix errors before training: {errors}")
```

**Outputs:**
- Per-county class distribution summary
- Flags suspicious counties (>95% single class indicates data poisoning)
- Geometry validity check
- Spatial coverage statistics

### 3. Training

```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/train.py \
    --task instance_seg \
    --raster_path "data/multi_county_training_mosaic.vrt" \
    --labels_path "data/multi_county_labels.gpkg" \
    --out_dir "outputs/train_multicounty" \
    --tile_size 512 \
    --epochs 20 \
    --batch_size 2 \
    --lr 0.0002 \
    --stratified_sampling
```

**Key Parameters:**
- `--stratified_sampling`: Enable balanced sampling across counties
- `--max_per_class`: Limit samples per class per epoch
- `--grad_accum`: Gradient accumulation steps
- `--normalize`: 'imagenet' or 'none'

**Outputs:**
- Model checkpoints: `outputs/train_multicounty/model_epoch_N.pth`
- Class definitions: `outputs/train_multicounty/classes.json`
- Training logs (stdout)

### 4. Inference on Multiple Counties

Inference can be run on one or more counties. The notebook provides:
1. County selection widget (select single or multiple counties)
2. Parameter configuration (tile size, score threshold, etc.)
3. Automatic loop that runs inference sequentially on each selected county

**In notebook:**
1. Select counties in "Select counties for inference" widget
2. Review parameters in "Setup inference parameters" cell
3. Run "Run inference on selected counties" cell — outputs saved to `outputs/train_multicounty/inferences_{county}/detections.gpkg` for each county

**Command-line (single county):**
```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/inference.py \
    --task instance_seg \
    --checkpoint "outputs/train_multicounty/model_final.pth" \
    --raster_path "data/counties/Benton/tiles/" \
    --out_vector "outputs/inferences_Benton/detections.gpkg" \
    --tile_size 512 \
    --score_thresh 0.2 \
    --normalize imagenet \
    --nms_iou_thresh 0.3 \
    --mask_path "data/NHDmask_Indiana/NHDfinalMaskIndiana.shp" \
    --min_cover_frac 0.01 \
    --class_area_csv "data/feature_size_threshholds.csv"
```

**Key Parameters:**
- `--score_thresh`: Confidence threshold (0.2 typical)
- `--nms_iou_thresh`: NMS overlap threshold (0.3 typical)
- `--min_cover_frac`: Minimum fraction of mask coverage (0.01 = 1%)
- `--stride`: Window stride (512//2 = 50% overlap)
- `--infer_batch`: Batch size for inference (5 typical)

**Outputs (per county):**
- GeoPackage: `outputs/train_multicounty/inferences_{county}/detections.gpkg`
  - Columns: `classname`, `score`, `geometry`
  - Geometries clipped to valid areas (NHD mask + size filters applied)

---

## Troubleshooting

### Model Predicts Single Class Heavily

Check data quality with validation:
```python
from scripts.validate_training_data import validate_training_data
validate_training_data(raster_path=..., labels_path=..., verbose=True)
```

Look for per-county warnings like "⚠ County X: 95.0% Tile_Outlet" — indicates poisoned data source.

### CRS Mismatch Errors

Ensure all county shapefiles are reprojected to target CRS before training. The `prepare_multicounty_training()` function will error if CRS doesn't match tiles.

### GPU Out of Memory

Reduce `--batch_size` or `--max_per_class`.

### Slow Training

Check CPU-to-GPU pipeline:
- Increase `--num_workers` (up to CPU count)
- Increase `--prefetch_factor` (3-4)
- Use `watch -n1 nvidia-smi` to monitor GPU utilization

---

## Recent Changes

### Data Validation Consolidation
- **Removed:** `validate_multicounty.py`, `debug_multicounty.py`, `analyze_training_data.py`
- **Added:** Single consolidated `scripts/validate_training_data.py`
- **Integration:** Validation now runs directly in notebook after data prep
- **Benefit:** Catches data poisoning (extreme class imbalance) before training

### Code Cleanup
- Removed unused imports across codebase
- Removed unused `compute_class_weights()` function
- Fixed import organization (moved from inside functions to module level)
- Updated file path documentation
- Consistent comment formatting

---

## Development Notes

### Adding New Counties

1. Ensure county data structure: `data/counties/{county}/tiles/*.tif` + `{county}/*.shp`
2. Add to selected counties in notebook
3. Run `prepare_multicounty_training()` with new county included
4. Validate before training

### Modifying Model Architecture

Edit `src/models/model.py`:
- `build_fasterrcnn_model()` for detection-only
- `build_maskrcnn_model()` for instance segmentation

Retraining required for different architecture.

### Class Changes

Update in notebook:
- `config['classes']` list
- `expected_classes` in validation call
- Class definitions in this README

Data preparation will auto-detect classes from shapefiles.

---

## References

- **PyTorch:** torchvision Faster R-CNN / Mask R-CNN implementations
- **Geospatial:** rasterio, geopandas, shapely
- **Distributed Training:** DDP (DistributedDataParallel)

---

## Status

✅ **Production-ready** with multi-county support and comprehensive data validation.

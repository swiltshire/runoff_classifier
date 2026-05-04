# Before & After: Code Comparisons

## Bug #1: Variable Shadowing in tiling.py

### BEFORE (WRONG ❌)
```python
def make_label_centered_training_windows(raster_path: str,
                                         labels_path: str,
                                         tile_size: int,
                                         max_per_class: int | None = None,
                                         classname_field: str = 'Classname',
                                         jitter: int = 64) -> List[Window]:
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        transform = src.transform
    gdf = gpd.read_file(labels_path)

    windows = []
    grouped = gdf.groupby(classname_field)
    for cls, group in grouped:
        count = 0
        for _, row in group.iterrows():              # ← 'row' is DataFrame row
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            inv = ~transform
            col, row_px = inv * (cx, cy)             # ← OVERWRITES 'row'!
            # ... rest of code tries to use 'row' but it's now a tuple
```

**Problem:** The variable `row` from the DataFrame iteration gets overwritten by the unpacking `col, row_px = ...`. Later references to `row` would either fail or give wrong values.

### AFTER (CORRECT ✓)
```python
def make_label_centered_training_windows(raster_path: str,
                                         labels_path: str,
                                         tile_size: int,
                                         max_per_class: int | None = None,
                                         classname_field: str = 'Classname',
                                         jitter: int = 64) -> List[Window]:
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        transform = src.transform
    gdf = gpd.read_file(labels_path)

    windows = []
    grouped = gdf.groupby(classname_field)
    for cls, group in grouped:
        count = 0
        for _, label_row in group.iterrows():        # ← Renamed to 'label_row'
            cx, cy = label_row.geometry.centroid.x, label_row.geometry.centroid.y
            inv = ~transform
            col, row_px = inv * (cx, cy)             # ← No shadowing
            # ... rest of code is correct
```

**Fix:** Simple but critical - renamed loop variable to `label_row` to avoid shadowing.

---

## Bug #2: Missing Class Validation in dataset.py

### BEFORE (SILENT FAILURES ❌)
```python
class ObjectDetectionTilesDataset(Dataset):
    def __init__(
        self,
        raster_path: str,
        labels_path: str,
        classes: List[str],
        tile_windows: List[Window],
        classname_field: str = 'Classname',
        normalize: str = 'none',
        include_masks: bool = False,
    ) -> None:
        super().__init__()
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.classname_field = classname_field
        self.classes = classes
        self.class_to_idx = {c: i + 1 for i, c in enumerate(classes)}
        self.tile_windows = tile_windows
        self.normalize = normalize
        self.include_masks = include_masks

        # read geodataframe and ensure crs aligns with raster
        with rasterio.open(self.raster_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.height = src.height
            self.width = src.width
            gdf = gpd.read_file(self.labels_path)
            if gdf.crs is None:
                raise ValueError('labels must have a valid crs')
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)
            self.gdf = gdf
            # No validation that classes match! ← SILENT FAILURE
```

**Problem:** When merging counties, County A might have classes [A, B, C] and County B might have [A, D, E]. The dataset expects [A, B, C, D, E] but will never see B in County B windows, causing:
- Class B gradient will be all zeros from County B
- Model learns B incorrectly (biased toward County A patterns)
- Poor generalization to unseen data

### AFTER (CLEAR WARNINGS ✓)
```python
class ObjectDetectionTilesDataset(Dataset):
    def __init__(
        self,
        raster_path: str,
        labels_path: str,
        classes: List[str],
        tile_windows: List[Window],
        classname_field: str = 'Classname',
        normalize: str = 'none',
        include_masks: bool = False,
    ) -> None:
        super().__init__()
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.classname_field = classname_field
        self.classes = classes
        self.class_to_idx = {c: i + 1 for i, c in enumerate(classes)}
        self.tile_windows = tile_windows
        self.normalize = normalize
        self.include_masks = include_masks

        # read geodataframe and ensure crs aligns with raster
        with rasterio.open(self.raster_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.height = src.height
            self.width = src.width
            gdf = gpd.read_file(self.labels_path)
            if gdf.crs is None:
                raise ValueError('labels must have a valid crs')
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)
            self.gdf = gdf

        # ← NEW: Validate that all expected classes appear in the labels
        if self.classname_field in self.gdf.columns:
            present_classes = set(self.gdf[self.classname_field].unique())
            expected_classes = set(self.classes)
            missing_classes = expected_classes - present_classes
            if missing_classes:
                print(f"WARNING: The following classes are not present in labels: {missing_classes}")
                print(f"         Present classes: {present_classes}")
                print(f"         This may cause training instability with multi-county data.")
            # check if labels contain classes not in the expected list
            unexpected_classes = present_classes - expected_classes
            if unexpected_classes:
                print(f"WARNING: Labels contain unexpected classes: {unexpected_classes}")
                print(f"         These will be treated as background (class 0) and skipped.")
```

**Fix:** Added explicit validation that alerts users to data issues before training even starts.

---

## Bug #3: No Stratified Sampling in train.py

### BEFORE (RANDOM SAMPLING ❌)
```python
def main():
    args = parse_args()
    # ... setup ...

    dataset = ObjectDetectionTilesDataset(
        raster_path=args.raster_path,
        labels_path=args.labels_path,
        classes=classes,
        tile_windows=tile_windows,
        classname_field='Classname',
        normalize=args.normalize,
        include_masks=(args.task == 'instance_seg'),
    )

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True  # ← Random shuffle - doesn't account for class balance

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,        # ← No stratification
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        persistent_workers=(True if args.num_workers > 0 else False),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    # print batches per rank
    if is_main_process():
        print(f"INFO [debug] dataloader batches per rank ≈ {len(dataloader)}")

    # ... training loop ...
```

**Problem:** Random sampling doesn't account for multi-county class imbalance. Example:
- County A has 1000 Bank_Erosion examples
- County B has 50 Spillway examples
- Random sampling → 95% of batches are Bank_Erosion → model overfits to Bank_Erosion

### AFTER (STRATIFIED SAMPLING ✓)
```python
def parse_args():
    parser = argparse.ArgumentParser(description='train detection or instance segmentation on geospatial tiles (ddp-ready)')
    # ... existing args ...
    parser.add_argument('--stratified_sampling', action='store_true',
                        help='use stratified sampling to balance classes (recommended for multi-county)')
    return parser.parse_args()


def main():
    args = parse_args()
    # ... setup ...

    dataset = ObjectDetectionTilesDataset(
        raster_path=args.raster_path,
        labels_path=args.labels_path,
        classes=classes,
        tile_windows=tile_windows,
        classname_field='Classname',
        normalize=args.normalize,
        include_masks=(args.task == 'instance_seg'),
    )

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        shuffle = False
    elif args.stratified_sampling:  # ← NEW: Use stratified sampling
        # use stratified sampling for single-GPU multi-county training
        sampler = StratifiedWeightedSampler(dataset, args.labels_path, classname_field='Classname')
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,  # ← Now uses weighted sampler if stratified_sampling
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        persistent_workers=(True if args.num_workers > 0 else False),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    # print batches per rank
    if is_main_process():
        print(f"INFO [debug] dataloader batches per rank ≈ {len(dataloader)}")
        if args.stratified_sampling:  # ← Log when stratified
            print(f"INFO [debug] using stratified sampling for balanced class representation")

    # ... training loop ...
```

**Fix:** Added `--stratified_sampling` flag and `StratifiedWeightedSampler` that weights windows by class rarity.

---

## New: StratifiedWeightedSampler (src/data/sampling.py)

### What It Does
```python
class StratifiedWeightedSampler(Sampler):
    """
    Weights each window by the rarity of classes it contains.

    Example:
    - Window 1: contains Bank_Erosion (appears 1000x) → weight = 1.0
    - Window 2: contains Spillway (appears 50x) → weight = 20.0
    - Window 3: contains Bank_Erosion + Spillway → weight = 10.5 (average)

    During training:
    - Window 2 gets sampled 20x more often
    - Balanced representation despite class imbalance
    """

    def __init__(self, dataset, labels_path: str, classname_field: str = 'Classname', replacement: bool = True):
        # 1. Map each window to its classes
        # 2. Count class frequency across all windows
        # 3. Compute weight = max_freq / class_freq
        # 4. Average weight per window
        pass

    def __iter__(self):
        # Sample windows with probability proportional to weights
        indices = np.random.choice(
            len(self.dataset),
            size=len(self.dataset),
            p=self.weights,
            replace=self.replacement
        )
        return iter(indices.tolist())
```

---

## Usage Examples

### Single County (No Changes Needed)
```bash
python scripts/train.py \
    --raster_path county_mosaic.vrt \
    --labels_path county_labels.gpkg \
    --epochs 20
```

### Multiple Counties (With Stratified Sampling - RECOMMENDED)
```bash
# 1. Validate data
python scripts/validate_multicounty.py \
    --raster_path merged.vrt \
    --labels_path merged.gpkg

# 2. Debug issues
python scripts/debug_multicounty.py \
    --raster_path merged.vrt \
    --labels_path merged.gpkg

# 3. Train with stratified sampling
python scripts/train.py \
    --raster_path merged.vrt \
    --labels_path merged.gpkg \
    --stratified_sampling \  # ← NEW FLAG
    --epochs 20 \
    --batch_size 2
```

---

## Expected Output with Fixes

### Before Validation
```
(Would fail silently or produce nonsensical detections)
```

### After Validation
```
============================================================
MULTI-COUNTY DATASET VALIDATION
============================================================

[1/6] Checking raster...
  ✓ Raster: 5000x5000 pixels
    CRS: EPSG:32616
    Resolution: (-0.5, 0.5)
    Bounds: BoundingBox(left=123456.0, bottom=4500000.0, right=125956.0, top=4502500.0)

[2/6] Checking labels...
  ✓ Labels: 2541 features
    CRS: EPSG:32616

[3/6] Checking class coverage...
  ✓ All expected classes present

  Class distribution:
    Bank_Erosion: 1200 (47.2%)
    Spillway: 680 (26.7%)
    Culvert_Structure: 450 (17.7%)
    Tile_Inlet: 150 (5.9%)
    Tile_Outlet: 61 (2.4%)

[4/6] Checking per-county class coverage...
  Counties: ['County_A', 'County_B', 'County_C']
  ✓ County_A: has all classes
  ✓ County_B: has all classes
  ⚠ County_C: missing {'Tile_Outlet'}

[5/6] Checking spatial overlap...
  ✓ Labels within raster bounds
    Coverage: ~25.3% of raster area

[6/6] Checking geometry integrity...
  ✓ All geometries valid

============================================================
✓ VALIDATION PASSED

⚠ 1 warnings:
  - County_C: missing {'Tile_Outlet'}
============================================================
```

### During Training
```
INFO [debug] starting training (world_size=1)
INFO [debug] using labels: .../merged_labels.gpkg
INFO [debug] dataloader batches per rank ≈ 127
INFO [debug] using stratified sampling for balanced class representation
epoch 1/20: 100%|██████| 127/127 [02:15<00:00, 1.07s/batch]
epoch 1 average loss (global): 0.8234
epoch 2/20: 100%|██████| 127/127 [02:14<00:00, 1.06s/batch]
epoch 2 average loss (global): 0.5621
...
```

---

## Key Takeaways

| Aspect | Impact | Benefit |
|--------|--------|---------|
| Variable shadowing fix | Reliable window generation | No silent failures in training data |
| Class validation | Early error detection | Can adjust data before wasting GPU time |
| Stratified sampling | Balanced class learning | Better multi-county generalization |
| New diagnostics | Data insights | Understand what the model is learning from |

All three fixes are complementary - use them together for best results on multi-county data!

# Multi-County Training Debug & Fixes

## Issues Identified

### 1. **Variable Shadowing Bug** (CRITICAL)
**Location:** `src/utils/tiling.py` in `make_label_centered_training_windows()`

**Problem:**
```python
for _, row in group.iterrows():  # 'row' is DataFrame row object
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    inv = ~transform
    col, row_px = inv * (cx, cy)  # OVERWRITES 'row' with tuple!
    # ... later in loop, accessing 'row' would fail or give wrong value
```

The variable `row` (DataFrame row from the loop) was being overwritten by `row_px` (a numeric value). This caused unpredictable behavior when processing multiple labels, especially with multi-county data where label iteration is more complex.

**Fix:** Renamed the loop variable to `label_row` to avoid shadowing.

---

### 2. **Missing Class Validation**
**Location:** `src/data/dataset.py` in `ObjectDetectionTilesDataset.__init__()`

**Problem:**
When merging labels from multiple counties, some classes might be completely absent from certain counties. For example:
- County A: has Bank_Erosion, Spillway, Culvert_Structure
- County B: has Bank_Erosion, Tile_Inlet, Tile_Outlet
- County C: has Spillway only

When merged and training starts, the model expects to learn all 5 classes, but windows from County C will never show Culvert_Structure or Tile_Inlet labels. This causes:
- Incorrect gradients for missing classes
- Biased predictions toward classes that appear in all counties
- Poor generalization across counties

**Fix:** Added validation warnings to detect and report missing/unexpected classes.

---

### 3. **No Stratified Sampling Strategy**
**Location:** `scripts/train.py` dataloader setup

**Problem:**
When training on multiple counties with `shuffle=True`, the dataloader samples randomly. This can lead to:
- **Batch sampling bias:** A batch might contain 4 samples from County A and 0 from County B
- **Class imbalance:** If County A has many Bank_Erosion examples but County B has few, the model learns County A's distribution
- **Poor multi-county generalization:** Model overfits to dominant county/class patterns

**Fix:** Implemented `StratifiedWeightedSampler` that:
- Tracks which classes appear in each window
- Assigns higher sampling probability to windows containing rare classes
- Ensures balanced representation across classes during training

---

## Changes Made

### 1. Fixed Variable Shadowing
```python
# Before (WRONG):
for _, row in group.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    col, row_px = inv * (cx, cy)  # overwrites 'row'!

# After (CORRECT):
for _, label_row in group.iterrows():
    cx, cy = label_row.geometry.centroid.x, label_row.geometry.centroid.y
    col, row_px = inv * (cx, cy)  # no shadowing
```

### 2. Added Class Validation Warnings
```python
# In dataset.py __init__:
present_classes = set(self.gdf[self.classname_field].unique())
missing_classes = expected_classes - present_classes
if missing_classes:
    print(f"WARNING: Classes not in labels: {missing_classes}")
```

### 3. Stratified Sampling
Created `src/data/sampling.py` with `StratifiedWeightedSampler` class that:
- Analyzes the merged dataset to understand class-to-window mapping
- Computes inverse frequency weights for each class
- Biases sampling toward windows with rare classes
- Ensures balanced training even with multi-county imbalance

### 4. Updated Training Script
Added `--stratified_sampling` flag to `scripts/train.py`:
```bash
python scripts/train.py \
    --raster_path merged.vrt \
    --labels_path merged_labels.gpkg \
    --stratified_sampling  # NEW FLAG
```

---

## How to Use

### For Single-County Training
No changes needed - works as before.

### For Multi-County Training
1. **Prepare data** (as usual):
   ```python
   from src.utils.prepare_multicounty_training import prepare_multicounty_training

   prepare_multicounty_training(
       county_data_dir="data/",
       selected_counties=["County_A", "County_B", "County_C"],
       out_labels="data/merged_labels.gpkg",
       out_vrt="data/merged_mosaic.vrt",
   )
   ```

2. **Run training with stratified sampling**:
   ```bash
   python scripts/train.py \
       --raster_path data/merged_mosaic.vrt \
       --labels_path data/merged_labels.gpkg \
       --epochs 20 \
       --stratified_sampling \
       --batch_size 2 \
       --max_per_class 500
   ```

3. **Debug potential issues** (NEW):
   ```bash
   python scripts/debug_multicounty.py \
       --raster_path data/merged_mosaic.vrt \
       --labels_path data/merged_labels.gpkg \
       --tile_size 512
   ```

---

## Diagnostic Script

A new debug script (`scripts/debug_multicounty.py`) provides detailed analysis:

```bash
python scripts/debug_multicounty.py \
    --raster_path merged.vrt \
    --labels_path merged_labels.gpkg \
    --tile_size 512
```

Output checks:
- ✓ Class distribution per county
- ✓ Raster properties and band statistics
- ✓ Training window generation and coverage
- ✓ Label-to-window overlap verification
- ✓ Warnings for imbalanced/missing classes

---

## Expected Improvements

With these fixes, multi-county training should:
1. **No longer suffer from variable shadowing bugs** → more stable training
2. **Warn about missing classes** → you can adjust training strategy
3. **Provide balanced class representation** → better generalization
4. **Show diagnostic information** → easier to debug data issues

---

## Testing Recommendation

Compare model performance:

```bash
# Before (multi-county without stratified sampling):
python scripts/train.py --raster_path merged.vrt --labels_path merged.gpkg

# After (with stratified sampling):
python scripts/train.py --raster_path merged.vrt --labels_path merged.gpkg --stratified_sampling
```

The stratified version should show:
- More stable training loss across counties
- Better recall for less-common classes
- Better generalization to new unseen counties

---

## Files Modified

1. **src/utils/tiling.py** - Fixed variable shadowing bug
2. **src/data/dataset.py** - Added class validation warnings
3. **scripts/train.py** - Added stratified sampling support
4. **scripts/debug_multicounty.py** - NEW diagnostic script
5. **src/data/sampling.py** - NEW stratified sampling implementation

---

## Next Steps

If problems persist:
1. Run the debug script to identify data issues
2. Check that all expected classes appear in each county's labels
3. Consider data augmentation for under-represented classes
4. Adjust `--max_per_class` to control sampling ratio
5. Use `--lr 0.0001` (lower learning rate) for more stable multi-county training

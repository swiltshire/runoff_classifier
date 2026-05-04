# Quick Start: Multi-County Debugging

## TL;DR - The Issues and Fixes

| Issue | Root Cause | Impact | Fix |
|-------|-----------|--------|-----|
| **Variable Shadowing** | Loop var `row` overwritten by `row_px` | Silent failures in window generation | ✓ Renamed to `label_row` |
| **Missing Class Warnings** | No validation when classes absent from some counties | Biased/poor predictions | ✓ Added warnings in dataset init |
| **No Balanced Sampling** | Random sampling doesn't account for class/county imbalance | Model overfits to dominant patterns | ✓ Added `StratifiedWeightedSampler` |

## Quick Validation

Before training on multiple counties, validate your data:

```bash
# 1. Validate merged dataset
python scripts/validate_multicounty.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg

# 2. Diagnose potential issues
python scripts/debug_multicounty.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg

# 3. Train with stratified sampling (RECOMMENDED for multi-county)
python scripts/train.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg \
    --stratified_sampling \
    --epochs 20 \
    --batch_size 2
```

## What Changed

### 1. Fixed Bug in `src/utils/tiling.py`
```python
# BEFORE (wrong - variable shadowing):
for _, row in group.iterrows():
    col, row_px = inv * (cx, cy)  # Overwrites 'row'!

# AFTER (correct):
for _, label_row in group.iterrows():
    col, row_px = inv * (cx, cy)  # No shadowing
```

### 2. Added Validation in `src/data/dataset.py`
- Warns if expected classes are missing from labels
- Warns if labels contain unexpected classes
- Helps identify data preparation issues early

### 3. New Stratified Sampling in `src/data/sampling.py`
- `StratifiedWeightedSampler`: weights windows by class rarity
- Ensures balanced class representation during training
- Especially important for multi-county data with class imbalance

### 4. Updated Training Script `scripts/train.py`
- New flag: `--stratified_sampling`
- Automatically imports and uses `StratifiedWeightedSampler`
- Logs when stratified sampling is active

## Expected Improvements

With these fixes, when training on multiple counties you should see:

✓ **No variable shadowing errors** → More stable training windows
✓ **Clear warnings about missing classes** → You can adjust training strategy
✓ **Better balanced learning** → Model sees fair representation of each class
✓ **Better generalization** → Detections work across counties, not just the dominant one

## Debugging Tips

If detections are still poor after applying these fixes:

1. **Check class balance** per county:
   ```bash
   python scripts/debug_multicounty.py --raster_path ... --labels_path ...
   ```
   Look for highly skewed class distributions - may need data augmentation.

2. **Lower learning rate** for multi-county:
   ```bash
   python scripts/train.py --lr 0.0001 --stratified_sampling ...
   ```
   Multi-domain training needs more careful optimization.

3. **Increase training time**:
   ```bash
   python scripts/train.py --epochs 30 --stratified_sampling ...
   ```
   More epochs help model learn county variations.

4. **Check imagery consistency**:
   - Are all counties imaged at same resolution?
   - Same sensor/season?
   - Run: `python scripts/debug_multicounty.py` to check raster properties

5. **Look for geometry issues**:
   ```bash
   python scripts/validate_multicounty.py --raster_path ... --labels_path ...
   ```
   Reports invalid/empty geometries that might affect training.

## Files Modified

- `src/utils/tiling.py` - Fixed variable shadowing
- `src/data/dataset.py` - Added class validation
- `src/data/sampling.py` - NEW: Stratified sampling implementation
- `scripts/train.py` - Added `--stratified_sampling` flag
- `scripts/debug_multicounty.py` - NEW: Diagnostic tool
- `scripts/validate_multicounty.py` - NEW: Pre-training validation
- `MULTICOUNTY_FIXES.md` - Detailed explanation

## Support

For issues:
1. Run `validate_multicounty.py` - identifies data problems
2. Run `debug_multicounty.py` - shows distribution details  
3. Check `MULTICOUNTY_FIXES.md` - detailed technical explanation
4. Use `--stratified_sampling` flag - recommended for multi-county

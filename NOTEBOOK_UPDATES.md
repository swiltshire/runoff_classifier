# Pipeline Updates for Multi-County Training

## Summary

Instead of modifying the large `pipeline.ipynb` file (which caused performance issues), I created a new companion notebook: `notebooks/multicounty_workflow.ipynb`

This new notebook demonstrates the recommended workflow using all the new fixes and features.

## What Changed

### New Notebook: `notebooks/multicounty_workflow.ipynb`

A complete step-by-step guide showing:

1. ✓ **Setup** - Configure paths and data directory
2. ✓ **Prepare** - Merge multi-county data 
3. ✓ **Validate** (NEW) - Run `validate_multicounty.py` to check data before training
4. ✓ **Debug** (NEW) - Run `debug_multicounty.py` to analyze class distribution
5. ✓ **Configure** - Set training hyperparameters with stratified sampling enabled
6. ✓ **Train** (UPDATED) - Training command now includes `--stratified_sampling` flag
7. ✓ **Infer** - Run inference on trained model
8. ✓ **Analyze** - Check results and class distribution

### Why a New Notebook?

- **Cleaner approach:** Doesn't modify the existing large notebook
- **Better documentation:** Shows the complete recommended workflow
- **Easier to maintain:** Can be updated independently
- **Clear migration path:** Shows what's new vs. what's unchanged

### How to Use

#### Option 1: Start Fresh with New Workflow (Recommended)
```
Use: notebooks/multicounty_workflow.ipynb
Benefits: 
  - See all new features in action
  - Includes validation and diagnostics
  - Uses stratified sampling by default
```

#### Option 2: Keep Using Original Pipeline
```
Use: notebooks/pipeline.ipynb
Note: You can manually add:
  1. Before training: run scripts/validate_multicounty.py
  2. Before training: run scripts/debug_multicounty.py  
  3. Add --stratified_sampling flag to train.py command

See multicounty_workflow.ipynb for examples of each step.
```

## Key Differences

### Original Pipeline (pipeline.ipynb)
```bash
# Training command (no stratified sampling)
!python -m torch.distributed.run --nproc_per_node=4 ../scripts/train.py \
    --task instance_seg \
    --raster_path "{vrt_path}" \
    --labels_path "{labels_path}" \
    # ... other flags ...
```

### New Workflow (multicounty_workflow.ipynb)
```bash
# Includes NEW fixes and features
!python -m torch.distributed.run --nproc_per_node=4 scripts/train.py \
    --task instance_seg \
    --raster_path "{vrt_path}" \
    --labels_path "{labels_path}" \
    # ... other flags ...
    --stratified_sampling  # ← NEW FLAG FOR BALANCED LEARNING
```

Plus new steps for validation and diagnostics before training.

## New Notebook Features

### Step 2: Validate Data (NEW)
```python
# Checks:
# - CRS alignment
# - All expected classes present  
# - Geometry validity
# - Spatial overlap
subprocess.run(['python', 'scripts/validate_multicounty.py', ...])
```

### Step 3: Debug Data (NEW)
```python
# Shows:
# - Class distribution per county
# - Training window coverage
# - Label-to-window overlap
# - Raster properties
subprocess.run(['python', 'scripts/debug_multicounty.py', ...])
```

### Step 5: Train with Stratified Sampling (UPDATED)
```bash
# NEW: --stratified_sampling flag
--stratified_sampling \  # ← Better multi-county training!
```

## Migration Guide

If you're using the original `pipeline.ipynb`:

1. **Keep it as is** - No changes needed for single-county training
2. **For multi-county, try the new workflow:**
   - Open: `notebooks/multicounty_workflow.ipynb`
   - Update paths to match your setup
   - Follow the step-by-step guide
3. **Compare results** - See if stratified sampling improves your multi-county detections
4. **Optionally update** - Manually add validation/debug steps to original if preferred

## Files in This Update

- `notebooks/multicounty_workflow.ipynb` (NEW) - Recommended workflow with all fixes
- `update_notebook.py` (can be deleted) - Helper script (not used due to performance)

## Documentation

For more information, see:

- `README_MULTICOUNTY.md` - Quick start
- `MULTICOUNTY_FIXES.md` - Technical details  
- `BEFORE_AFTER.md` - Code comparisons
- `IMPLEMENTATION_SUMMARY.md` - What changed
- `INDEX.md` - Complete index
- `SAGEMAKER_TESTING.md` - Testing guide

## Quick Commands

### Using New Workflow Notebook
```bash
cd ~/SageMaker/runoff_classifier
git checkout multicounty-fixes
# Open: notebooks/multicounty_workflow.ipynb in JupyterLab
# Run cells in order
```

### Using Original Pipeline (with manual additions)
```bash
# Add validation before training
python scripts/validate_multicounty.py --raster_path ... --labels_path ...

# Add debugging
python scripts/debug_multicounty.py --raster_path ... --labels_path ...

# Add to training command:
--stratified_sampling \
```

## Summary

✓ New comprehensive notebook showing all fixes and features
✓ No modifications to existing large notebook (avoids crashes)
✓ Clear step-by-step workflow
✓ Includes validation, debugging, and training with stratified sampling
✓ Fully documented with examples

**Start with `notebooks/multicounty_workflow.ipynb` for the best experience!**

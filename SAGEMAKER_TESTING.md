# Testing in SageMaker - Quick Setup Guide

## Branch Info
- **Branch Name:** `multicounty-fixes`
- **Commit:** 958097f
- **Status:** ✓ Pushed to GitHub

## Quick Setup for SageMaker

### 1. Update Your Local Repo
```bash
cd ~/SageMaker/runoff_classifier  # or your project path
git fetch origin
git checkout multicounty-fixes
git pull
```

### 2. Verify Changes Downloaded
```bash
git log -1 --oneline
# Should show: "fix: multicounty training - variable shadowing, class validation, stratified sampling"

git status
# Should show "On branch multicounty-fixes"
```

### 3. Verify All Files Are Present
```bash
ls -la scripts/debug_multicounty.py
ls -la scripts/validate_multicounty.py
ls -la src/data/sampling.py
ls -la *.md | grep -E "(MULTICOUNTY|README_MULTI|BEFORE_AFTER|IMPLEMENTATION|INDEX)"
```

### 4. Test Python Syntax
```bash
python -m py_compile src/utils/tiling.py src/data/dataset.py src/data/sampling.py scripts/train.py
# Should complete without errors
```

---

## Testing Workflow in SageMaker

### Step 1: Prepare Multi-County Data
```python
# In your notebook
from src.utils.prepare_multicounty_training import prepare_multicounty_training

prepare_multicounty_training(
    county_data_dir="data/",
    selected_counties=["County_A", "County_B", "County_C"],
    out_labels="data/merged_labels.gpkg",
    out_vrt="data/merged_mosaic.vrt",
    verified_only=True,
    debug_plots=False,
)
```

### Step 2: Validate Data
```bash
python scripts/validate_multicounty.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg \
    --classes Bank_Erosion Spillway Culvert_Structure Tile_Inlet Tile_Outlet
```

### Step 3: Diagnose Issues
```bash
python scripts/debug_multicounty.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg \
    --tile_size 512
```

### Step 4: Train with Stratified Sampling (NEW)
```bash
# Single GPU
python scripts/train.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg \
    --stratified_sampling \
    --epochs 20 \
    --batch_size 2 \
    --max_per_class 500 \
    --out_dir outputs/multicounty_test

# Or with torchrun for distributed training (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg \
    --stratified_sampling \
    --epochs 20 \
    --batch_size 2
```

### Step 5: Run Inference
```bash
python scripts/inference.py \
    --model_path outputs/multicounty_test/model_epoch_20.pth \
    --raster_path data/merged_mosaic.vrt \
    --out_dir outputs/multicounty_test/inferences
```

---

## What to Look For During Testing

### ✓ Good Signs (Things Should Work)
- Validation script completes without errors
- Debug script shows balanced class distribution
- Training starts with warning about stratified sampling
- Loss decreases smoothly across epochs
- No variable shadowing errors

### ⚠️ Warnings to Expect (Normal)
```
WARNING: The following classes are not present in labels: {...}
WARNING: Labels contain unexpected classes: {...}
```
These are normal and help identify data issues.

### ❌ Problems to Watch For
- CRS mismatch errors → re-run prepare_multicounty_training
- Geometry errors → check shapefile validity
- Out-of-memory → reduce batch_size or max_per_class
- Loss not decreasing → try lower learning rate (--lr 0.0001)

---

## Comparing with Baseline

### Run Without Fixes (Optional - for comparison)
```bash
# Switch to old branch temporarily
git checkout speedups-infer-train

# Train on same data (note: won't have --stratified_sampling flag)
python scripts/train.py \
    --raster_path data/merged_mosaic.vrt \
    --labels_path data/merged_labels.gpkg \
    --epochs 20 \
    --out_dir outputs/baseline_test

# Switch back to new branch
git checkout multicounty-fixes
```

### Compare Results
- Check inference quality on each branch
- Look for better generalization on new counties
- Compare loss curves and metrics

---

## Documentation in the Branch

All documentation files are included:

| File | Purpose |
|------|---------|
| `README_MULTICOUNTY.md` | Quick start and troubleshooting |
| `INDEX.md` | Complete documentation index |
| `MULTICOUNTY_FIXES.md` | Technical deep dive |
| `IMPLEMENTATION_SUMMARY.md` | What changed and why |
| `BEFORE_AFTER.md` | Side-by-side code comparisons |

Read these in order:
1. `README_MULTICOUNTY.md` - Get oriented
2. `BEFORE_AFTER.md` - See what changed
3. `MULTICOUNTY_FIXES.md` - Understand why

---

## Key Files Modified

**Modified:**
- `src/utils/tiling.py` - Fixed variable shadowing
- `src/data/dataset.py` - Added class validation
- `scripts/train.py` - Added --stratified_sampling flag

**New:**
- `src/data/sampling.py` - StratifiedWeightedSampler
- `scripts/debug_multicounty.py` - Diagnostics
- `scripts/validate_multicounty.py` - Validation

---

## Troubleshooting in SageMaker

### Issue: Module not found errors
```bash
# Ensure you're in the repo directory
cd ~/SageMaker/runoff_classifier

# Check Python path
python -c "import sys; print(sys.path)"

# Test imports
python -c "from src.data.sampling import StratifiedWeightedSampler; print('OK')"
```

### Issue: Git branch issues
```bash
# Show all branches
git branch -a

# Verify you're on the right branch
git status

# If stuck, hard reset to remote
git fetch origin
git reset --hard origin/multicounty-fixes
```

### Issue: Environment issues
```bash
# Reinstall dependencies if needed
pip install -r requirements.txt  # if you have one
# or install packages individually
pip install torch torchvision geopandas rasterio shapely gdal
```

---

## Testing Checklist

- [ ] Branch checked out successfully
- [ ] All new files present
- [ ] Python syntax verified
- [ ] Validation script runs on test data
- [ ] Debug script shows meaningful output
- [ ] Training starts with --stratified_sampling
- [ ] Loss decreases during training
- [ ] Inference runs successfully
- [ ] Compare results with baseline (optional)
- [ ] Document any issues found

---

## After Testing

If everything works:
1. Create a Pull Request on GitHub
2. Document any findings in the PR
3. Merge into main branch
4. Update documentation as needed

If issues found:
1. Open an issue on the branch
2. Include output from debug scripts
3. Describe the problem clearly
4. Include any error messages

---

## Questions or Issues?

Refer to documentation files:
- Quick help: `README_MULTICOUNTY.md`
- Technical details: `MULTICOUNTY_FIXES.md`
- Code changes: `BEFORE_AFTER.md`
- Complete index: `INDEX.md`

Good luck testing! 🚀

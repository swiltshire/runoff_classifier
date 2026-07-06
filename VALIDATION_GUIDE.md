# Integration Guide: Consolidated Validation Script

## Quick Start

After running `prepare_multicounty_training()`, add this cell to your notebook to validate the merged dataset:

```python
# Add validation after prepare_multicounty_training()
from scripts.validate_training_data import validate_training_data

is_valid, errors, warnings, summary = validate_training_data(
    raster_path=out_vrt,                    # Same VRT path from prepare_multicounty_training
    labels_path=out_labels,                 # Same GPKG path from prepare_multicounty_training
    expected_classes=[
        "Bank_Erosion",
        "Spillway", 
        "Culvert_Structure",
        "Tile_Inlet",
        "Tile_Outlet"
    ],
    verbose=True
)

# Check results before training
if not is_valid:
    print("\n❌ VALIDATION FAILED - Cannot proceed with training!")
    print(f"Errors: {errors}")
    print("\nFix data issues and re-run prepare_multicounty_training()")
    sys.exit(1)
    
if warnings:
    print(f"\n⚠️  {len(warnings)} warnings detected - review before training")
    for w in warnings:
        print(f"  - {w}")
else:
    print("\n✓ No warnings - data looks good!")

print(f"\n✓ Validation PASSED")
print(f"  Total labels: {summary['n_labels']}")
print(f"  Spatial coverage: {summary['spatial_coverage']*100:.1f}%")
print(f"  Raster: {summary['raster_shape']}")

# Display per-county summary
if 'county_distribution' in summary:
    print("\n📊 Per-County Class Distribution:")
    for county, dist in sorted(summary['county_distribution'].items()):
        total = sum(dist.values())
        print(f"\n  {county} ({total} labels):")
        for cls in sorted(dist.keys()):
            count = dist[cls]
            pct = 100 * count / total
            print(f"    {cls:20s}: {count:4d} ({pct:5.1f}%)")
```

## What It Validates

1. **Files Exist**: Both raster and labels files present
2. **Raster Valid**: CRS, bounds, resolution readable
3. **Labels Valid**: CRS matches raster, file not empty
4. **Classes Present**: All expected classes found in data
5. **Geometry Valid**: No invalid or empty geometries; within bounds
6. **Per-County Balance**: Flags counties with extreme class imbalance (>95% single class)
   - This is the key check that would have caught the Benton poisoning (100% Tile_Outlet)

## Key Checks for Multi-County Data

The script specifically looks for:
- **Extreme class imbalances** (<5% or >60% of any single class)
- **Suspicious counties** (>95% single class - potential data poison)
- **Spatial coverage** (warns if <1% of raster contains labels)
- **Missing classes** in expected list

## Output

The function returns:
- `is_valid` (bool): Overall pass/fail status
- `errors` (list): Critical issues blocking training
- `warnings` (list): Issues to review before training
- `summary` (dict): Statistics for inspection:
  - `raster_shape`: (height, width)
  - `raster_crs`: Coordinate reference system
  - `raster_bounds`: (minx, miny, maxx, maxy)
  - `n_labels`: Total feature count
  - `spatial_coverage`: Fraction of raster with labels
  - `class_distribution`: Overall class counts/percentages
  - `county_distribution`: Per-county class breakdown
  - `suspicious_counties`: List of counties with potential data issues

## Integration Points

**In notebook (after data prep):**
```python
# Prepare data
prepare_multicounty_training(...)

# Validate
is_valid, errors, warnings, summary = validate_training_data(...)

# If valid, can proceed to training
if is_valid:
    dataloader = build_dataloader(...)
    train(...)
```

**From command line (debugging):**
```bash
python scripts/validate_training_data.py \
  --raster /path/to/multi_county.vrt \
  --labels /path/to/multi_county_labels.gpkg \
  --classes Bank_Erosion Spillway Culvert_Structure Tile_Inlet Tile_Outlet
```

## Interpreting Results

✓ PASS: All checks passed, no warnings → Safe to train
⚠️  WARNINGS: Checks passed but review warnings → Can train but understand risks
❌ FAIL: Errors found → Fix data and re-run prep before training

**Common scenarios:**
- Missing class in expected list → Add to expected list or check data quality
- Suspicious county detected → Investigate county data source (like we did with Benton)
- Low spatial coverage → Make sure labels overlap raster tiles
- Class imbalance warning → Expected for real-world data, but extreme values indicate issues

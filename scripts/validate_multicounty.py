"""
Pre-training validation for multi-county datasets.
Run this after prepare_multicounty_training to ensure data is ready.
"""

import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


def validate_multicounty_dataset(raster_path: str, labels_path: str, expected_classes: list):
    """
    Comprehensive validation of merged multi-county dataset.
    Returns: (is_valid: bool, warnings: list, errors: list)
    """
    warnings = []
    errors = []

    print("\n" + "="*60)
    print("MULTI-COUNTY DATASET VALIDATION")
    print("="*60)

    # -----------------------------------------------------------------------
    # 1. Check raster exists and is valid
    # -----------------------------------------------------------------------
    print("\n[1/6] Checking raster...")
    if not os.path.exists(raster_path):
        errors.append(f"Raster not found: {raster_path}")
        return False, warnings, errors

    try:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            raster_shape = (src.height, src.width)
            raster_bounds = src.bounds
            raster_res = src.res
            print(f"  ✓ Raster: {raster_shape[0]}x{raster_shape[1]} pixels")
            print(f"    CRS: {raster_crs}")
            print(f"    Resolution: {raster_res}")
            print(f"    Bounds: {raster_bounds}")
    except Exception as e:
        errors.append(f"Failed to read raster: {e}")
        return False, warnings, errors

    # -----------------------------------------------------------------------
    # 2. Check labels exist and have valid CRS
    # -----------------------------------------------------------------------
    print("\n[2/6] Checking labels...")
    if not os.path.exists(labels_path):
        errors.append(f"Labels not found: {labels_path}")
        return False, warnings, errors

    try:
        gdf = gpd.read_file(labels_path)
        if gdf.empty:
            errors.append("Labels file is empty!")
            return False, warnings, errors
        if gdf.crs is None:
            errors.append("Labels have no CRS!")
            return False, warnings, errors
        if gdf.crs != raster_crs:
            errors.append(f"CRS mismatch: labels {gdf.crs} vs raster {raster_crs}")
            return False, warnings, errors
        print(f"  ✓ Labels: {len(gdf)} features")
        print(f"    CRS: {gdf.crs}")
    except Exception as e:
        errors.append(f"Failed to read labels: {e}")
        return False, warnings, errors

    # -----------------------------------------------------------------------
    # 3. Check class coverage
    # -----------------------------------------------------------------------
    print("\n[3/6] Checking class coverage...")
    if "Classname" not in gdf.columns:
        errors.append("Labels missing 'Classname' column")
        return False, warnings, errors

    present_classes = set(gdf["Classname"].unique())
    expected_set = set(expected_classes)

    missing = expected_set - present_classes
    extra = present_classes - expected_set

    if missing:
        warnings.append(f"Missing classes: {missing}")
        print(f"  ⚠ Missing classes: {missing}")
    else:
        print(f"  ✓ All expected classes present")

    if extra:
        warnings.append(f"Extra classes not in expected list: {extra}")
        print(f"  ⚠ Unexpected classes: {extra}")

    # Show class distribution
    print("\n  Class distribution:")
    for cls, count in sorted(gdf["Classname"].value_counts().items()):
        pct = 100 * count / len(gdf)
        print(f"    {cls}: {count} ({pct:.1f}%)")

    # -----------------------------------------------------------------------
    # 4. Check per-county class coverage
    # -----------------------------------------------------------------------
    print("\n[4/6] Checking per-county class coverage...")
    if "county" in gdf.columns:
        counties = sorted(gdf["county"].unique())
        print(f"  Counties: {counties}")

        county_classes = {}
        for county in counties:
            county_gdf = gdf[gdf["county"] == county]
            county_classes[county] = set(county_gdf["Classname"].unique())
            missing_in_county = expected_set - county_classes[county]

            if missing_in_county:
                warnings.append(f"{county}: missing {missing_in_county}")
                print(f"  ⚠ {county}: missing {missing_in_county}")
            else:
                print(f"  ✓ {county}: has all classes")

        # Check for highly imbalanced counties
        for county in counties:
            county_count = len(gdf[gdf["county"] == county])
            total_count = len(gdf)
            pct = 100 * county_count / total_count
            if pct < 5:
                warnings.append(f"{county}: only {pct:.1f}% of total labels")
                print(f"  ⚠ {county}: only {pct:.1f}% of labels")
    else:
        print("  ℹ No 'county' column found (single-county dataset?)")

    # -----------------------------------------------------------------------
    # 5. Check spatial overlap
    # -----------------------------------------------------------------------
    print("\n[5/6] Checking spatial overlap...")
    label_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Check if labels fall within raster bounds
    raster_minx, raster_miny, raster_maxx, raster_maxy = raster_bounds
    label_minx, label_miny, label_maxx, label_maxy = label_bounds

    if (label_minx < raster_minx or label_maxx > raster_maxx or
        label_miny < raster_miny or label_maxy > raster_maxy):
        warnings.append("Some labels extend outside raster bounds")
        print(f"  ⚠ Labels extend beyond raster bounds")
    else:
        print(f"  ✓ Labels within raster bounds")

    # Estimate coverage
    coverage_fraction = (label_maxx - label_minx) * (label_maxy - label_miny) / \
                       ((raster_maxx - raster_minx) * (raster_maxy - raster_miny))
    print(f"    Coverage: ~{coverage_fraction*100:.1f}% of raster area")

    if coverage_fraction < 0.01:
        warnings.append("Very low coverage - only <1% of raster has labels")

    # -----------------------------------------------------------------------
    # 6. Check for geometry issues
    # -----------------------------------------------------------------------
    print("\n[6/6] Checking geometry integrity...")
    invalid_geoms = gdf[~gdf.geometry.is_valid]
    if len(invalid_geoms) > 0:
        warnings.append(f"{len(invalid_geoms)} invalid geometries")
        print(f"  ⚠ Found {len(invalid_geoms)} invalid geometries")
    else:
        print(f"  ✓ All geometries valid")

    empty_geoms = gdf[gdf.geometry.is_empty]
    if len(empty_geoms) > 0:
        warnings.append(f"{len(empty_geoms)} empty geometries")
        print(f"  ⚠ Found {len(empty_geoms)} empty geometries")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    is_valid = len(errors) == 0

    if is_valid:
        print("✓ VALIDATION PASSED")
        if warnings:
            print(f"\n⚠ {len(warnings)} warnings:")
            for w in warnings:
                print(f"  - {w}")
    else:
        print("✗ VALIDATION FAILED")
        print(f"\n✗ {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        if warnings:
            print(f"\n⚠ {len(warnings)} warnings:")
            for w in warnings:
                print(f"  - {w}")

    print("="*60 + "\n")
    return is_valid, warnings, errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate multi-county training dataset before training"
    )
    parser.add_argument('--raster_path', type=str, required=True,
                        help='Path to merged VRT or raster')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to merged labels (GPKG)')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=["Bank_Erosion", "Spillway", "Culvert_Structure", "Tile_Inlet", "Tile_Outlet"],
                        help='Expected class names')
    args = parser.parse_args()

    is_valid, warnings, errors = validate_multicounty_dataset(
        args.raster_path,
        args.labels_path,
        args.classes
    )

    # Exit with appropriate code for scripts
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Consolidated post-training-prep validation script.
Validates multi-county dataset before training begins.
Run this after prepare_multicounty_training() to catch data quality issues.
"""

import sys
import os
import argparse

import geopandas as gpd
import rasterio


def _log(msg: str, level: str = "INFO") -> None:
    """Print formatted log message."""
    prefix = f"[{level}]"
    print(f"{prefix} {msg}")


def validate_training_data(
    raster_path: str,
    labels_path: str,
    expected_classes: list,
    verbose: bool = True,
) -> tuple:
    """
    Comprehensive validation of prepared training dataset.
    
    Returns
    -------
    (is_valid, errors, warnings, summary_dict)
    """
    errors = []
    warnings = []
    summary = {}

    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING DATA VALIDATION")
        print("=" * 70)

    # ========================================================================
    # [1/6] CHECK FILES EXIST
    # ========================================================================
    if verbose:
        print("\n[1/6] Checking files...")

    if not os.path.exists(raster_path):
        errors.append(f"Raster not found: {raster_path}")
        if verbose:
            _log(f"Raster not found: {raster_path}", "ERROR")
        return False, errors, warnings, summary

    if not os.path.exists(labels_path):
        errors.append(f"Labels not found: {labels_path}")
        if verbose:
            _log(f"Labels not found: {labels_path}", "ERROR")
        return False, errors, warnings, summary

    if verbose:
        _log("✓ Both files exist", "PASS")

    # ========================================================================
    # [2/6] VALIDATE RASTER
    # ========================================================================
    if verbose:
        print("\n[2/6] Validating raster...")

    try:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            raster_shape = (src.height, src.width)
            raster_bounds = src.bounds
            raster_res = src.res

        if verbose:
            _log(f"Raster: {raster_shape[0]}x{raster_shape[1]} pixels", "PASS")
            _log(f"  CRS: {raster_crs}", "INFO")
            _log(f"  Resolution: {raster_res}", "INFO")

        summary["raster_shape"] = raster_shape
        summary["raster_crs"] = raster_crs
        summary["raster_bounds"] = raster_bounds

    except Exception as e:
        errors.append(f"Failed to read raster: {e}")
        if verbose:
            _log(f"Failed to read raster: {e}", "ERROR")
        return False, errors, warnings, summary

    # ========================================================================
    # [3/6] VALIDATE LABELS
    # ========================================================================
    if verbose:
        print("\n[3/6] Validating labels...")

    try:
        gdf = gpd.read_file(labels_path)

        if gdf.empty:
            errors.append("Labels file is empty!")
            if verbose:
                _log("Labels file is empty!", "ERROR")
            return False, errors, warnings, summary

        if gdf.crs is None:
            errors.append("Labels have no CRS!")
            if verbose:
                _log("Labels have no CRS!", "ERROR")
            return False, errors, warnings, summary

        if gdf.crs != raster_crs:
            errors.append(f"CRS mismatch: labels {gdf.crs} vs raster {raster_crs}")
            if verbose:
                _log(f"CRS mismatch: labels {gdf.crs} vs raster {raster_crs}", "ERROR")
            return False, errors, warnings, summary

        if verbose:
            _log(f"✓ Labels: {len(gdf)} features", "PASS")
            _log(f"  CRS: {gdf.crs}", "INFO")

        summary["n_labels"] = len(gdf)

    except Exception as e:
        errors.append(f"Failed to read labels: {e}")
        if verbose:
            _log(f"Failed to read labels: {e}", "ERROR")
        return False, errors, warnings, summary

    # ========================================================================
    # [4/6] CHECK CLASS PRESENCE & BALANCE
    # ========================================================================
    if verbose:
        print("\n[4/6] Checking class distribution...")

    if "Classname" not in gdf.columns:
        errors.append("Labels missing 'Classname' column")
        if verbose:
            _log("Labels missing 'Classname' column", "ERROR")
        return False, errors, warnings, summary

    present_classes = set(gdf["Classname"].unique())
    expected_set = set(expected_classes)

    missing_classes = expected_set - present_classes
    extra_classes = present_classes - expected_set

    if missing_classes:
        errors.append(f"Missing classes: {missing_classes}")
        if verbose:
            _log(f"Missing classes: {missing_classes}", "ERROR")

    if extra_classes:
        warnings.append(f"Extra classes not in expected list: {extra_classes}")
        if verbose:
            _log(f"Unexpected classes: {extra_classes}", "WARN")

    # Check overall class balance
    class_dist = gdf["Classname"].value_counts()
    total = len(gdf)

    if verbose:
        _log("Overall class distribution:", "INFO")
        for cls in sorted(present_classes):
            count = class_dist.get(cls, 0)
            pct = 100 * count / total
            print(f"  {cls:20s}: {count:6d} ({pct:5.1f}%)")

    # Detect extreme imbalances (potential data poison)
    for cls, count in class_dist.items():
        pct = 100 * count / total
        if pct < 5:
            warnings.append(f"{cls} only {pct:.1f}% - might be under-represented")
            if verbose:
                _log(f"{cls}: only {pct:.1f}% of data", "WARN")
        # Skip Background class check - expected to be high from negative examples
        if pct > 60 and cls != 'Background':
            warnings.append(f"{cls} {pct:.1f}% - might indicate data poison")
            if verbose:
                _log(f"{cls}: {pct:.1f}% - check county distributions", "WARN")

    summary["class_distribution"] = class_dist.to_dict()

    if missing_classes:
        return False, errors, warnings, summary

    # ========================================================================
    # [5/6] CHECK GEOMETRY & SPATIAL VALIDITY
    # ========================================================================
    if verbose:
        print("\n[5/6] Checking geometry and spatial validity...")

    invalid_geoms = gdf[~gdf.geometry.is_valid]
    empty_geoms = gdf[gdf.geometry.is_empty]

    if len(invalid_geoms) > 0:
        warnings.append(f"{len(invalid_geoms)} invalid geometries")
        if verbose:
            _log(f"⚠ Found {len(invalid_geoms)} invalid geometries", "WARN")

    if len(empty_geoms) > 0:
        warnings.append(f"{len(empty_geoms)} empty geometries")
        if verbose:
            _log(f"⚠ Found {len(empty_geoms)} empty geometries", "WARN")

    # Check spatial overlap
    label_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    raster_minx, raster_miny, raster_maxx, raster_maxy = raster_bounds
    label_minx, label_miny, label_maxx, label_maxy = label_bounds

    if (label_minx < raster_minx or label_maxx > raster_maxx or
        label_miny < raster_miny or label_maxy > raster_maxy):
        warnings.append("Some labels extend outside raster bounds")
        if verbose:
            _log("⚠ Some labels extend outside raster bounds", "WARN")

    coverage_frac = ((label_maxx - label_minx) * (label_maxy - label_miny) /
                     ((raster_maxx - raster_minx) * (raster_maxy - raster_miny)))

    if coverage_frac < 0.01:
        warnings.append("Very low spatial coverage (<1% of raster)")
        if verbose:
            _log(f"⚠ Low coverage: {coverage_frac*100:.2f}% of raster", "WARN")

    if verbose:
        _log(f"✓ Geometry valid ({len(gdf)-len(invalid_geoms)-len(empty_geoms)}/{len(gdf)} features)", "PASS")
        _log(f"  Spatial coverage: {coverage_frac*100:.1f}%", "INFO")

    summary["spatial_coverage"] = coverage_frac

    # ========================================================================
    # [6/6] CHECK PER-COUNTY CLASS DISTRIBUTION (catch data poison!)
    # ========================================================================
    if verbose:
        print("\n[6/6] Checking per-county class balance...")

    if "county" in gdf.columns:
        counties = sorted(gdf["county"].unique())
        county_summary = {}
        suspicious_counties = []

        for county in counties:
            county_gdf = gdf[gdf["county"] == county]
            county_dist = county_gdf["Classname"].value_counts()
            county_total = len(county_gdf)

            county_summary[county] = county_dist.to_dict()

            # Check for extreme class imbalance in county (e.g., 100% one class)
            for cls, count in county_dist.items():
                pct = 100 * count / county_total
                if pct >= 95:
                    suspicious_counties.append((county, cls, pct))

        if suspicious_counties:
            if verbose:
                _log("⚠ SUSPICIOUS COUNTIES DETECTED:", "WARN")
            for county, cls, pct in suspicious_counties:
                msg = f"  {county}: {pct:.1f}% {cls}"
                warnings.append(msg)
                if verbose:
                    _log(msg, "WARN")

        summary["county_distribution"] = county_summary
        summary["suspicious_counties"] = suspicious_counties

        if verbose and not suspicious_counties:
            _log(f"✓ County balance looks good ({len(counties)} counties)", "PASS")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    is_valid = len(errors) == 0

    if verbose:
        print("\n" + "=" * 70)
        if is_valid:
            _log("✓✓✓ VALIDATION PASSED ✓✓✓", "PASS")
        else:
            _log("✗✗✗ VALIDATION FAILED ✗✗✗", "ERROR")

        if errors:
            print(f"\n✗ {len(errors)} ERRORS:")
            for e in errors:
                print(f"  - {e}")

        if warnings:
            print(f"\n⚠ {len(warnings)} WARNINGS:")
            for w in warnings:
                print(f"  - {w}")

        print("=" * 70 + "\n")

    return is_valid, errors, warnings, summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate training dataset after prepare_multicounty_training"
    )
    parser.add_argument(
        "--raster",
        type=str,
        required=True,
        help="Path to VRT or merged raster",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to merged labels GPKG",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=[
            "Bank_Erosion",
            "Spillway",
            "Culvert_Structure",
            "Tile_Inlet",
            "Tile_Outlet",
        ],
        help="Expected class names",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    is_valid, errors, warnings, summary = validate_training_data(
        raster_path=args.raster,
        labels_path=args.labels,
        expected_classes=args.classes,
        verbose=args.verbose,
    )

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

"""
Debug script to diagnose multi-county training issues.
Run this after preparing multicounty training data to identify distribution mismatches.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.tiling import make_label_centered_training_windows


def analyze_labels(labels_path: str):
    """Analyze class distribution and spatial coverage."""
    print("\n=== LABEL ANALYSIS ===")
    gdf = gpd.read_file(labels_path)
    print(f"Total features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")

    if "county" in gdf.columns:
        print("\nFeatures per county:")
        print(gdf["county"].value_counts())

    if "Classname" in gdf.columns:
        print("\nClass distribution:")
        class_counts = gdf["Classname"].value_counts()
        print(class_counts)
        print(f"Total classes: {len(class_counts)}")

        # Per-county class distribution
        if "county" in gdf.columns:
            print("\nClass distribution per county:")
            for county in sorted(gdf["county"].unique()):
                county_gdf = gdf[gdf["county"] == county]
                print(f"\n{county}:")
                print(county_gdf["Classname"].value_counts().to_string())


def analyze_raster(raster_path: str):
    """Analyze raster properties."""
    print("\n=== RASTER ANALYSIS ===")
    with rasterio.open(raster_path) as src:
        print(f"CRS: {src.crs}")
        print(f"Shape: {src.height} x {src.width}")
        print(f"Bounds: {src.bounds}")
        print(f"Resolution: {src.res}")
        print(f"Bands: {src.count}")
        print(f"Data type: {src.dtypes}")

        # Read sample pixel statistics
        if src.count >= 3:
            data = src.read([1, 2, 3])
            for band_idx in range(3):
                band_data = data[band_idx]
                print(f"\nBand {band_idx + 1} stats:")
                print(f"  Min: {band_data.min()}, Max: {band_data.max()}")
                print(f"  Mean: {band_data.mean():.2f}, Std: {band_data.std():.2f}")


def analyze_training_windows(raster_path: str, labels_path: str, tile_size: int = 512):
    """Analyze training window generation and label coverage."""
    print("\n=== TRAINING WINDOW ANALYSIS ===")
    windows = make_label_centered_training_windows(
        raster_path=raster_path,
        labels_path=labels_path,
        tile_size=tile_size,
        max_per_class=None,  # include all
        classname_field='Classname',
        jitter=64,
    )

    print(f"Total windows: {len(windows)}")

    # Check for overlaps and coverage
    window_coords = [(w.col_off, w.row_off) for w in windows]
    unique_coords = len(set(window_coords))
    print(f"Unique window positions: {unique_coords}")
    print(f"Duplicate windows (after dedup): {len(windows) - unique_coords}")

    # Spatial extent coverage
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height

    covered_pixels = set()
    for w in windows:
        for col in range(w.col_off, min(w.col_off + w.width, width)):
            for row in range(w.row_off, min(w.row_off + w.height, height)):
                covered_pixels.add((col, row))

    total_pixels = width * height
    coverage = len(covered_pixels) / total_pixels * 100
    print(f"Pixel coverage: {coverage:.1f}% ({len(covered_pixels)}/{total_pixels})")

    # Check for windows outside raster bounds
    out_of_bounds = 0
    for w in windows:
        if (w.col_off + w.width > width) or (w.row_off + w.height > height):
            out_of_bounds += 1
    if out_of_bounds > 0:
        print(f"WARNING: {out_of_bounds} windows exceed raster bounds!")


def check_label_window_overlap(raster_path: str, labels_path: str, tile_size: int = 512):
    """Verify that labels actually exist in the training windows."""
    print("\n=== LABEL-WINDOW OVERLAP CHECK ===")

    with rasterio.open(raster_path) as src:
        transform = src.transform
        raster_crs = src.crs

    gdf = gpd.read_file(labels_path)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    windows = make_label_centered_training_windows(
        raster_path=raster_path,
        labels_path=labels_path,
        tile_size=tile_size,
        max_per_class=None,
        classname_field='Classname',
        jitter=64,
    )

    # For each window, count how many labels intersect it
    labels_per_window = []
    windows_with_no_labels = 0

    for w in windows:
        # Window bounds in world coordinates
        minx, miny = transform * (w.col_off, w.row_off)
        maxx, maxy = transform * (w.col_off + w.width, w.row_off + w.height)
        from shapely.geometry import box
        window_geom = box(min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

        hits = gdf[gdf.geometry.intersects(window_geom)]
        labels_per_window.append(len(hits))
        if len(hits) == 0:
            windows_with_no_labels += 1

    labels_per_window = np.array(labels_per_window)
    print(f"Windows with at least one label: {(labels_per_window > 0).sum()}/{len(windows)}")
    print(f"Windows with NO labels: {windows_with_no_labels}")
    print(f"Labels per window - Mean: {labels_per_window.mean():.2f}, Median: {np.median(labels_per_window):.1f}, Max: {labels_per_window.max()}")

    if windows_with_no_labels > len(windows) * 0.1:
        print(f"WARNING: >{10}% of windows contain no labels! Training may be inefficient.")


def main():
    parser = argparse.ArgumentParser(description="Debug multicounty training data")
    parser.add_argument('--raster_path', type=str, required=True, help='Path to VRT or mosaic raster')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to merged labels (GPKG)')
    parser.add_argument('--tile_size', type=int, default=512, help='Training tile size')
    args = parser.parse_args()

    if not os.path.exists(args.raster_path):
        print(f"ERROR: Raster not found: {args.raster_path}")
        return

    if not os.path.exists(args.labels_path):
        print(f"ERROR: Labels not found: {args.labels_path}")
        return

    analyze_raster(args.raster_path)
    analyze_labels(args.labels_path)
    analyze_training_windows(args.raster_path, args.labels_path, args.tile_size)
    check_label_window_overlap(args.raster_path, args.labels_path, args.tile_size)

    print("\n=== DIAGNOSTICS COMPLETE ===")
    print("Check output above for:")
    print("  - Class imbalance warnings")
    print("  - Low pixel coverage")
    print("  - Windows without labels")
    print("  - CRS mismatches")


if __name__ == "__main__":
    main()

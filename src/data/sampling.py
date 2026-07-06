"""
Stratified sampling utilities for multi-county training.
Ensures balanced class representation across counties.
"""

import numpy as np
from torch.utils.data import Sampler
import torch
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import box
import rasterio


class StratifiedWeightedSampler(Sampler):
    """
    Weighted sampler that ensures each class is represented fairly,
    accounting for multi-county label distribution imbalance.
    """

    def __init__(self, dataset, labels_path: str, classname_field: str = 'Classname', replacement: bool = True):
        """
        Parameters
        ----------
        dataset : ObjectDetectionTilesDataset
            The dataset to sample from
        labels_path : str
            Path to labels file
        classname_field : str
            Column name containing class labels
        replacement : bool
            Whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement

        # Read labels and map each window to its classes
        gdf = gpd.read_file(labels_path)
        window_classes = defaultdict(list)  # window_idx -> list of classes

        with rasterio.open(dataset.raster_path) as src:
            transform = src.transform

        for idx, window in enumerate(dataset.tile_windows):
            # Get world bounds for this window
            minx, miny = transform * (window.col_off, window.row_off)
            maxx, maxy = transform * (window.col_off + window.width, window.row_off + window.height)
            window_geom = box(min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

            # Find intersecting labels
            hits = gdf[gdf.geometry.intersects(window_geom)]
            if len(hits) > 0:
                classes_in_window = hits[classname_field].unique()
                window_classes[idx].extend(classes_in_window)

        # Compute weights: each window gets weight inversely proportional to 
        # the frequency of its represented classes
        self.weights = np.ones(len(dataset))
        class_freq = defaultdict(int)

        # Count class occurrences across windows
        for idx, classes in window_classes.items():
            for cls in set(classes):  # count each class once per window
                class_freq[cls] += 1

        # Assign weights: windows with rare classes get higher weight
        if class_freq:
            max_freq = max(class_freq.values())
            for idx, classes in window_classes.items():
                if classes:
                    # Weight is inverse of average class frequency in this window
                    avg_weight = np.mean([max_freq / class_freq[cls] for cls in set(classes)])
                    self.weights[idx] = avg_weight

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

    def __iter__(self):
        indices = np.random.choice(
            len(self.dataset),
            size=len(self.dataset),
            p=self.weights,
            replace=self.replacement
        )
        return iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)

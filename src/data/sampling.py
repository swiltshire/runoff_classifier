"""
Stratified sampling utilities for multi-county training.
Ensures balanced class representation across counties.
"""

import math

import numpy as np
from torch.utils.data import Sampler
import torch
import torch.distributed as dist
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import box
import rasterio


def _compute_window_weights(dataset, labels_path: str, classname_field: str = 'Classname') -> np.ndarray:
    """
    Compute a per-window sampling weight array, inversely proportional to the
    frequency (in number of distinct windows) of the classes each window contains.
    Windows containing only high-frequency classes are down-weighted; windows
    containing any rare class are up-weighted. Shared by StratifiedWeightedSampler
    and DistributedWeightedSampler so both use identical weighting logic.
    """
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
    weights = np.ones(len(dataset))
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
                weights[idx] = avg_weight

    # Normalize weights
    weights = weights / weights.sum()
    return weights


class StratifiedWeightedSampler(Sampler):
    """
    Weighted sampler that ensures each class is represented fairly,
    accounting for multi-county label distribution imbalance.

    Single-process only. Use DistributedWeightedSampler under DDP.
    """

    def __init__(self, dataset, labels_path: str, classname_field: str = 'Classname', replacement: bool = True, seed: int = 0):
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
        seed : int
            Base seed; combined with the current epoch (via set_epoch) for reproducible
            per-epoch draws.
        """
        self.dataset = dataset
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.weights = _compute_window_weights(dataset, labels_path, classname_field)

    def set_epoch(self, epoch: int) -> None:
        """Called once per epoch by the training loop; reseeds the per-epoch draw."""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(
            len(self.dataset),
            size=len(self.dataset),
            p=self.weights,
            replace=self.replacement
        )
        return iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)


class DistributedWeightedSampler(Sampler):
    """
    DDP-compatible counterpart to StratifiedWeightedSampler. Each rank independently
    draws its own class-frequency-weighted (with replacement) subset of windows per
    epoch, so class rebalancing actually takes effect under torch.distributed training
    (plain torch.utils.data.DistributedSampler performs uniform sampling only).
    """

    def __init__(
        self,
        dataset,
        labels_path: str,
        classname_field: str = 'Classname',
        num_replicas: int = None,
        rank: int = None,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.weights = _compute_window_weights(dataset, labels_path, classname_field)
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        """Called once per epoch by the training loop; reseeds this rank's per-epoch draw."""
        self.epoch = epoch

    def __iter__(self):
        # Fold rank into the seed so each rank draws an independent-but-reproducible
        # weighted sample: valid since sampling is with replacement from a shared target
        # distribution, so no shared-then-shard draw is needed for correctness.
        rng = np.random.default_rng(self.seed + self.epoch * 10_000 + self.rank)
        indices = rng.choice(
            len(self.dataset),
            size=self.num_samples,
            p=self.weights,
            replace=True,
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


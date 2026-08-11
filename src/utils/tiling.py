from __future__ import annotations
from typing import List, Tuple
import random
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd


def make_grid_windows(raster_path: str, tile_size: int, stride: int) -> List[Window]:
    # create a list of sliding windows that cover the raster
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
    windows = []
    for row_off in range(0, height - tile_size + 1, stride):
        for col_off in range(0, width - tile_size + 1, stride):
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
    # add border tiles if the image is not an exact multiple
    if (height - tile_size) % stride != 0:
        row_off = height - tile_size
        for col_off in range(0, width - tile_size + 1, stride):
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
    if (width - tile_size) % stride != 0:
        col_off = width - tile_size
        for row_off in range(0, height - tile_size + 1, stride):
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
    # ensure bottom-right corner included
    windows.append(Window(col_off=width - tile_size, row_off=height - tile_size, width=tile_size, height=tile_size))
    # deduplicate
    unique = {(w.col_off, w.row_off): w for w in windows}
    return list(unique.values())


def _stratified_quota(counts: dict, quota: int) -> dict:
    """Distribute `quota` total items across groups in `counts` (name -> available
    count) as evenly as possible, using a water-filling algorithm: groups with
    fewer available items than their fair share get capped at what they have,
    and the leftover quota is redistributed among the remaining groups. This
    guarantees no single group can consume the entire quota when other groups
    also have items available.
    """
    remaining = dict(counts)
    alloc = {k: 0 for k in counts}
    quota_left = quota
    active = sorted(k for k, v in remaining.items() if v > 0)
    while quota_left > 0 and active:
        share = max(1, quota_left // len(active))
        for k in list(active):
            if quota_left <= 0:
                break
            take = min(share, remaining[k], quota_left)
            alloc[k] += take
            remaining[k] -= take
            quota_left -= take
            if remaining[k] <= 0:
                active.remove(k)
    return alloc


def make_label_centered_training_windows(raster_path: str,
                                         labels_path: str,
                                         tile_size: int,
                                         max_per_class: int | None = None,
                                         classname_field: str = 'Classname',
                                         jitter: int = 64,
                                         county_field: str = 'county',
                                         verbose: bool = True) -> List[Window]:
    # create training windows centered (with jitter) around label centroids
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        transform = src.transform
    gdf = gpd.read_file(labels_path)
    if gdf.crs is None:
        raise ValueError('labels must have a valid crs')
    # reproject if needed
    # note: we assume labels already match raster crs in the dataset class

    windows = []
    grouped = gdf.groupby(classname_field)
    for cls, group in grouped:
        if max_per_class is not None and len(group) > max_per_class:
            if county_field in group.columns:
                # Stratify the cap across counties so a single county's rows
                # (which are concatenated in county-block order upstream) can't
                # silently absorb the whole cap and starve out other counties'
                # geographic/imagery diversity for this class.
                counts = group[county_field].value_counts().to_dict()
                alloc = _stratified_quota(counts, max_per_class)
                parts = []
                for county, n_take in alloc.items():
                    if n_take <= 0:
                        continue
                    county_rows = group[group[county_field] == county]
                    if n_take >= len(county_rows):
                        parts.append(county_rows)
                    else:
                        parts.append(county_rows.sample(n=n_take))
                selected = pd.concat(parts) if parts else group.iloc[0:0]
                if verbose:
                    print(f"[tiling] {cls}: capped {len(group)} -> {max_per_class} "
                          f"({len(counts)} counties, per-county alloc: {alloc})")
            else:
                selected = group.sample(n=max_per_class)
        else:
            selected = group

        for _, label_row in selected.iterrows():
            cx, cy = label_row.geometry.centroid.x, label_row.geometry.centroid.y
            inv = ~transform
            col, row_px = inv * (cx, cy)
            # apply random jitter in pixel space
            jx = random.randint(-jitter, jitter)
            jy = random.randint(-jitter, jitter)
            col = int(col + jx)
            row_px = int(row_px + jy)
            # compute top-left corner for a tile centered on (col,row_px)
            col_off = max(0, min(col - tile_size // 2, width - tile_size))
            row_off = max(0, min(row_px - tile_size // 2, height - tile_size))
            windows.append(Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size))
    # deduplicate windows while preserving list
    unique = {(w.col_off, w.row_off): w for w in windows}
    return list(unique.values())


def adjust_boxes_to_global(boxes, window):
    # convert window-local pixel boxes to global pixel boxes
    if boxes.numel() == 0:
        return boxes
    offset = boxes.new_tensor([window.col_off, window.row_off, window.col_off, window.row_off])
    return boxes + offset

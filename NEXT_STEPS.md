# Postmortem: Tile_Outlet detection volume regression (Benton County) — RESOLVED

**Status: RESOLVED and validated end-to-end on SageMaker (2026-08-11).**

## The problem
A past ("fluke") training/inference run detected **914 raw Tile_Outlet detections** in
Benton County (552 manually confirmed correct, ~80%/80% precision/recall above score 0.98).
Every run since then topped out around **~240 raw detections**, despite deliberate attempts
to reproduce the fluke's training conditions (county set, class balance, thresholds).

## Root cause
`scripts/inference.py`'s AOI (NHD waterway mask) filtering used a hardcoded
`MASK_DOWNSAMPLE` constant that was changed from `1` (fluke era) to `16` (commit `ee32002`,
2026-07-07) purely to speed up mask rasterization/loading. This constant controlled the
resolution of the rasterized AOI mask used to keep/reject both candidate inference windows
and final detection boxes:

```python
mr0 = (r0 - row0) // downsample
mr1 = (r1 - row0) // downsample
...
if mr1 <= mr0 or mc1 <= mc0:
    continue   # detection silently, permanently dropped as "outside AOI"
```

Any detection box smaller than `downsample` pixels in either dimension collapses to a
zero-size chip after integer division and is rejected **regardless of true location**.
Tile_Outlet's median bounding-box diagonal is ~12.5px (see
`/memories/repo/codebase_notes.md`) — at `downsample=16` (~8ft at 6-inch imagery), the large
majority of true-positive Tile_Outlet detections were silently discarded. Larger classes
(Culvert_Structure ~400px, Bank_Erosion, Spillway) were far less affected, which is why this
selectively collapsed Tile_Outlet recall specifically. This single bug, confirmed via direct
diff against the fluke-era commit, fully explains the 914-vs-240 detection volume gap —
not training data composition, county selection, or model architecture (all of which were
investigated and ruled out as primary causes along the way).

## Fix (final, shipped state — commit `48797b9`, branch `fix/mosaic-then-reproject`)
Two vector/shapely-based AOI overlap approaches were tried and abandoned first: an exact
per-box `shapely` intersection test, then a vectorized bulk version. Both were correct but
too slow against the real AOI shapefile (NHD statewide waterway buffers — far more
geometrically complex than any synthetic benchmark), stalling real inference runs even at a
few hundred boxes.

The shipped fix instead reverted to a pure raster-array-lookup design (like the original
fluke-era code), fixing only the actual bug (the downsample factor) rather than the
filtering mechanism:
- **`src/utils/fast_mask.py`**: `filter_boxes_by_mask_raster()` — per-box AOI coverage test
  via plain numpy array slicing against the rasterized mask, mirroring
  `filter_windows_by_mask_raster()`'s existing logic. No shapely/geopandas in the hot path,
  so performance is insensitive to real-world AOI polygon complexity.
- **`scripts/inference.py`**: `MASK_DOWNSAMPLE = 1` (single value, used for a single shared
  mask for both window-level and box-level filtering — window-level cost is bounded by
  window pixel count, not mask size, so full resolution costs nothing meaningful there).
- **Persistent per-county mask caching bug fixed**: `resolve_raster_input()` was calling
  `write_mosaic_vrt()` + `clear_mask_cache()` unconditionally on every single run, clearing
  the mask cache before it could ever be reused. Now skips both when the VRT already exists.

**Validated**: re-run end-to-end on SageMaker — fast (no stalls) and confirmed to fully
recover detection volume. Both the small-object AOI-filtering bug and the broader
914-vs-240 recall gap are resolved by this fix.

## Key files
- `src/utils/fast_mask.py` — `get_mask_clipped()`, `filter_windows_by_mask_raster()`,
  `filter_boxes_by_mask_raster()`, `clear_mask_cache()`.
- `scripts/inference.py` — `MASK_DOWNSAMPLE` constant, AOI mask setup in `main()`,
  `resolve_raster_input()`.

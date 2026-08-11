# Investigation: recovering fluke-level Tile_Outlet detection volume (Benton County)

Status as of 2026-08-06, updated with **git-archaeology-confirmed findings** (live investigation
of the actual fluke-era commit, dated 6/29/2026). These are no longer speculation — they are
verified against real diffed code and real captured notebook execution output preserved in git
history. **No code changes have been made yet — investigation is still ongoing.**

## The problem
A past ("fluke") training run detected **914 raw Tile_Outlet detections** in Benton County,
of which **552 were manually confirmed correct** (~60% overall precision). Above a score
threshold of 0.98, the model had ~80% precision and ~80% recall. Since then, no run has
exceeded **~240 raw detections** in Benton, despite deliberate attempts to reproduce the
fluke's conditions. Goal: **increase detection volume/recall, precision is secondary** (more
false positives are acceptable if it recovers volume).

## Root cause of the original fluke (known, not in question)
The fluke run's Benton County training labels were accidentally filtered to
`Classname == 'Tile_Outlet' AND VerifiedTr == 1` only (a data-prep bug), while several other
training counties were included with **all 5 feature classes** present, unfiltered. The run
was technically multi-class (not single-class), but Tile_Outlet ended up a large plurality of
the training set due to this Benton bug. This data-poisoning bug is well understood — the open
question is *why current "correctly" filtered/configured runs can't reach anywhere near the
same detection volume*, not whether the bug happened.

## Artifacts are lost — must proceed by experiment, not archaeology
- No saved logs/config for the fluke run exist.
- `outputs/model_epoch_*.pth` and `outputs/model_final.pth` (repo root) are orphaned
  checkpoints from an unrelated/earlier run (predate the current per-mode output-dir
  convention: `outputs/train_multicounty_{FOCUS_CLASS}/` for single-class mode,
  `outputs/train_multicounty/` for multi-class). `outputs/train_multicounty/` is currently
  empty and no `train_multicounty_Tile_Outlet/` dir exists yet.
- `outputs/inference_summary.json` on disk is from an unrelated small (28-detection) run, not
  the fluke or the 240-detection run — no historical threshold record recoverable there either.
- ~~Fluke run's exact training-county set is *unconfirmed*~~ **SUPERSEDED — now confirmed via
  git archaeology, see below: all 16 available counties were selected, Benton first.**
- **Conclusion: reconstruct via forward controlled experiments, comparing raw detection counts
  under matched inference settings, not by trying to recover old configs.** (Still true in
  spirit, but git archaeology has now recovered much more ground truth than expected — see new
  section below.)

## CONFIRMED via git archaeology (fluke-era commit `65a6e38`, parent `09f5474`/`23529c7`, 6/29/2026)

Git history preserved the *actual executed notebook outputs* (not just code), which served as a
receipt of exactly what ran. Four concrete, code-verified facts recovered:

1. **Exact county list/order, recovered from real captured stdout in the notebook's git
   history**: 16 counties were selected (via `sorted(...)`, i.e. alphabetical), not a small
   3-county set as previously believed:
   `['Benton', 'Boone', 'Brown', 'Carroll', 'Cass', 'Hamilton', 'Howard', 'Jackson', 'Johnson',
   'Miami', 'Montgomery', 'Newton', 'Tipton', 'Wabash', 'Warren', 'White']`. **Benton is first.**

2. **`tiling.py`'s per-class window cap at that commit had no stratification** (the current
   `_stratified_quota()` water-filling fix did not exist yet):
   ```python
   for cls, group in grouped:
       count = 0
       for _, label_row in group.iterrows():   # row order == county concatenation order
           ...
           count += 1
           if max_per_class is not None and count >= max_per_class:
               break
   ```
   Labels were concatenated via `pd.concat(gdfs)` in `selected_counties` order
   (confirmed in `prepare_multicounty_training.py` at that commit: `for c in selected_counties:
   ... gdfs.append(gdf)`). The fluke run's training config had **`max_per_class: 500`**
   (confirmed from the notebook's actual training-config cell and the literal `train.py`
   invocation: `--max_per_class 500 --stratified_sampling --tile_size 512 --epochs 20
   --batch_size 2 --grad_accum 1 --lr 0.0002`). Since Benton is first in concatenation order and
   had an abundant (poisoned, `VerifiedTr==1`-only-filtered) supply of Tile_Outlet rows, **Benton
   almost certainly consumed the entire 500-row cap by itself**, meaning the effective
   Tile_Outlet training signal was ~100% Benton-derived despite 16 counties being nominally
   selected.

3. **The county-exclusion ("poison_counties") check operated at whole-county granularity, not
   per-label**: a county was only dropped if `gdf.empty` after the `VerifiedTr==1` filter, or if
   *zero* of its labels overlapped *any* available tile ("ERROR: {c} labels do not overlap any
   tiles"). A county with **majority-missing imagery tiles but at least one overlapping label**
   passed through untouched — its other labels (sitting over missing/NoData tile regions) still
   proceeded to window generation. Compounding this, `dataset.py` at that commit had **no
   NoData/blank-image validation** — windows generated over missing-imagery regions were fed
   directly to the model as ordinary "confirmed positive" training examples, with no filtering
   for blank/near-blank raster content.

4. **Imagery-year selection had no per-county pinning at that time.** `indiana_cogs.py`
   auto-selected `# find newest year having the county` — the newest imagery year available from
   the remote service per county — with no mechanism to match the year against when the
   county's labels were actually digitized. This is the confirmed root cause of the
   imagery-year/label mismatch issue, and is exactly what `data/training_county_imagery_years.csv`
   (a later, current-era addition) was built to fix by pinning an explicit year per county.

**Net effect**: the fluke run was nominally a 16-county, 5-class multi-class run, but due to
facts #2-3 above, the *effective* Tile_Outlet supervision signal was overwhelmingly
Benton-derived (and poisoned/uncertain in class purity), while still getting the visual-diversity
benefit of other counties' other classes (relevant to hypothesis #2 below). This is now the
leading, code-confirmed explanation — not speculation.

**Open question**: whether Benton's actual fluke-era Tile_Outlet row count exceeded 500 (full
cap monopolization) or merely a large fraction of it (partial monopolization). No artifact from
that era has yet been found to confirm the exact row count — worth checking if
`validate_training_data.py` was ever run and logged during that period.

## CONFIRMED via git archaeology round 2 — INFERENCE-SIDE BUG (likely the dominant cause, 2026-08-11)

Direct diff of `src/models/model.py`, `scripts/train.py`, and `scripts/inference.py` between the
fluke commit (`65a6e38`) and current HEAD (`dd63b2d`, branch `fix/mosaic-then-reproject`):

- **`model.py`**: functionally IDENTICAL. Same stock-pretrained `fasterrcnn_resnet50_fpn`/
  `maskrcnn_resnet50_fpn`, same predictor-head swap. A custom small-object anchor generator was
  tried in between (`b266a60`) and reverted (`b13e27f`, "was causing high/noisy epoch-1 loss")
  — net zero difference today vs. fluke.
- **`train.py`**: only diff is `jitter=64` (fluke, hardcoded) vs. `jitter=max(1, args.tile_size
  // 8)` (current) — identical value (64) since `tile_size=512` unchanged in both eras. No
  functional difference.
- **`inference.py`**: CLI args/defaults are 100% identical (`score_thresh`, `nms_iou_thresh`,
  `final_box_iou`, `tile_size`, `stride`, `min_cover_frac=0.0` default, all unchanged), and the
  notebook passes the same overrides in both eras (`tile_size=512`, `stride=256`,
  `nms_iou_thresh=0.3`, `final_box_iou=0.3`, `min_cover_frac=0.01`, same
  `mask_path=../data/NHDmask_Indiana/NHDfinalMaskIndiana.shp`).

  **BUT: the hardcoded `MASK_DOWNSAMPLE` constant changed from `1` (fluke) to `16` (current,
  commit `ee32002` "fix mask downsampling: change from 1x to 16x", 2026-07-07, done purely for
  loading-speed reasons).** This constant controls the resolution at which the NHD-waterway AOI
  mask is rasterized and used to filter BOTH candidate inference windows AND final detections
  (two separate filter blocks in `inference.py`, identical logic, both present unchanged since
  fluke era):
  ```python
  mrmin = (rmin - row0) // ds   # ds = MASK_DOWNSAMPLE
  mrmax = (rmax - row0) // ds
  mcmin = (cmin - col0) // ds
  mcmax = (cmax - col0) // ds
  if mrmax <= mrmin or mcmax <= mcmin:
      continue   # detection silently dropped as "outside AOI"
  ```
  Any detection box with full-resolution height or width **smaller than `ds` pixels** collapses
  to zero span after integer division and is **automatically rejected, independent of true
  location relative to the AOI polygon.** At `ds=1` (fluke) this never spuriously triggers. At
  `ds=16` (≈8ft at 6-inch imagery), **any box under 16px in either dimension is guaranteed
  rejected.** Tile_Outlet's median bounding-box diagonal is ~12.5px (per earlier pixel-size
  inspection cell) — meaning a large fraction of true-positive Tile_Outlet detections are
  silently, deterministically discarded today regardless of model confidence or actual AOI
  membership. Larger classes (Culvert_Structure ~400px diagonal, Bank_Erosion, Spillway) are far
  less affected since their boxes comfortably exceed 16px — this selectively explains why
  Tile_Outlet recall specifically collapsed while larger-object classes may not show the same
  degree of regression.

  This is now the **leading, most mechanistically direct explanation** for "recall was multiple
  times higher" in the fluke run — a deterministic geometric inference-time filtering bug, not a
  training-data or model-architecture effect. It is independent of (and likely compounds with)
  the training-side county-monopolization hypothesis (#0) documented above.

  **FIXED on 2026-08-11 (branch `fix/mosaic-then-reproject`, confirmed newest/current branch,
  already contains `feature/single-class-training`).** Implemented: added `load_aoi_gdf()` and
  `aoi_cover_fraction()` to `src/utils/fast_mask.py` — an exact, non-rasterized AOI overlap test
  using true shapely geometric intersection (with an `sindex` spatial-index prefilter for speed)
  instead of the downsampled raster-mask lookup. In `scripts/inference.py`:
  - `MASK_DOWNSAMPLE=16` and `filter_windows_by_mask_raster()` are UNCHANGED and still used for
    the window-level pre-filter (windows are 512px tiles — far larger than the 16px cell size,
    so no precision loss there, and this is where the real perf win from `ee32002` lives).
  - The "fast box-level mask enforcement (pre-vectorization)" block was rewritten to convert each
    surviving (post per-rank-NMS) box to world coordinates and test it with `aoi_cover_fraction()`
    against a once-per-rank-loaded `aoi_gdf`, instead of indexing into the coarse `mask_raster`
    array. This eliminates the integer-division collapse-to-zero bug entirely (exact test, no
    minimum object size).
  - The "fast raster mask keep-only at polygon level" block (final, post-merge filter) was
    REMOVED entirely — since the exact filter above now runs before vectorization, per-rank
    output files are already exactly AOI-filtered, making the second raster-based filter
    redundant. This also simplifies the code (one filter path instead of two).
  - Note investigated during implementation: this box-level pre-filter's rejections were found
    to be **permanent/irrecoverable** (boxes are `index_select`'d out immediately, never
    reconsidered later) — contradicting the initial assumption that it was a "safe, tolerable"
    coarse pre-filter with a safety-net second check. This is why the fix was applied here
    (not just at the final filter) and why the redundant final filter was removed rather than
    also patched.
  - Validated with a synthetic unit-style test (tiny AOI square + tiny boxes fully inside, fully
    outside, and straddling the edge) confirming `aoi_cover_fraction()` returns exact fractional
    overlap (1.0 / 0.0 / 0.5 respectively) regardless of box size — unlike the old ds=16 raster
    lookup which would have rejected all of these (since 12-unit boxes < 16-unit downsample
    cell).
  - `py_compile` clean on both modified files; no lint/type errors via `get_errors`.
  - **Not yet done**: no SageMaker training/inference run has been executed yet to empirically
    confirm this recovers detection volume — next step is to re-run inference with this fix on
    Benton County and compare raw detection counts against the current ~240 baseline.

## Experiments already run (ground truth for what's been tried)

| # | Config | Counties | Mode | score_thresh | nms_iou_thresh/final_box_iou | Result |
|---|--------|----------|------|--------------|------------------------------|--------|
| 0 (fluke) | Benton poisoned to VerifiedTr=1 Tile_Outlet only; other counties full 5-class | believed Benton+Boone+Brown (unconfirmed) | multi-class (accidentally) | unknown | **confirmed 0.3 the entire time, including this run** | 914 raw / 552 confirmed correct; ~80%/80% P&R @ score≥0.98 |
| 1 | `INCLUDE_BACKGROUND_CLASS=False` (positives-only: only confirmed VerifiedTr=1 Tile_Outlet, no negatives at all) | Benton, Boone, Brown | single-class | 0.5 | 0.3 | **~240 raw detections** |
| 2 | Same positives-only config as #1 | ALL training counties | single-class | 0.5 | 0.3 | **fewer than 240** (exact number not yet recorded) |

Both experiments 1 and 2 are RAW detection counts (pre-manual-review), directly comparable to
the fluke's 914 raw figure.

## Hypotheses RULED OUT (tested or logically excluded)

1. **Hard-negative mining as primary cause** — RULED OUT. The original theory was that today's
   single-class recipe's tunable negative pool (`POSITIVE_RATIO`/`IN_CLASS_RATIO`, mixing in
   `VerifiedTr=0` "hard negative" lookalikes) teaches the model to suppress borderline calls
   that the fluke model happily called positive. But experiment #1 used
   `INCLUDE_BACKGROUND_CLASS=False` — a **positives-only** run with zero negatives of any kind
   — and still only reached 240. Since removing ALL explicit negatives didn't close the gap,
   hard-negative mining alone cannot be the main explanation.
2. **NMS/`final_box_iou` settings** — RULED OUT. Confirmed `nms_iou_thresh`/
   `final_box_iou = 0.3` was used consistently the entire time, including in the fluke run —
   it was never a variable that changed between runs, so it cannot explain the 914-vs-240 gap.
   (Mechanically: NMS only suppresses same-class detections whose boxes overlap ≥ the IoU
   threshold — it can't be suppressing well-separated, distinct real-world detections; it
   mainly cleans up duplicate detections of the same object from overlapping inference tiles.)
3. **Training-window/volume cap (`max_per_class`) as a limiter for experiments 1/2 specifically**
   — still RULED OUT for those two runs. `train.py`/notebook training-config cell sets
   `max_per_class = 100_000` (effectively uncapped) whenever `SINGLE_CLASS_MODE=True` — so raw
   Tile_Outlet training-window *volume* should be higher in experiments 1/2 than in the fluke
   run, not lower. **However**, git archaeology (see CONFIRMED section above) shows the cap
   mechanism was critical to explaining the *fluke run itself*: at `max_per_class=500` combined
   with the pre-fix row-order cap logic and Benton-first concatenation order, the cap wasn't
   just "a volume limiter" — it was a *county-monopolization* mechanism that concentrated
   (rather than diluted) the Tile_Outlet training signal onto Benton. This is a distinct effect
   from raw volume and is now a leading (not ruled out) hypothesis — see new hypothesis #0 below.

## Hypotheses NOT YET RULED OUT — leading candidates

0. **County-order cap-monopolization (NEW top candidate, git-confirmed mechanism, untested as a
   reproduction experiment).** As detailed in the CONFIRMED section above: the fluke-era
   `tiling.py` cap logic + `max_per_class=500` + Benton-first alphabetical county order meant
   Benton's poisoned Tile_Outlet rows almost certainly consumed the *entire* cap, concentrating
   ~100% of Tile_Outlet training signal onto Benton despite 16 counties being nominally
   selected — while still benefiting from other counties' other-class visual diversity (a
   multi-class run, not single-class). Current `tiling.py` already replaced this with a fair
   `_stratified_quota()` water-filling split, which is more "correct" but structurally
   *prevents* this concentration effect from ever recurring, even if all 16 counties were
   reselected today. This is now the single most direct, evidence-backed candidate mechanism —
   not yet tested as a reproduction experiment (see Phase B, step 2a below).
1. **Geographic/domain-shift dilution (top candidate).** Confirmed empirically: experiment #2
   (all counties) produced *fewer* Benton detections than experiment #1 (3 counties), with
   everything else held constant. In single-class positives-only mode, adding more counties
   does NOT dilute the Tile_Outlet class itself (all confirmed Tile_Outlet from every selected
   county are still included) — but it does dilute the model's visual *specialization* to
   Benton's specific imagery characteristics (soil color, crop patterns, imagery
   vintage/sensor). Training capacity gets spent generalizing across diverse counties instead
   of "overfitting" to Benton's specific look, which is plausibly exactly what gave the fluke
   run (small, ~3-county set) its outsized raw recall on Benton specifically.
2. **Single-class vs. multi-class training context (top candidate, fully untested).** The
   fluke run was technically multi-class (5 classes + implicit background) — Boone/Brown
   contributed ALL their classes normally (not just Tile_Outlet), giving the shared
   backbone/RPN/FPN much richer, more diverse visual training signal (varied real-world
   scenery, object shapes, implicit negatives from everywhere not boxed) than today's
   single-class-positives-only recipe, which shows the model *only* narrow, label-centered
   Tile_Outlet crops — zero other visual content, even from Boone/Brown. Multi-task/multi-class
   training is a known implicit regularizer in object detection; it's plausible this training
   *breadth*, not the Benton-poisoning artifact per se, is what let the fluke model achieve
   such high raw recall. **This has never been isolated as a variable** — no run has tried
   correctly-filtered multi-class training (i.e., multi-class mode, but without the Benton
   VerifiedTr=1-only bug) on a small/matched county set.
3. **Model-level torchvision postprocessing defaults never exposed (secondary/lower
   confidence).** `src/models/model.py`'s `build_fasterrcnn_model()`/`build_maskrcnn_model()`
   only swap the box_predictor head to match `num_classes` — they never override
   `model.roi_heads.score_thresh` (torchvision default 0.05), `nms_thresh` (default 0.5), or
   `detections_per_img` (default 100). With 512px inference tiles at 50% overlap (stride 256),
   a Benton tile unusually dense with tiny (~12.5px median diagonal) tile outlets could
   silently hit the internal 100-detections-per-image cap *before* `scripts/inference.py`'s own
   `--score_thresh`/`--nms_iou_thresh` filtering (lines ~342, ~407-411) ever runs. Plausible but
   lower-probability given typical tile density; cheap to test by exposing these as args.
4. **Checkpoint/epoch selection (secondary, untested).** Only `model_final.pth` (final epoch)
   has been used for inference in recent runs. Earlier-epoch checkpoints are often
   higher-recall/lower-precision before a model fully converges — an easy, zero-retrain lever
   once a fresh single-class or multi-class run produces per-epoch checkpoints (current
   `train_multicounty_Tile_Outlet/` dir doesn't exist yet; needs experiments below to run
   first).

## Recommended next steps (priority order)

### Phase A — Cheap, no retrain
1. Sweep `score_thresh` only (e.g. 0.1, 0.2, 0.3, 0.5, 0.7, 0.9) on the existing
   experiment-#1 checkpoint (3-county, positives-only) to characterize its confidence
   distribution / build a threshold-vs-count curve. This doesn't require retraining and helps
   contextualize whatever number Phase B produces.

### Phase B — Retraining experiments on SageMaker (highest priority, targets the leading hypotheses)
2a. **Direct reproduction test of hypothesis #0 (single most informative untested run, do this
    first).** Train multi-class mode on the exact confirmed fluke-era 16-county list (Benton
    first: `Benton, Boone, Brown, Carroll, Cass, Hamilton, Howard, Jackson, Johnson, Miami,
    Montgomery, Newton, Tipton, Wabash, Warren, White`), keeping everything else on the
    *current, fixed* pipeline (correct per-county imagery years via
    `training_county_imagery_years.csv`, complete tiles, no Benton `VerifiedTr` poisoning, and
    the current fair `_stratified_quota()` cap logic — i.e. do NOT reintroduce any bugs). If
    this run's Benton detection count is still far below 914, it demonstrates that the cap
    *fairness* fix (not just the poisoning/imagery bugs) is itself a major reason detection
    volume dropped — i.e. today's "more correct" pipeline structurally prevents the
    concentration effect that produced the fluke's volume. This would directly justify
    designing an intentional, controlled "priority county" concentration knob (as opposed to
    accepting the fair split) as the actual path to recovering detection volume. Only after
    this result should any code change to `tiling.py`/`prepare_multicounty_training.py` be
    considered.
2. **County-count sweep**: repeat the positives-only single-class recipe
   (`INCLUDE_BACKGROUND_CLASS=False`, `score_thresh=0.5`, `nms_iou_thresh=0.3`) at multiple
   county-set sizes — Benton-only, Benton+Boone+Brown (re-confirm ~240 baseline), a ~6-county
   set, all counties (re-confirm experiment #2's "fewer than 240") — to build a detection-count
   vs. county-count curve and quantify the dilution effect. Benton-only and all-counties runs
   are independent and can run in parallel.
3. **Multi-class isolation experiment (single most informative untested run)**: train with
   `SINGLE_CLASS_MODE=False` (full 5-class, CORRECTLY filtered — i.e. do not reintroduce the
   Benton VerifiedTr=1-only bug) restricted to the same 3 counties as experiment #1 (Benton,
   Boone, Brown). Compare Benton raw detection count against both 914 (fluke) and 240
   (experiment #1). If this run's count jumps well above 240, it confirms multi-class training
   breadth (not the historical data-poisoning bug) is the major recall driver, and the path
   forward is multi-class training on a small/focused county set rather than single-class
   positives-only mode. Run in parallel with step 2.
4. If step 3 confirms the hypothesis, follow up combining both levers: multi-class mode + a
   small/focused county set (Benton, Boone, Brown, or even Benton-only) + `score_thresh≈0.5` —
   the closest legitimate structural reproduction of the fluke's conditions, minus the actual
   labeling bug.
5. For every checkpoint from steps 2-4, repeat the Phase A `score_thresh` sweep on Benton and
   log raw detection counts (+ precision, if confirmed/incorrect ground truth is available to
   score against) next to the 914/552 fluke baseline.

### Phase C — Small code change, secondary lever
6. Modify `build_fasterrcnn_model()`/`build_maskrcnn_model()` in `src/models/model.py` to
   accept `box_score_thresh`, `box_nms_thresh`, `box_detections_per_img` params and set
   `model.roi_heads.score_thresh`/`nms_thresh`/`detections_per_img` explicitly (instead of
   leaving torchvision's 0.05/0.5/100 defaults baked in). Thread through as new
   `scripts/inference.py` CLI args. This removes the invisible 100-box-per-tile cap as a
   possible confound, especially relevant for Benton's outlet-dense tiles.
7. Re-run a subset of Phase A/B sweeps with `box_detections_per_img` raised (e.g. 300-500) to
   rule in/out tile-density truncation.

### Phase D — Only if A-C don't recover volume
8. Epoch-checkpoint sweep on whichever Phase B run is most promising (compare
   `model_epoch_5/10/15/final.pth` at fixed thresholds).
9. If county dilution is confirmed but multi-class context (step 3) doesn't fully close the
   gap, consider county-weighted sampling (deliberately oversample Benton-style imagery) rather
   than uniform county representation — reintroducing the fluke's implicit Benton-specialization
   without abandoning multi-county training entirely.

## Key file/line references
- `notebooks/pipeline.ipynb`:
  - Cell `#VSC-63dd8650` — `SINGLE_CLASS_MODE`, `FOCUS_CLASS`, `POSITIVE_RATIO`,
    `IN_CLASS_RATIO`, `INCLUDE_BACKGROUND_CLASS` toggles.
  - Cell `#VSC-c3988cf8` — training hyperparameters incl. `max_per_class` (100_000 in
    single-class mode vs 500 multi-class), `tile_size=512`, `epochs=20`.
  - Inference config cell (~raw json line 1245-1254) — `tile_size=512`, `stride=256`,
    `score_thresh`, `nms_iou_thresh=0.3`, `final_box_iou=0.3`.
- `src/utils/prepare_multicounty_training.py` — `_filter_singleclass_labels()` (~lines
  160-358): implements `positive_ratio`/`in_class_ratio`/`include_background` logic; main
  entry `prepare_multicounty_training()` (~lines 413-550) does the county merge/VerifiedTr
  filter (~469-480).
- `src/models/model.py` — `build_fasterrcnn_model()`/`build_maskrcnn_model()` (~lines 5-30):
  target for Phase C threshold exposure. Also home of the (reverted) custom small-object
  anchor generator experiment (`_small_object_anchor_generator()`) — see "out of scope" note
  below.
- `scripts/inference.py` — CLI args `--score_thresh`/`--nms_iou_thresh`/`--final_box_iou`
  (~lines 187-189), score filter (~line 342), per-class NMS (~lines 407-411), final polygon
  NMS (~lines 619-634).
- `scripts/train.py` — `--max_per_class`, `--stratified_sampling`, `--classes_json` args.
- `scripts/validate_training_data.py` — run after every Phase B data-prep step to confirm
  intended class/county ratios actually landed (flags >95% single-class per county, >60%
  single-class overall).
- `src/data/sampling.py` (`StratifiedWeightedSampler`) / `src/utils/tiling.py`
  (`make_label_centered_training_windows()`, `_stratified_quota()`) — relevant to how training
  windows are generated/capped per class and per county; not currently suspected as the main
  driver but relevant if Phase B results are surprising.

## Constraints / decisions already made
- Training and inference run on **SageMaker**, not locally — next steps are config/parameter
  changes to run via SageMaker jobs.
- Prioritize detection volume/recall over precision throughout (user's explicit preference).
- The custom small-object anchor generator (9 anchors/location vs stock 3,
  `_small_object_anchor_generator()` in `model.py`) is explicitly OUT OF SCOPE — already tried
  and reverted due to training instability (very high/noisy early loss, occasional NaN; would
  require training RPN/ROI heads from scratch since it breaks pretrained head weight shapes).
  Current model uses stock torchvision anchors + full pretrained detection weights
  (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`/`MaskRCNN_ResNet50_FPN_Weights.DEFAULT`).
- Not attempting to recover the literal lost fluke config — confirmed unrecoverable; this is
  forward-looking controlled experimentation instead.

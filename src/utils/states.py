# src/utils/states.py
"""Per-state configuration and the single source of truth for every local
path and S3 key the pipeline uses.

Layout (state-first; identical structure locally and on S3):

    data/
      shared/                       feature_size_threshholds.csv, county_boundaries/ (TIGER)
      training/                     merged training labels/VRT, classes JSON
      {STATE}/
        training_county_imagery_years.csv
        nhd_mask/nhd_mask_{state}.*
        counties/{county}/          labels *.shp, tiles/, canonical_tiles/, mosaic VRT + mask caches
        _canonical_chip_cache/
    outputs/
      models/{model}/               model_*.pth, classes.json, train_windows.json
      {STATE}/
        _native_mosaic_cache/, logs/
        inference/{model}/{county}/detections_{county}_{timestamp}.gpkg

    s3://{S3_BUCKET}/{S3_ROOT}/
      shared/                       classes_w_background.json, feature_size_threshholds.csv
      models/{model}/               model_final.pth, classes.json
      {STATE}/
        training_county_imagery_years.csv
        training_labels/{county}/*.shp
        nhd_mask/nhd_mask_{state}.*
        canonical_mosaic/{epsg}/chips/, canonical_mosaic/manifests/{county}.json
        raw_tile_archive/{county}/
        inference/{model}/detections_{county}_{timestamp}.gpkg

Every function that touches state-specific data takes the state code
explicitly - there is deliberately no default state, because many county names
(Lake, Clark, Jackson, Marion, ...) exist in more than one state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from utils.common import project_root, safe_name

S3_BUCKET = "sagemaker-gst-stage.sharing"
S3_ROOT = "serge-wiltshire/runoff-classifier-data"

# Canonical chips are always 0.5 ft/pixel (6-inch), matching the imagery the
# model was trained on; size thresholds are in square feet. Every state's
# canonical CRS must therefore be a projected CRS in (US survey) feet.
CANONICAL_RES = 0.5

DEFAULT_MODEL = "train_multicounty"

# Shared inference parameters (notebooks + statewide driver).
DEFAULT_INFER_CONFIG = {
    "task": "instance_seg",
    "tile_size": 512,          # keep in sync with training tile_size
    "stride": 256,             # 50% overlap
    "infer_batch": 5,
    "score_thresh": 0.75,
    "normalize": "imagenet",
    "nms_iou_thresh": 0.3,
    "final_box_iou": 0.3,
    "min_cover_frac": 0.01,
    # county-boundary filter: keep a detection only if >=50% of its box lies
    # inside the county, so straddlers belong to exactly one county's output
    "county_min_cover_frac": 0.5,
}


@dataclass(frozen=True)
class StateConfig:
    code: str                      # two-letter postal code, e.g. "IN"
    name: str
    fips: str                      # Census state FIPS, e.g. "18"
    counties: Tuple[str, ...]      # county names; folder names are safe_name(county)
    canonical_crs: Optional[str]   # None = not yet decided (chip prep refuses to run)
    imagery_source: Optional[str]  # key into utils.imagery_sources; None = not implemented


_IN_COUNTIES = (
    'Adams', 'Allen', 'Bartholomew', 'Benton', 'Blackford', 'Boone', 'Brown',
    'Carroll', 'Cass', 'Clark', 'Clay', 'Clinton', 'Crawford', 'Daviess',
    'Dearborn', 'Decatur', 'DeKalb', 'Delaware', 'Dubois', 'Elkhart', 'Fayette',
    'Floyd', 'Fountain', 'Franklin', 'Fulton', 'Gibson', 'Grant', 'Greene',
    'Hamilton', 'Hancock', 'Harrison', 'Hendricks', 'Henry', 'Howard', 'Huntington',
    'Jackson', 'Jasper', 'Jay', 'Jefferson', 'Jennings', 'Johnson', 'Knox',
    'Kosciusko', 'LaGrange', 'Lake', 'LaPorte', 'Lawrence', 'Madison', 'Marion',
    'Marshall', 'Martin', 'Miami', 'Monroe', 'Montgomery', 'Morgan', 'Newton',
    'Noble', 'Ohio', 'Orange', 'Owen', 'Parke', 'Perry', 'Pike', 'Porter',
    'Posey', 'Pulaski', 'Putnam', 'Randolph', 'Ripley', 'Rush', 'Scott',
    'Shelby', 'Spencer', 'Starke', 'Steuben', 'St. Joseph', 'Sullivan', 'Switzerland',
    'Tippecanoe', 'Tipton', 'Union', 'Vanderburgh', 'Vermillion', 'Vigo', 'Wabash',
    'Warren', 'Warrick', 'Washington', 'Wayne', 'Wells', 'White', 'Whitley',
)

# Census 2020 county list (st17_il_cou2020.txt), "County" suffix removed.
_IL_COUNTIES = (
    'Adams', 'Alexander', 'Bond', 'Boone', 'Brown', 'Bureau', 'Calhoun', 'Carroll',
    'Cass', 'Champaign', 'Christian', 'Clark', 'Clay', 'Clinton', 'Coles', 'Cook',
    'Crawford', 'Cumberland', 'DeKalb', 'De Witt', 'Douglas', 'DuPage', 'Edgar',
    'Edwards', 'Effingham', 'Fayette', 'Ford', 'Franklin', 'Fulton', 'Gallatin',
    'Greene', 'Grundy', 'Hamilton', 'Hancock', 'Hardin', 'Henderson', 'Henry',
    'Iroquois', 'Jackson', 'Jasper', 'Jefferson', 'Jersey', 'Jo Daviess', 'Johnson',
    'Kane', 'Kankakee', 'Kendall', 'Knox', 'Lake', 'LaSalle', 'Lawrence', 'Lee',
    'Livingston', 'Logan', 'McDonough', 'McHenry', 'McLean', 'Macon', 'Macoupin',
    'Madison', 'Marion', 'Marshall', 'Mason', 'Massac', 'Menard', 'Mercer', 'Monroe',
    'Montgomery', 'Morgan', 'Moultrie', 'Ogle', 'Peoria', 'Perry', 'Piatt', 'Pike',
    'Pope', 'Pulaski', 'Putnam', 'Randolph', 'Richland', 'Rock Island', 'St. Clair',
    'Saline', 'Sangamon', 'Schuyler', 'Scott', 'Shelby', 'Stark', 'Stephenson',
    'Tazewell', 'Union', 'Vermilion', 'Wabash', 'Warren', 'Washington', 'Wayne',
    'White', 'Whiteside', 'Will', 'Williamson', 'Winnebago', 'Woodford',
)

STATES = {
    "IN": StateConfig(
        code="IN", name="Indiana", fips="18", counties=_IN_COUNTIES,
        # NAD83(HARN) / Indiana East (ftUS). Must stay fixed: every existing IN
        # canonical chip and manifest is on this grid.
        canonical_crs="EPSG:2968",
        imagery_source="indiana_gis",
    ),
    "IL": StateConfig(
        code="IL", name="Illinois", fips="17", counties=_IL_COUNTIES,
        canonical_crs=None,      # TODO: choose once the IL imagery has been investigated
        imagery_source=None,     # TODO: implement an IL fetcher in utils/imagery_sources.py
    ),
}


def get_state(state: str | StateConfig) -> StateConfig:
    if isinstance(state, StateConfig):
        return state
    try:
        return STATES[str(state).strip().upper()]
    except KeyError:
        raise KeyError(f"unknown state {state!r}; configured: {sorted(STATES)}") from None


def require_canonical_crs(state: str | StateConfig) -> str:
    """Return the state's canonical CRS, refusing to continue unless it's set
    and is a projected CRS in feet (CANONICAL_RES and the area thresholds are
    in feet)."""
    cfg = get_state(state)
    if not cfg.canonical_crs:
        raise NotImplementedError(
            f"{cfg.code}: canonical CRS is not configured yet - set StateConfig.canonical_crs "
            f"for {cfg.code} in src/utils/states.py before generating canonical chips."
        )
    from pyproj import CRS

    crs = CRS.from_user_input(cfg.canonical_crs)
    units = {(ax.unit_name or "").lower() for ax in crs.axis_info}
    if not crs.is_projected or not units or not all("foot" in u for u in units):
        raise ValueError(
            f"{cfg.code}: canonical CRS {cfg.canonical_crs} must be a projected CRS in feet "
            f"(got projected={crs.is_projected}, units={sorted(units)})"
        )
    return cfg.canonical_crs


# ---------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------

def data_root() -> Path:
    return project_root() / "data"


def outputs_root() -> Path:
    return project_root() / "outputs"


def shared_data_dir() -> Path:
    return data_root() / "shared"


def county_boundaries_dir() -> Path:
    return shared_data_dir() / "county_boundaries"


def feature_size_thresholds_path() -> Path:
    return shared_data_dir() / "feature_size_threshholds.csv"


def training_data_dir() -> Path:
    return data_root() / "training"


def state_data_dir(state) -> Path:
    return data_root() / get_state(state).code


def counties_dir(state) -> Path:
    return state_data_dir(state) / "counties"


def county_dir(state, county: str) -> Path:
    return counties_dir(state) / safe_name(county.strip())


def raw_tiles_dir(state, county: str) -> Path:
    return county_dir(state, county) / "tiles"


def canonical_tiles_dir(state, county: str) -> Path:
    return county_dir(state, county) / "canonical_tiles"


def chip_cache_dir(state) -> Path:
    return state_data_dir(state) / "_canonical_chip_cache"


def imagery_years_csv(state) -> Path:
    return state_data_dir(state) / "training_county_imagery_years.csv"


def nhd_mask_dir(state) -> Path:
    return state_data_dir(state) / "nhd_mask"


def nhd_mask_path(state) -> Path:
    return nhd_mask_dir(state) / f"nhd_mask_{get_state(state).code.lower()}.shp"


def state_outputs_dir(state) -> Path:
    return outputs_root() / get_state(state).code


def native_mosaic_cache_dir(state) -> Path:
    return state_outputs_dir(state) / "_native_mosaic_cache"


def logs_dir(state) -> Path:
    return state_outputs_dir(state) / "logs"


def inference_dir(state, model: str = DEFAULT_MODEL) -> Path:
    return state_outputs_dir(state) / "inference" / model


def county_inference_dir(state, county: str, model: str = DEFAULT_MODEL) -> Path:
    return inference_dir(state, model) / safe_name(county.strip())


def models_dir() -> Path:
    return outputs_root() / "models"


def model_dir(model: str = DEFAULT_MODEL) -> Path:
    return models_dir() / model


# Final merged inference output: detections_{county}_{YYYYmmdd_HHMMSS}.gpkg
# (per-rank partials end in _rankN.gpkg and never match).
_FINAL_DETECTIONS_RE = re.compile(r"detections_(?P<county>.+)_\d{8}_\d{6}\.gpkg")


def final_detections_county(filename: str) -> Optional[str]:
    """County name if `filename` is a final merged detections gpkg, else None."""
    m = _FINAL_DETECTIONS_RE.fullmatch(filename)
    return m.group("county") if m else None


# ---------------------------------------------------------------------
# S3 keys (bucket-relative; use s3_uri() for s3:// URIs)
# ---------------------------------------------------------------------

def s3_uri(key: str) -> str:
    return f"s3://{S3_BUCKET}/{key}"


def s3_shared_prefix() -> str:
    return f"{S3_ROOT}/shared"


def s3_model_prefix(model: str = DEFAULT_MODEL) -> str:
    return f"{S3_ROOT}/models/{model}"


def s3_state_prefix(state) -> str:
    return f"{S3_ROOT}/{get_state(state).code}"


def s3_canonical_prefix(state) -> str:
    return f"{s3_state_prefix(state)}/canonical_mosaic"


def s3_manifest_key(state, county: str) -> str:
    return f"{s3_canonical_prefix(state)}/manifests/{safe_name(county.strip())}.json"


def s3_raw_archive_prefix(state) -> str:
    return f"{s3_state_prefix(state)}/raw_tile_archive"


def s3_training_labels_prefix(state) -> str:
    return f"{s3_state_prefix(state)}/training_labels"


def s3_nhd_mask_prefix(state) -> str:
    return f"{s3_state_prefix(state)}/nhd_mask"


def s3_imagery_years_key(state) -> str:
    return f"{s3_state_prefix(state)}/training_county_imagery_years.csv"


def s3_inference_prefix(state, model: str = DEFAULT_MODEL) -> str:
    return f"{s3_state_prefix(state)}/inference/{model}"

# county_boundaries.py
"""County boundary sourcing from US Census TIGER/Line.

Provides per-county boundary polygons used to filter inference detections to a
county's administrative extent (analogous to the NHD AOI mask, see
src/utils/fast_mask.py and scripts/inference.py --county_mask_path).

Design:
- The national full-resolution TIGER/Line county file (tl_{year}_us_county.zip,
  ~80 MB, all US counties) is downloaded ONCE into data/county_boundaries/ and
  kept as the source of truth. Full-res TIGER/Line is used (not the generalized
  cb_*_500k cartographic files) because boundary accuracy matters for the
  majority-inside (>=50% coverage) detection-filter rule at county edges.
- Each requested county is extracted to a small single-feature GeoJSON cached at
  data/county_boundaries/{state_fips}/{county}.geojson, so the big zip is only
  read on cache misses.
- National scope (keyed by state FIPS, default Indiana "18") so this extends to
  other states without code changes.

The cached GeoJSON is left in TIGER's native CRS (EPSG:4269); consumers
(get_mask_clipped in fast_mask.py) reproject to the raster CRS themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .indiana_cogs import (
    ensure_dir,
    head_length,
    make_session,
    normalize_county_key,
    project_root,
    safe_name,
)

logger = logging.getLogger("county_boundaries")

TIGER_YEAR = 2024
TIGER_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"
)

# State FIPS codes for states this pipeline targets. Extend as needed.
STATE_FIPS = {
    "indiana": "18",
}
DEFAULT_STATE_FIPS = STATE_FIPS["indiana"]


def _boundaries_dir(data_dir: Optional[Path] = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return project_root() / "data" / "county_boundaries"


def ensure_national_county_zip(
    data_dir: Optional[Path] = None,
    year: int = TIGER_YEAR,
) -> Path:
    """Download the national TIGER/Line county zip once; return its local path.

    Skips the download when a local copy already exists and matches the remote
    Content-Length (guards against a previously-interrupted partial download
    being silently reused). Downloads go to a .part temp file and are atomically
    renamed into place only on success.
    """
    out_dir = _boundaries_dir(data_dir)
    ensure_dir(out_dir)
    url = TIGER_URL_TEMPLATE.format(year=year)
    zip_path = out_dir / f"tl_{year}_us_county.zip"

    session = make_session()

    if zip_path.exists():
        remote_len = head_length(session, url)
        if remote_len is None or zip_path.stat().st_size == remote_len:
            return zip_path
        logger.warning(
            "[county_boundaries] cached %s size %d != remote %d; re-downloading",
            zip_path.name, zip_path.stat().st_size, remote_len,
        )
        zip_path.unlink()

    logger.info("[county_boundaries] downloading %s -> %s", url, zip_path)
    tmp_path = zip_path.with_suffix(".zip.part")
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    remote_len = head_length(session, url)
    if remote_len is not None and tmp_path.stat().st_size != remote_len:
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"TIGER county zip download incomplete: got {tmp_path.stat().st_size} "
            f"bytes, expected {remote_len}"
        )
    tmp_path.replace(zip_path)
    logger.info("[county_boundaries] downloaded %s (%d bytes)", zip_path.name, zip_path.stat().st_size)
    return zip_path


def get_county_boundary_path(
    county: str,
    state_fips: str = DEFAULT_STATE_FIPS,
    data_dir: Optional[Path] = None,
    year: int = TIGER_YEAR,
) -> Path:
    """Return the path to a single-feature GeoJSON of `county`'s boundary.

    On first request for a county, extracts it from the (downloaded-once)
    national TIGER/Line county file and caches it at
    data/county_boundaries/{state_fips}/{county}.geojson. Subsequent calls are
    a pure existence check.

    County name matching uses normalize_county_key() (whitespace/case
    insensitive), so "La Porte" / "LaPorte" / "LAPORTE" all resolve identically.

    Raises ValueError if the county matches zero or multiple TIGER features
    within the given state.
    """
    out_dir = _boundaries_dir(data_dir) / state_fips
    ensure_dir(out_dir)
    out_path = out_dir / f"{safe_name(county)}.geojson"
    if out_path.exists():
        return out_path

    # geopandas import deferred: keeps module import light for callers that
    # only hit the cached-path fast path.
    import geopandas as gpd

    zip_path = ensure_national_county_zip(data_dir=data_dir, year=year)
    logger.info("[county_boundaries] extracting %s (state FIPS %s) from %s",
                county, state_fips, zip_path.name)
    gdf = gpd.read_file(zip_path)

    key = normalize_county_key(county)
    in_state = gdf[gdf["STATEFP"] == state_fips]
    if len(in_state) == 0:
        raise ValueError(f"no TIGER counties found for state FIPS {state_fips!r}")
    matches = in_state[in_state["NAME"].map(normalize_county_key) == key]

    if len(matches) == 0:
        available = sorted(in_state["NAME"].tolist())
        raise ValueError(
            f"county {county!r} not found in TIGER state FIPS {state_fips}. "
            f"Available: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"county {county!r} matched {len(matches)} TIGER features in state "
            f"FIPS {state_fips}: {matches['NAME'].tolist()}"
        )

    matches.to_file(out_path, driver="GeoJSON")
    logger.info("[county_boundaries] cached boundary -> %s", out_path)
    return out_path

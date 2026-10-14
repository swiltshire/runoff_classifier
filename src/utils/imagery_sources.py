# src/utils/imagery_sources.py
"""Per-state raw imagery fetchers.

`download_raw_tiles()` is the only entry point the canonical chip pipeline uses
to obtain raw tiles. Each state's `StateConfig.imagery_source` names its
fetcher here. A fetcher must write GeoTIFF tiles (any projected CRS - they are
mosaicked and reprojected per native CRS afterwards) into `dest_dir` and return
a dict with at least `year` and `failed` (number of tiles that failed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from utils.states import get_state


def _indiana_gis(county: str, dest_dir: Path, *, max_workers: int, imagery_year: Optional[int]) -> Dict:
    from utils.indiana_cogs import download_6in_tiles

    return download_6in_tiles(county, dest_dir, max_workers=max_workers, imagery_year=imagery_year)


_FETCHERS = {
    "indiana_gis": _indiana_gis,
}


def download_raw_tiles(
    state,
    county: str,
    dest_dir: Path,
    *,
    max_workers: int = 16,
    imagery_year: Optional[int] = None,
) -> Dict:
    cfg = get_state(state)
    fetcher = _FETCHERS.get(cfg.imagery_source or "")
    if fetcher is None:
        raise NotImplementedError(
            f"{cfg.code}: no raw imagery source is implemented yet "
            f"(StateConfig.imagery_source={cfg.imagery_source!r}). Add a fetcher to "
            f"src/utils/imagery_sources.py and set imagery_source for {cfg.code} in "
            f"src/utils/states.py."
        )
    return fetcher(county, Path(dest_dir), max_workers=max_workers, imagery_year=imagery_year)

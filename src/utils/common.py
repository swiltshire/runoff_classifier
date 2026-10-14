# src/utils/common.py
"""State-agnostic helpers shared across the pipeline."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def project_root() -> Path:
    """Auto-detect the project root (works from the repo root, notebooks/, or scripts/)."""
    cwd = Path.cwd().resolve()
    if cwd.name.lower() == "notebooks":
        return cwd.parent
    for p in [cwd] + list(cwd.parents):
        if (p / "data").is_dir() or (p / ".git").is_dir():
            return p
    return cwd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", s)


def normalize_county_key(name: str) -> str:
    """Canonical lookup key for a county name, robust to spacing/casing
    differences (e.g. "La Porte", "LaPorte", "LAPORTE" all normalize to the
    same key). Use this wherever a county name from one source (CSV, notebook
    widget, folder name) is compared against another.
    """
    return re.sub(r"\s+", "", name).strip().lower()


def make_session() -> requests.Session:
    retry_cfg = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=64, pool_maxsize=64)
    s = requests.Session()
    s.headers.update({"accept-encoding": "identity", "connection": "keep-alive"})
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def head_length(session: requests.Session, url: str) -> Optional[int]:
    try:
        r = session.head(url, timeout=30)
        if r.ok:
            cl = r.headers.get("content-length")
            return int(cl) if cl else None
    except (requests.RequestException, ValueError):
        pass
    return None


def load_imagery_years_csv(csv_path: Path) -> Dict[str, int]:
    """Load a county -> imagery year mapping CSV (columns `County,Data Year`,
    e.g. "Bartholomew,21"). 2-digit years are converted to 4-digit.

    Returns {normalize_county_key(county): year}; callers must normalize their
    own county name the same way before looking it up.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Imagery years CSV not found: {csv_path}")

    result: Dict[str, int] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(reader.fieldnames) != {"County", "Data Year"}:
            raise ValueError(f"CSV must have exactly columns 'County' and 'Data Year', got: {reader.fieldnames}")
        for row in reader:
            year = int(row["Data Year"].strip())
            if year < 100:
                year += 2000
            result[normalize_county_key(row["County"])] = year
    return result

# indiana_cogs.py

import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

SERVICE_URL = "https://gisdata.in.gov/server/rest/services/Hosted/Indiana_Orthoimagery_Tile_Footprints/FeatureServer"

# Reference CRS for reprojection (set via set_reference_crs(), persisted to outputs/.reference_crs)
CANONICAL_CRS = "EPSG:2968"  # Default until set_reference_crs() is called


# ---------------- project root ----------------

def project_root() -> Path:
    """auto-detect project root."""
    cwd = Path.cwd().resolve()
    if cwd.name.lower() == "notebooks":
        return cwd.parent
    for p in [cwd] + list(cwd.parents):
        if (p / "data").is_dir() or (p / ".git").is_dir():
            return p
    return cwd


# ---------------- retry session ----------------

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


def set_reference_crs(crs: str) -> None:
    """Update the global reference CRS for re-projection.
    
    Args:
        crs: CRS string (e.g., "EPSG:2968")
    """
    global CANONICAL_CRS
    CANONICAL_CRS = crs


# ---------------- helpers ----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", s)

def normalize_county_key(name: str) -> str:
    """Canonical lookup key for a county name, robust to spacing/casing
    differences (e.g. "La Porte", "LaPorte", "LAPORTE" all normalize to the
    same key). Use this at every point where a county name from one source
    (CSV, notebook widget, folder name) is compared against another, instead
    of exact string equality.
    """
    return re.sub(r"\s+", "", name).strip().lower()

def head_length(session: requests.Session, url: str) -> Optional[int]:
    try:
        r = session.head(url, timeout=30)
        if r.ok:
            cl = r.headers.get("content-length")
            return int(cl) if cl else None
    except (requests.RequestException, ValueError):
        pass
    return None


# ---------------- defensive validation ----------------

def is_valid_tiff_header(path: Path) -> bool:
    """Check if file has valid TIFF header."""
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig in (b"II*\x00", b"MM\x00*")
    except (IOError, OSError):
        return False


def matches_remote_size(path: Path, remote_size: Optional[int]) -> bool:
    """Verify local file matches remote size."""
    if remote_size is None:
        return True  # can't verify
    try:
        return path.stat().st_size == remote_size
    except (IOError, OSError):
        return False


# ---------------- reference CRS ----------------

def get_remote_crs(url: str) -> str:
    """Query remote raster CRS without downloading.
    
    Opens remote GeoTIFF via rasterio and returns CRS as string.
    Returns "UNKNOWN" if CRS cannot be determined after retries.
    
    Retries up to 2 times with backoff to distinguish network hiccups
    (which succeed on retry) from actual data corruption (consistent failures).
    """
    max_retries = 2
    
    for attempt in range(max_retries + 1):
        try:
            import rasterio
            with rasterio.open(url) as ds:
                if ds.crs:
                    # Return as "EPSG:code" format if possible
                    epsg = ds.crs.to_epsg()
                    if epsg:
                        return f"EPSG:{epsg}"
                    return str(ds.crs)
                return "UNKNOWN"
        except Exception:
            # Retry with backoff on transient errors
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))  # exponential backoff: 0.5s, 1s
                continue
            # After all retries exhausted, mark as UNKNOWN
            return "UNKNOWN"


# ---------------- FeatureServer queries ----------------

def get_layers(session: requests.Session) -> List[Tuple[int,int,str]]:
    """Find Footprint_YYYY layers sorted newest->oldest."""
    r = session.get(SERVICE_URL, params={"f":"pjson"}, timeout=30)
    r.raise_for_status()
    js = r.json()
    out = []
    for lyr in js.get("layers", []):
        m = re.search(r"(\d{4})$", lyr["name"])
        if m:
            out.append((int(m.group(1)), int(lyr["id"]), lyr["name"]))
    return sorted(out, reverse=True)

def county_where(county: str) -> str:
    # Some counties (La Porte/LaPorte, De Kalb/DeKalb, La Grange/LaGrange)
    # are spelled with a space in our CSV/notebooks but without one in the
    # ArcGIS service's `county` attribute (or vice versa), and a literal
    # LIKE match against only one spelling silently returns zero rows in
    # that case. Same for punctuation: we spell "St. Joseph" with a period
    # but the service stores "St Joseph" (confirmed via a distinct-values
    # query). OR together the as-given, space-stripped, period-stripped, and
    # period+space-stripped variants (rather than using SQL REPLACE(), which
    # this ArcGIS service's query engine doesn't support - it silently
    # returns zero features instead of erroring).
    names = {county.strip(), county.strip().replace(".", "").strip()}
    variants = set()
    for name in names:
        name = re.sub(r"\s+", " ", name)
        variants.add(name.replace("'", "''").upper())
        variants.add(re.sub(r"\s+", "", name).replace("'", "''").upper())
    return " OR ".join(f"UPPER(county) LIKE '%{v}%'" for v in sorted(variants))


def load_training_imagery_years(csv_path: Path) -> Dict[str, int]:
    """Load training county to imagery year mapping from CSV.
    
    CSV format: County,Data Year (e.g., "Bartholomew,21")
    Converts 2-digit years to 4-digit (21 -> 2021, 23 -> 2023).
    
    Args:
        csv_path: Path to training_county_imagery_years.csv
        
    Returns:
        Dict mapping normalize_county_key(county name) -> 4-digit year (e.g.,
        {"bartholomew": 2021, "benton": 2023}). Keys are normalized via
        `normalize_county_key()` (spaces stripped, lowercased) rather than the
        raw CSV spelling, so callers must normalize their own county name the
        same way before looking it up.
        
    Raises:
        FileNotFoundError: If CSV not found
        ValueError: If CSV format invalid
    """
    import csv
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Imagery years CSV not found: {csv_path}")
    
    result = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(reader.fieldnames) != {"County", "Data Year"}:
            raise ValueError(f"CSV must have exactly columns 'County' and 'Data Year', got: {reader.fieldnames}")
        
        for row in reader:
            county_key = normalize_county_key(row["County"])
            year_str = row["Data Year"].strip()
            
            # Convert 2-digit year to 4-digit
            year_int = int(year_str)
            if year_int < 100:
                year_int = 2000 + year_int
            
            result[county_key] = year_int
    
    return result


def year_to_layer_id(year: int, session: requests.Session) -> Optional[Tuple[int, str]]:
    """Map imagery year to feature server layer ID and name.
    
    Args:
        year: 4-digit year (e.g., 2021)
        session: requests session
        
    Returns:
        Tuple (layer_id, layer_name) or None if year not found
    """
    layers = get_layers(session)
    for lyr_year, lyr_id, lyr_name in layers:
        if lyr_year == year:
            return (lyr_id, lyr_name)
    return None


def get_tile_count(
    session: requests.Session,
    layer_url: str,
    county: str,
    pixel_size: str = "06 in."
) -> int:
    """Count 6-inch tiles for a county in a specific layer.
    
    Args:
        session: requests session
        layer_url: Feature server layer URL (e.g., SERVICE_URL/0)
        county: County name
        pixel_size: Pixel size filter (default "06 in.")
        
    Returns:
        Total count of matching tiles (handles pagination)
    """
    where = county_where(county)
    count = 0
    offset = 0
    
    while True:
        params = {
            "where": where,
            "outFields": "pixel_size",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": 2000,
            "f": "json"
        }
        r = session.get(f"{layer_url}/query", params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        feats = js.get("features", [])
        
        # Count features matching pixel_size filter
        count += sum(1 for f in feats if f.get("attributes", {}).get("pixel_size", "") == pixel_size)
        
        if not feats or not js.get("exceededTransferLimit"):
            break
        offset += len(feats)
    
    return count


def find_complete_imagery_year(
    session: requests.Session,
    county: str,
    min_year: int = 2020
) -> Tuple[int, int, int, str]:
    """Find the most recent year with a complete tile set for a county.
    
    Completeness determined by tile count: selects most recent year where 
    tile count equals the maximum seen (indicating full coverage from that year).
    
    Args:
        session: requests session
        county: County name
        min_year: Minimum year to check (default 2020)
        
    Returns:
        Tuple (selected_year, tile_count, layer_id, layer_name)
        
    Raises:
        RuntimeError: If no complete year found or no 6-inch tiles found for county
    """
    layers = get_layers(session)
    if not layers:
        raise RuntimeError("No Footprint_YYYY layers found")
    
    # Collect tile counts per year (newest to oldest)
    year_counts = {}
    for year, layer_id, layer_name in layers:
        if year < min_year:
            break
        
        layer_url = f"{SERVICE_URL}/{layer_id}"
        count = get_tile_count(session, layer_url, county)
        year_counts[year] = (count, layer_id, layer_name)
    
    if not year_counts:
        raise RuntimeError(f"No layers found after year {min_year}")
    
    # Find max count
    max_count = max(count for count, _, _ in year_counts.values())
    if max_count == 0:
        raise RuntimeError(f"No 6-inch tiles found for county {county} in years {min_year}+")
    
    # Return most recent year with max count
    for year in sorted(year_counts.keys(), reverse=True):
        count, layer_id, layer_name = year_counts[year]
        if count == max_count:
            return (year, count, layer_id, layer_name)


def quick_reference_crs_from_county(
    county: str,
    session: requests.Session,
    imagery_years_csv: Optional[Path] = None,
) -> str:
    """Determine the pipeline's reference CRS by reading a single
    representative tile from one county.

    Correctness note: with the mosaic-then-reproject architecture (see
    `prepare_reprojected_tiles.py`), the reference CRS choice is a pure
    efficiency/convenience decision, not a correctness one - EVERY native-CRS
    group (including one that already matches the chosen reference CRS) is
    warped through the same `gdalwarp -t_srs` pipeline, so picking a CRS
    that isn't the true county-wide majority just means a few more tiles get
    warped instead of passed through unchanged.

    Args:
        county: county name to sample (e.g. "Benton")
        session: requests session
        imagery_years_csv: optional path to training_county_imagery_years.csv;
            if provided and `county` has an entry, uses that pinned year
            (consistent with the rest of the pipeline); otherwise auto-detects
            the newest/most-complete year for `county`.

    Returns:
        CRS string, e.g. "EPSG:2968"

    Raises:
        RuntimeError: if no 6-inch tile / CRS can be found for `county`
    """
    imagery_years: Dict[str, int] = {}
    if imagery_years_csv:
        imagery_years_csv = Path(imagery_years_csv)
        if imagery_years_csv.exists():
            imagery_years = load_training_imagery_years(imagery_years_csv)

    pinned_year = imagery_years.get(normalize_county_key(county))
    if pinned_year is not None:
        layer_info = year_to_layer_id(pinned_year, session)
        if not layer_info:
            raise RuntimeError(f"CSV specifies year {pinned_year} for {county} but Footprint_{pinned_year} layer not found")
        layer_id, layer_name = layer_info
        year = pinned_year
    else:
        year, _, layer_id, layer_name = find_complete_imagery_year(session, county)

    layer_url = f"{SERVICE_URL}/{layer_id}"
    attrs = fetch_attrs(session, layer_url, county_where(county))
    six_inch = [a for a in attrs if a.get("pixel_size") == "06 in." and a.get("url_tif")]
    if not six_inch:
        raise RuntimeError(f"No 6-inch tiles found for {county} in year {year}")

    sample_url = six_inch[0]["url_tif"]
    crs = get_remote_crs(sample_url)
    print(f"[Quick CRS] Sampled {county} ({year}, 1 of {len(six_inch)} tiles): {crs}")
    return crs


def fetch_attrs(session: requests.Session, layer_url: str, where: str) -> List[Dict]:
    out = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": "name,url_tif,pixel_size",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": 2000,
            "f": "json"
        }
        r = session.get(layer_url+"/query", params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        feats = js.get("features", [])
        out.extend([f["attributes"] for f in feats])
        if not feats or not js.get("exceededTransferLimit"):
            break
        offset += len(feats)
    return out


# ---------------- resumable download ----------------

def download_one(url: str, dest: Path, session: requests.Session) -> str:
    ensure_dir(dest.parent)

    remote = head_length(session, url)

    # if file exists, validate it
    if dest.exists():
        if matches_remote_size(dest, remote) and is_valid_tiff_header(dest):
            return "skipped"
        else:
            dest.unlink(missing_ok=True)

    # download fresh with retry logic
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()

                content_type = r.headers.get("content-type", "").lower()
                if "tif" not in content_type and "image" not in content_type:
                    raise RuntimeError(f"not a tif: {content_type}")

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            
            # Validate after successful download
            if not matches_remote_size(dest, remote):
                dest.unlink(missing_ok=True)
                raise RuntimeError("size mismatch after download")
            
            if not is_valid_tiff_header(dest):
                dest.unlink(missing_ok=True)
                raise RuntimeError("invalid tiff header after download")
            
            # All validations passed
            return "downloaded"

        except (requests.RequestException, OSError, RuntimeError):
            dest.unlink(missing_ok=True)
            if attempt < max_retries:
                # Retry with exponential backoff: 0.5s, 1s
                time.sleep(0.5 * (2 ** attempt))
            else:
                # All retries exhausted
                return "failed"


# ---------------- public api ----------------

def download_6in_tiles(county: str, max_workers: int = 16, imagery_year: Optional[int] = None) -> Dict:
    """Download 6-inch tiles for a county.
    
    Args:
        county: County name
        max_workers: Thread pool size (default 16)
        imagery_year: Specific year to download (if None, auto-detects most complete year)
        
    Returns:
        Dict with download stats including 'year' showing which year was selected
    """
    s = make_session()
    root = project_root()

    county_dir = safe_name(county.strip())
    dest_dir = root / "data" / "counties" / county_dir / "tiles"
    ensure_dir(dest_dir)

    # Determine which year to use
    if imagery_year is not None:
        # Strict mode: use specified year
        year_layer = year_to_layer_id(imagery_year, s)
        if not year_layer:
            raise RuntimeError(f"Specified year {imagery_year} not available (Footprint_{imagery_year} layer not found)")
        
        layer_id, layer_name = year_layer
        layer_url = f"{SERVICE_URL}/{layer_id}"
        attrs = fetch_attrs(s, layer_url, county_where(county))
        
        # Validate 6-inch tiles exist for this county in this year
        six = [a for a in attrs if a.get("pixel_size") == "06 in." and a.get("url_tif")]
        if not six:
            raise RuntimeError(f"No 6-inch tiles found for {county} in year {imagery_year}")
        
        year = imagery_year
        tiles = six
    else:
        # Auto-detect mode: find most complete year
        year, tile_count, layer_id, layer_name = find_complete_imagery_year(s, county)
        layer_url = f"{SERVICE_URL}/{layer_id}"
        attrs = fetch_attrs(s, layer_url, county_where(county))
        
        # Filter for 6-inch tiles
        six = [a for a in attrs if a.get("pixel_size") == "06 in." and a.get("url_tif")]
        if not six:
            raise RuntimeError(f"Auto-detect failed: found {tile_count} tiles but none are 6-inch")
        
        tiles = six

    # build job list
    jobs = []
    for a in tiles:
        name = safe_name(a.get("name") or os.path.basename(a["url_tif"]))
        if not name.lower().endswith(".tif"):
            name += ".tif"
        jobs.append((a["url_tif"], dest_dir / name))

    # progress bar
    downloaded = skipped = failed = 0
    failed_files = []
    year_source = "specified" if imagery_year is not None else "auto-detected"
    desc = f"{county_dir} 06in tiles ({year}, {year_source})"

    with tqdm(total=len(jobs), unit="file", desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_job = {pool.submit(download_one, url, dest, s): (url, dest) for url, dest in jobs}
            for fut in as_completed(future_to_job):
                url, dest = future_to_job[fut]
                try:
                    res = fut.result(timeout=120)
                    if res == "downloaded": downloaded += 1
                    elif res == "skipped": skipped += 1
                    else:
                        failed += 1
                        failed_files.append((dest.name, url))
                except Exception:
                    failed += 1
                    failed_files.append((dest.name, url))
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        "dl": downloaded,
                        "skip": skipped,
                        "fail": failed
                    })

    if failed_files:
        print(f"\n⚠ WARNING: {len(failed_files)}/{len(jobs)} raw tile download(s) FAILED for "
              f"{county} ({year}) - the resulting native-CRS mosaic for this area will have a "
              f"gap wherever these tiles would have been, which produces blank canonical chips "
              f"there. Re-run download_6in_tiles for this county to retry:")
        for name, url in failed_files:
            print(f"    - {name}  <-  {url}")

    return {
        "county": county,
        "year": year,
        "year_source": year_source,
        "output_dir": str(dest_dir),
        "total": len(jobs),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "failed_files": failed_files,
    }
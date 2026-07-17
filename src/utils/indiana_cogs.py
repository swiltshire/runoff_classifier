# indiana_cogs.py

import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

SERVICE_URL = "https://gisdata.in.gov/server/rest/services/Hosted/Indiana_Orthoimagery_Tile_Footprints/FeatureServer"

# Dynamic reference CRS for re-projection (determined by survey_training_crs)
CANONICAL_CRS = "EPSG:2968"  # Default; will be set dynamically based on training data


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


def has_raster_data(path: Path) -> bool:
    """Ensure file contains actual raster data (not empty/corrupted)."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            # read small window for speed
            arr = ds.read(1, window=((0, min(256, ds.height)), (0, min(256, ds.width))))
            return arr.max() != arr.min()
    except Exception:
        return False


# ---------------- defensive validation ----------------

def is_valid_tiff_header(path: Path) -> bool:
    """Check if file has valid TIFF header."""
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig in (b"II*\x00", b"MM\x00*")
    except:
        return False


def matches_remote_size(path: Path, remote_size: Optional[int]) -> bool:
    """Verify local file matches remote size."""
    if remote_size is None:
        return True  # can't verify
    try:
        return path.stat().st_size == remote_size
    except:
        return False


def has_raster_data(path: Path) -> bool:
    """Ensure file contains actual raster data (not empty/corrupted)."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            # read small window for speed
            arr = ds.read(1, window=((0, min(256, ds.height)), (0, min(256, ds.width))))
            return arr.max() != arr.min()
    except:
        return False


# ----------- diagnostics for imagery metadata -----------

def inspect_layer_metadata(session: requests.Session, year: int = None) -> Dict:
    """Inspect available fields and sample tile attributes.
    
    Use this to understand what metadata is available for imagery (capture date, season, etc).
    
    Returns dict with:
      - fields: list of all available field names in the layer
      - sample_attributes: first tile's attributes (to see what data looks like)
      - sample_url: example URL from the layer
    """
    layers = get_layers(session)
    if not layers:
        raise RuntimeError("No Footprint_YYYY layers found")
    
    # Use specified year or newest
    if year:
        matching = [l for l in layers if l[0] == year]
        if not matching:
            raise RuntimeError(f"No layer for year {year}. Available: {[l[0] for l in layers]}")
        layer_year, layer_id, layer_name = matching[0]
    else:
        layer_year, layer_id, layer_name = layers[0]
    
    layer_url = f"{SERVICE_URL}/{layer_id}"
    
    # Get layer info to see available fields
    r = session.get(f"{layer_url}", params={"f": "json"}, timeout=30)
    r.raise_for_status()
    layer_info = r.json()
    
    fields = [f["name"] for f in layer_info.get("fields", [])]
    
    # Query one feature with ALL fields to see what metadata exists
    params = {
        "where": "1=1",
        "outFields": "*",  # all fields
        "returnGeometry": False,
        "resultRecordCount": 1,
        "f": "json"
    }
    r = session.get(f"{layer_url}/query", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    
    sample_attrs = js.get("features", [{}])[0].get("attributes", {})
    
    return {
        "layer": layer_name,
        "year": layer_year,
        "available_fields": fields,
        "sample_attributes": sample_attrs,
        "field_count": len(fields),
        "attribute_count": len(sample_attrs)
    }


# ----------- county metadata table builder -----------

def get_remote_crs(url: str) -> str:
    """Query remote raster CRS without downloading.
    
    Opens remote GeoTIFF via rasterio and returns CRS as string.
    Returns "UNKNOWN" if CRS cannot be determined after retries.
    
    Retries up to 2 times with backoff to distinguish network hiccups
    (which succeed on retry) from actual data corruption (consistent failures).
    """
    import time
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
        except Exception as e:
            # Retry with backoff on transient errors
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))  # exponential backoff: 0.5s, 1s
                continue
            # After all retries exhausted, mark as UNKNOWN
            return "UNKNOWN"


def build_county_metadata_table(
    counties: List[str],
    session: requests.Session,
    imagery_years_csv: Optional[Path] = None,
) -> List[Dict]:
    """Build metadata table for counties (6-inch tiles only).
    
    For each county, queries feature server and remote rasters to gather metadata on 6-inch tiles
    (matching the same tiles downloaded by download_6in_tiles()):
    - Capture dates (list of distinct dates)
    - Pixel size (will be "06 in." after filtering)
    - Tile count
    - CRS distribution (histogram)
    - % of tiles conforming to reference CRS (from global CANONICAL_CRS)
    
    If imagery_years_csv provided, uses specified years for each county (strict mode - fails if year unavailable).
    Otherwise, auto-detects by finding newest layer with 6-inch tiles per county.
    
    Returns list of dicts, one per county, with keys:
      - county: county name
      - imagery_year: year selected (from CSV or auto-detected)
      - layer: layer name (Footprint_YYYY)
      - capture_dates: sorted list of distinct capture dates
      - pixel_size: pixel size string (always "06 in." after filtering)
      - tile_count: number of 6-inch tiles
      - crs_list: list of distinct CRS strings found in 6-inch tiles
      - crs_dict: dict mapping CRS -> count
      - canonical_pct: % of 6-inch tiles in current CANONICAL_CRS
    """
    # Load imagery years if CSV provided
    imagery_years = {}
    if imagery_years_csv:
        imagery_years = load_training_imagery_years(imagery_years_csv)
    
    layers = get_layers(session)
    if not layers:
        raise RuntimeError("No Footprint_YYYY layers found")
    
    results = []
    
    print(f"\n[Metadata] Querying {len(counties)} counties (using reference CRS: {CANONICAL_CRS})...")
    
    for county in tqdm(counties, desc="Counties processed", unit="county"):
        try:
            # Find layer with 6-inch tiles for THIS county
            # If imagery_years_csv provided and county in it, use that specific year (strict mode)
            # Otherwise, auto-detect: find newest layer with 6-inch tiles
            layer_info = None
            
            if imagery_years and county in imagery_years:
                # Strict mode: use specified year from CSV
                specified_year = imagery_years[county]
                year_layer = year_to_layer_id(specified_year, session)
                if not year_layer:
                    raise RuntimeError(f"CSV specifies year {specified_year} but Footprint_{specified_year} layer not found")
                
                layer_id, layer_name = year_layer
                test_url = f"{SERVICE_URL}/{layer_id}"
                attrs_list = fetch_attrs(session, test_url, county_where(county))
                
                # Validate that this layer has 6-in tiles for county
                six_inch_attrs = [a for a in attrs_list if a.get("pixel_size") == "06 in."]
                if not six_inch_attrs:
                    raise RuntimeError(f"Incomplete tile set: County {county} in year {specified_year} has 0 6-inch tiles")
                
                layer_info = (specified_year, layer_id, layer_name, test_url, attrs_list)
            else:
                # Auto-detect mode: find newest layer with 6-inch tiles for this county
                for year, layer_id, layer_name in layers:
                    test_url = f"{SERVICE_URL}/{layer_id}"
                    attrs_list = fetch_attrs(session, test_url, county_where(county))
                    
                    # Check if this layer has 6-in tiles for this county
                    if any(a.get("pixel_size", "") in ("06 in.",) for a in attrs_list):
                        layer_info = (year, layer_id, layer_name, test_url, attrs_list)
                        break
            
            if not layer_info:
                raise RuntimeError("No layers with 6-inch tiles found")
            
            year, layer_id, layer_name, layer_url, attrs_list = layer_info
            
            # Extract metadata from attributes (same filter as download_6in_tiles)
            # Filter for 6-inch tiles only
            capture_dates = set()
            pixel_sizes = set()
            urls = []
            
            for attrs in attrs_list:
                # Only process 6-inch tiles (must match download_6in_tiles filtering)
                # Uses exact same filter: pixel_size in ("06 in.",) and has url_tif
                if attrs.get("pixel_size", "") not in ("06 in.",) or not attrs.get("url_tif"):
                    continue
                
                # Try various date field names
                for date_field in ["capture_date", "capture_date_text", "date", "acquisition_date", "Photo_Date"]:
                    if date_field in attrs and attrs[date_field]:
                        capture_dates.add(str(attrs[date_field]))
                        break
                
                # Get pixel size (will be "06 in." after filter)
                if "pixel_size" in attrs and attrs["pixel_size"]:
                    pixel_sizes.add(str(attrs["pixel_size"]))
                
                # Get URL for CRS extraction
                if "url_tif" in attrs and attrs["url_tif"]:
                    urls.append(attrs["url_tif"])
            
            # Query remote CRS for each tile (parallel - I/O bound)
            crs_dict = {}
            canonical_count = 0
            
            pixel_size_str = sorted(pixel_sizes)[0] if pixel_sizes else "N/A"
            
            print(f"  {county}: Found {len(urls)} 6-in tiles ({pixel_size_str}), querying CRS (parallel)...")
            
            # Use ThreadPoolExecutor for parallel CRS queries (I/O bound)
            # 48 workers for ml.g4dn.12xlarge (48 vCPUs, I/O-bound networking)
            with ThreadPoolExecutor(max_workers=48) as executor:
                futures = {executor.submit(get_remote_crs, url): url for url in urls}
                
                for future in tqdm(as_completed(futures), total=len(urls), desc=f"    {county} CRS", unit="tile", leave=False):
                    try:
                        crs = future.result(timeout=60)
                        if crs not in crs_dict:
                            crs_dict[crs] = 0
                        crs_dict[crs] += 1
                        
                        if crs == CANONICAL_CRS:
                            canonical_count += 1
                    except Exception:
                        # Track failures as UNKNOWN
                        if "UNKNOWN" not in crs_dict:
                            crs_dict["UNKNOWN"] = 0
                        crs_dict["UNKNOWN"] += 1
            
            canonical_pct = (canonical_count / len(urls) * 100) if urls else 0.0
            
            if len(urls) > 0:
                crs_summary = ", ".join([f"{k}({v})" for k, v in sorted(crs_dict.items())])
                print(f"    ✓ CRS: {crs_summary} | EPSG:2968 conformance: {canonical_pct:.1f}%")
            else:
                print(f"    (No 6-in tiles to query)")
            
            results.append({
                "county": county,
                "imagery_year": year,
                "layer": layer_name,
                "capture_dates": sorted(capture_dates),
                "pixel_size": pixel_size_str,
                "tile_count": len(urls),
                "crs_list": sorted(crs_dict.keys()),
                "crs_dict": crs_dict,
                "canonical_pct": round(canonical_pct, 1),
            })
            
        except Exception as e:
            print(f"  ✗ {county}: ERROR - {str(e)}")
            results.append({
                "county": county,
                "imagery_year": None,
                "layer": None,
                "capture_dates": [],
                "pixel_size": "ERROR",
                "tile_count": 0,
                "crs_list": [],
                "crs_dict": {},
                "canonical_pct": 0.0,
                "note": f"Error: {str(e)}"
            })
    
    print(f"\n[Metadata] Complete. Processed {len(results)} counties.\n")
    return results


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
    c = county.replace("'","''").upper()
    return f"UPPER(county) LIKE '%{c}%'"


def load_training_imagery_years(csv_path: Path) -> Dict[str, int]:
    """Load training county to imagery year mapping from CSV.
    
    CSV format: County,Data Year (e.g., "Bartholomew,21")
    Converts 2-digit years to 4-digit (21 -> 2021, 23 -> 2023).
    
    Args:
        csv_path: Path to training_county_imagery_years.csv
        
    Returns:
        Dict mapping county name -> 4-digit year (e.g., {"Bartholomew": 2021, "Benton": 2023})
        
    Raises:
        FileNotFoundError: If CSV not found
        ValueError: If CSV format invalid
    """
    import csv
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Imagery years CSV not found: {csv_path}")
    
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(reader.fieldnames) != {"County", "Data Year"}:
            raise ValueError(f"CSV must have exactly columns 'County' and 'Data Year', got: {reader.fieldnames}")
        
        for row in reader:
            county_name = row["County"].strip().title()
            year_str = row["Data Year"].strip()
            
            # Convert 2-digit year to 4-digit
            year_int = int(year_str)
            if year_int < 100:
                year_int = 2000 + year_int
            
            result[county_name] = year_int
    
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


def survey_training_crs(
    counties: List[str],
    session: requests.Session,
    imagery_years_csv: Path
) -> Tuple[str, Dict[str, int]]:
    """Survey training counties to determine most common CRS.
    
    Samples one 6-inch tile from each training county and queries its remote CRS.
    Returns the most common CRS and the full distribution.
    
    Args:
        counties: List of training county names
        session: requests session
        imagery_years_csv: Path to training_county_imagery_years.csv
        
    Returns:
        Tuple (reference_crs_string, {crs: count, ...}) - e.g., ("EPSG:2968", {"EPSG:2968": 18, "EPSG:2967": 4})
        
    Raises:
        RuntimeError: If unable to determine CRS for training data
    """
    imagery_years = load_training_imagery_years(imagery_years_csv)
    crs_counts = {}
    
    print(f"\n[CRS Survey] Sampling {len(counties)} training counties...")
    
    for county in tqdm(counties, desc="CRS survey", unit="county"):
        if county not in imagery_years:
            print(f"  ⚠ {county}: Not in imagery_years CSV, skipping")
            continue
        
        try:
            year = imagery_years[county]
            layer_info = year_to_layer_id(year, session)
            if not layer_info:
                print(f"  ⚠ {county} ({year}): Layer not found, skipping")
                continue
            
            layer_id, layer_name = layer_info
            layer_url = f"{SERVICE_URL}/{layer_id}"
            
            # Get sample tile
            attrs = fetch_attrs(session, layer_url, county_where(county))
            six_inch = [a for a in attrs if a.get("pixel_size") == "06 in." and a.get("url_tif")]
            
            if not six_inch:
                print(f"  ⚠ {county} ({year}): No 6-inch tiles found, skipping")
                continue
            
            # Query CRS of first tile
            sample_url = six_inch[0]["url_tif"]
            crs = get_remote_crs(sample_url)
            crs_counts[crs] = crs_counts.get(crs, 0) + 1
            
        except Exception as e:
            print(f"  ✗ {county}: {str(e)}")
    
    if not crs_counts:
        raise RuntimeError("Unable to determine CRS from any training county")
    
    # Find most common CRS
    reference_crs = max(crs_counts, key=crs_counts.get)
    print(f"\n[CRS Survey] Selected reference CRS: {reference_crs}")
    print(f"[CRS Survey] Distribution: {', '.join(f'{crs}({cnt})' for crs, cnt in sorted(crs_counts.items(), key=lambda x: -x[1]))}\n")
    
    return reference_crs, crs_counts


def fetch_all_indiana_counties(session: requests.Session) -> List[str]:
    """Query Indiana feature server for all available counties.
    
    Uses the most recent Footprint_YYYY layer and extracts individual county names
    from multi-county coverage regions. Returns sorted, deduplicated county names.
    This is the canonical source for valid Indiana counties with available tiles.
    """
    layers = get_layers(session)
    if not layers:
        raise RuntimeError("No Footprint_YYYY layers found on feature server")
    
    # Use the newest layer
    year, layer_id, layer_name = layers[0]
    layer_url = f"{SERVICE_URL}/{layer_id}"
    
    # Query for all distinct county values
    params = {
        "where": "1=1",  # select all
        "outFields": "county",
        "returnDistinctValues": True,
        "f": "json"
    }
    r = session.get(f"{layer_url}/query", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    
    # Extract individual county names from multi-county coverage regions
    # Feature server returns comma-separated region strings like "Bartholomew, Brown, Jackson"
    all_counties = set()
    for feat in js.get("features", []):
        if "attributes" in feat and "county" in feat["attributes"]:
            region = feat["attributes"]["county"].strip()
            # Split on comma and clean each county name
            county_names = [c.strip().title() for c in region.split(",")]
            all_counties.update(county_names)
    
    if not all_counties:
        raise RuntimeError(f"Feature server returned no counties from layer {layer_name}")
    
    return sorted(all_counties)


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

        except (requests.RequestException, OSError, RuntimeError) as e:
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

    county_dir = safe_name(county.strip().title())
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
    downloaded = resumed = skipped = failed = 0
    year_source = "specified" if imagery_year is not None else "auto-detected"
    desc = f"{county_dir} 06in tiles ({year}, {year_source})"

    with tqdm(total=len(jobs), unit="file", desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(download_one, url, dest, s) for url, dest in jobs]
            for fut in as_completed(futures):
                try:
                    res = fut.result(timeout=120)
                    if res == "downloaded": downloaded += 1
                    elif res == "resumed": resumed += 1
                    elif res == "skipped": skipped += 1
                    else: failed += 1
                except:
                    failed += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        "dl": downloaded,
                        "res": resumed,
                        "skip": skipped,
                        "fail": failed
                    })

    return {
        "county": county,
        "year": year,
        "year_source": year_source,
        "output_dir": str(dest_dir),
        "total": len(jobs),
        "downloaded": downloaded,
        "resumed": resumed,
        "skipped": skipped,
        "failed": failed
    }
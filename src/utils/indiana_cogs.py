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
    Returns "UNKNOWN" if CRS cannot be determined.
    """
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
        return "UNKNOWN"


def build_county_metadata_table(
    counties: List[str],
    session: requests.Session,
) -> List[Dict]:
    """Build metadata table for counties (6-inch tiles only).
    
    For each county, queries feature server and remote rasters to gather metadata on 6-inch tiles
    (matching the same tiles downloaded by download_6in_tiles()):
    - Capture dates (list of distinct dates)
    - Pixel size (will be "06 in." after filtering)
    - Tile count
    - CRS distribution (histogram)
    - % of tiles conforming to reference CRS (EPSG:2968)
    
    Returns list of dicts, one per county, with keys:
      - county: county name
      - capture_dates: sorted list of distinct capture dates
      - pixel_size: pixel size string (always "06 in." after filtering)
      - tile_count: number of 6-inch tiles
      - crs_list: list of distinct CRS strings found in 6-inch tiles
      - crs_dict: dict mapping CRS -> count
      - canonical_pct: % of 6-inch tiles in EPSG:2968
    """
    CANONICAL_CRS = "EPSG:2968"  # Indiana West
    
    layers = get_layers(session)
    if not layers:
        raise RuntimeError("No Footprint_YYYY layers found")
    
    # Loop through layers (newest first) until we find one with 6-inch tiles
    # This matches the logic in download_6in_tiles()
    layer_info = None
    for year, layer_id, layer_name in layers:
        test_url = f"{SERVICE_URL}/{layer_id}"
        # Quick single check: do a broad query to see if this layer has ANY 6-inch tiles
        # Don't filter by county - just get a sample of what's in the layer
        try:
            r = session.get(
                f"{test_url}/query",
                params={
                    "where": "1=1",
                    "outFields": "pixel_size",
                    "returnGeometry": "false",
                    "resultRecordCount": 100,
                    "f": "json"
                },
                timeout=30
            )
            r.raise_for_status()
            js = r.json()
            feats = js.get("features", [])
            has_6in = any(f.get("attributes", {}).get("pixel_size", "") in ("06 in.",) for f in feats)
            if has_6in:
                layer_info = (year, layer_id, layer_name)
                break
        except Exception:
            continue  # try next layer if this one errors
    
    if not layer_info:
        raise RuntimeError("No layers with 6-inch tiles found")
    
    year, layer_id, layer_name = layer_info
    layer_url = f"{SERVICE_URL}/{layer_id}"
    
    results = []
    
    print(f"\n[Metadata] Using {layer_name} (year {year}) - 6-inch tiles detected")
    print(f"[Metadata] Querying {len(counties)} counties...")
    
    for county in tqdm(counties, desc="Counties processed", unit="county"):
        try:
            # Use fetch_attrs for consistent feature server query (same as download_6in_tiles)
            attrs_list = fetch_attrs(session, layer_url, county_where(county))
            
            if not attrs_list:
                print(f"  ⚠ {county}: No tiles found")
                results.append({
                    "county": county,
                    "capture_dates": [],
                    "pixel_size": "N/A",
                    "tile_count": 0,
                    "crs_list": [],
                    "crs_dict": {},
                    "canonical_pct": 0.0,
                    "note": "No tiles found"
                })
                continue
            
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
            
            # Query remote CRS for each tile
            crs_dict = {}
            canonical_count = 0
            
            pixel_size_str = sorted(pixel_sizes)[0] if pixel_sizes else "N/A"
            
            print(f"  {county}: Found {len(urls)} 6-in tiles ({pixel_size_str}), querying CRS...")
            
            for url in tqdm(urls, desc=f"    {county} CRS", unit="tile", leave=False):
                crs = get_remote_crs(url)
                if crs not in crs_dict:
                    crs_dict[crs] = 0
                crs_dict[crs] += 1
                
                if crs == CANONICAL_CRS:
                    canonical_count += 1
            
            canonical_pct = (canonical_count / len(urls) * 100) if urls else 0.0
            
            if len(urls) > 0:
                crs_summary = ", ".join([f"{k}({v})" for k, v in sorted(crs_dict.items())])
                print(f"    ✓ CRS: {crs_summary} | EPSG:2968 conformance: {canonical_pct:.1f}%")
            else:
                print(f"    (No 6-in tiles to query)")
            
            results.append({
                "county": county,
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

    # download fresh
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

    except (requests.RequestException, OSError, RuntimeError):
        dest.unlink(missing_ok=True)
        return "failed"

    # validate after download
    if not matches_remote_size(dest, remote):
        dest.unlink(missing_ok=True)
        return "failed"

    if not is_valid_tiff_header(dest):
        dest.unlink(missing_ok=True)
        return "failed"

    return "downloaded"


# ---------------- public api ----------------

def download_6in_tiles(county: str, max_workers: int = 16) -> Dict:
    """
    super-simple downloader:
      - only 6-inch tiles
      - saves to project_root/data/counties/[County]/tiles/
      - global progress bar (file-based)
    """
    s = make_session()
    root = project_root()

    county_dir = safe_name(county.strip().title())
    dest_dir = root / "data" / "counties" / county_dir / "tiles"
    ensure_dir(dest_dir)

    # find newest year having the county
    for year, layer_id, layer_name in get_layers(s):
        layer_url = f"{SERVICE_URL}/{layer_id}"
        attrs = fetch_attrs(s, layer_url, county_where(county))

        # filter: only 6-inch tiles
        six = [
            a for a in attrs
            if a.get("pixel_size","") in ("06 in.")
            and a.get("url_tif")
        ]
        if six:
            tiles = six
            break
    else:
        raise RuntimeError(f"no 6-inch tiles found for county {county}")

    # build job list
    jobs = []
    for a in tiles:
        name = safe_name(a.get("name") or os.path.basename(a["url_tif"]))
        if not name.lower().endswith(".tif"):
            name += ".tif"
        jobs.append((a["url_tif"], dest_dir / name))

    # progress bar
    downloaded = resumed = skipped = failed = 0
    desc = f"{county_dir} 06in tiles ({year})"

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
        "output_dir": str(dest_dir),
        "total": len(jobs),
        "downloaded": downloaded,
        "resumed": resumed,
        "skipped": skipped,
        "failed": failed
    }
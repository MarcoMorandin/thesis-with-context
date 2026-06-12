"""
Phase 1 Dataset Download Script — 1TB Balanced Mix
====================================================
Covers three priority domains:
  - Meteorology & Earth Observation  (~290 GB, 29.6%)
  - Mobility & Traffic               (~200 GB, 20.4%)
  - Solar / PV                       (~160 GB, 16.3%)

Usage:
  # Dry-run to see what would be downloaded
  python scripts/download_phase1.py --dry-run

  # Download a specific domain
  python scripts/download_phase1.py --domain meteorology
  python scripts/download_phase1.py --domain mobility
  python scripts/download_phase1.py --domain solar

  # Download everything
  python scripts/download_phase1.py --all

  # Override data root
  python scripts/download_phase1.py --all --data-root /mnt/4TB/data

Requirements:
  uv add cdsapi boto3 gluonts earthnet requests tqdm

Environment variables:
  CDS_API_KEY   — Copernicus Climate Data Store API key (for ERA5)
  AWS_DEFAULT_REGION=us-east-1  — for GOES-16 (public bucket, no auth needed)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# Load .env from project root (two levels up from scripts/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)  # env vars already set in shell take precedence
    except ImportError:
        pass  # dotenv not installed; fall back to shell environment only

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/raw"))

STORAGE_LAYOUT = {
    "meteorology": DATA_ROOT / "meteorology",
    "mobility":    DATA_ROOT / "mobility",
    "solar":       DATA_ROOT / "solar",
}

STATUS_FILE = Path("data/download_status.json")

# Expected sizes in GB (from Notion plan)
DATASET_SIZES_GB = {
    "meteonet":           90,
    "earthnet2021":      100,
    "era5_eu":            80,
    "rainbench":          20,
    "i24_wavex":          50,
    "i24_motion":        100,
    "marvel":             30,
    "gluonts_electricity": 0.5,
    "skippd":             31,
    "girasol":            16,
    "goes16_nsrdb":      100,
    "solarnet":           50,
}


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {}


def save_status(status: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def mark_done(status: dict, key: str):
    status[key] = {"state": "done", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_status(status)


def is_done(status: dict, key: str) -> bool:
    return status.get(key, {}).get("state") == "done"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], cwd: Path | None = None):
    """Run a subprocess, streaming output."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def pip_install(*packages: str):
    import importlib
    import shutil
    if shutil.which("uv"):
        run(["uv", "add", "--quiet", *packages])
    else:
        run([sys.executable, "-m", "pip", "install", "--quiet", *packages])
    importlib.invalidate_caches()


def dry_run_summary():
    total = sum(DATASET_SIZES_GB.values())
    print("\n=== DRY RUN — Phase 1 Download Plan ===\n")
    domains = {
        "Meteorology & Earth Obs (~290 GB)": [
            ("MeteoNet",            "meteonet",           90,   "wget from meteonet.umr-cnrm.fr"),
            ("EarthNet2021",        "earthnet2021",       100,  "earthnet Python library"),
            ("ERA5 EU subset",      "era5_eu",            80,   "CDS API (requires CDS_API_KEY)"),
            ("RainBench",           "rainbench",          20,   "GCS bucket — manual access request required"),
        ],
        "Mobility & Traffic (~200 GB)": [
            ("I-24 MOTION subset",  "i24_motion",         100,  "i24motion.org — manual token required"),
            ("MARVEL",              "marvel",             30,   "marvel-project.eu — manual request"),
            ("GluonTS Electricity", "gluonts_electricity", 0.5, "GluonTS Python library"),
        ],
        "Solar / PV (~160 GB)": [
            ("SKIPP'D",             "skippd",             31,   "GitHub + Zenodo"),
            ("Girasol",             "girasol",            16,   "Zenodo"),
            ("GOES-16 + NSRDB",     "goes16_nsrdb",       100,  "AWS S3 (public) + NREL API"),
            ("SolarNet",            "solarnet",           50,   "Zenodo"),
        ],
    }
    for domain, datasets in domains.items():
        print(f"  {domain}")
        for name, key, size, method in datasets:
            print(f"    {name:<30} {size:>6.1f} GB  [{method}]")
        print()
    print(f"  Total (3 domains): ~{sum(v for k,v in DATASET_SIZES_GB.items()):,.0f} GB")
    print(f"\n  Status file: {STATUS_FILE}")
    print(f"  Data root:   {DATA_ROOT}\n")


# ---------------------------------------------------------------------------
# Meteorology & Earth Observation
# ---------------------------------------------------------------------------

def download_meteonet(dest: Path, status: dict, dry_run: bool):
    """
    MeteoNet — ~90 GB
    Radar rain bands (5-min) + 500 ground stations (6-min)
    Source: https://meteonet.umr-cnrm.fr/dataset/

    MeteoNet hosts its data as per-year tar.gz archives split by zone and type.
    Actual URL structure (verified):
      /dataset/data/{zone}/{type}/{zone}_{type}_{year}.tar.gz
    Zones: NW, SE  |  Years: 2016, 2017, 2018
    Types: radar/rainfall, radar/reflectivity_new_product, ground_stations, masks
    """
    key = "meteonet"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download MeteoNet (~90 GB) to {dest}")
        return

    base_url = "https://meteonet.umr-cnrm.fr/dataset/data"

    # Real archive paths on the server (verified at meteonet.umr-cnrm.fr, 2024)
    # Reflectivity: 2016/2017 use old_product, 2018 uses new_product
    # Masks: .grib format (not tar.gz)
    # Each entry: (relative_path_on_server, local_filename, is_tar)
    archives = []
    for zone in ("NW", "SE"):
        for year in ("2016", "2017", "2018"):
            archives += [
                (f"{zone}/radar/rainfall/{zone}_rainfall_{year}.tar.gz",
                 f"{zone}_rainfall_{year}.tar.gz", True),
                (f"{zone}/ground_stations/{zone}_ground_stations_{year}.tar.gz",
                 f"{zone}_ground_stations_{year}.tar.gz", True),
            ]
            if year == "2018":
                archives.append(
                    (f"{zone}/radar/reflectivity_new_product/{zone}_reflectivity_new_product_{year}.tar.gz",
                     f"{zone}_reflectivity_new_product_{year}.tar.gz", True)
                )
            else:
                archives.append(
                    (f"{zone}/radar/reflectivity_old_product/{zone}_reflectivity_old_product_{year}.tar.gz",
                     f"{zone}_reflectivity_old_product_{year}.tar.gz", True)
                )
        archives.append(
            (f"{zone}/masks/{zone}_masks.grib", f"{zone}_masks.grib", False)
        )

    try:
        import requests  # type: ignore
        from tqdm import tqdm  # type: ignore
    except ImportError:
        pip_install("requests", "tqdm")
        import requests  # type: ignore
        from tqdm import tqdm  # type: ignore

    for rel_path, local_name, _is_tar in archives:
        out_path = dest / local_name
        if out_path.exists():
            print(f"    [exists] {local_name}")
            continue
        url = f"{base_url}/{rel_path}"
        print(f"    Downloading {local_name} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm(
                total=total or None, unit="B", unit_scale=True, desc=local_name
            ) as bar:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    bar.update(len(chunk))

    print(f"    Extracting archives ...")
    for _rel_path, local_name, is_tar in archives:
        if not is_tar:
            continue
        arc_path = dest / local_name
        if arc_path.exists():
            print(f"      Extracting {local_name} ...")
            with tarfile.open(arc_path, "r:gz") as tar:
                tar.extractall(dest)

    mark_done(status, key)
    print(f"  [done] MeteoNet")


def download_earthnet2021(dest: Path, status: dict, dry_run: bool):
    """
    EarthNet2021 — ~100 GB
    Sentinel-2 imagery (10m) + E-OBS weather time series
    Source: https://www.earthnet.tech/

    Uses the official `earthnet` Python package.
    """
    key = "earthnet2021"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download EarthNet2021 (~100 GB) to {dest}")
        return

    try:
        import earthnet  # type: ignore
    except ImportError:
        print("    Installing earthnet ...")
        pip_install("earthnet")
        import earthnet  # type: ignore

    print("    Downloading EarthNet2021 (all splits) ...")
    # API: Downloader.get(data_dir, splits)
    # splits can be "all" or a list: ["train", "iid", "ood", "extreme", "seasonal"]
    earthnet.Downloader.get(str(dest), "all")

    mark_done(status, key)
    print(f"  [done] EarthNet2021")


def download_era5(dest: Path, status: dict, dry_run: bool):
    """
    ERA5 EU subset — ~80 GB
    2 years (2020-2021), Europe bbox, 6 variables, 6-hourly, 0.25°
    Source: Copernicus CDS API — requires CDS_API_KEY env var

    Setup (new CDS, active since September 2024):
      1. Register at https://cds.climate.copernicus.eu/
      2. Accept the ERA5 license at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
      3. Copy your Personal Access Token from https://cds.climate.copernicus.eu/profile
      4. export CDS_API_KEY=<personal_access_token>
      5. The script writes ~/.cdsapirc automatically from env var

    NOTE: The legacy CDS API v2 (uid:token format, /api/v2 URL) was decommissioned
    on September 26, 2024. The new API uses a single Personal Access Token and
    the URL https://cds.climate.copernicus.eu/api (no /v2).
    """
    key = "era5_eu"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download ERA5 EU subset (~80 GB) to {dest}")
        return

    api_key = os.environ.get("CDS_API_KEY")
    if not api_key:
        print("  [WARN] CDS_API_KEY not set. Skipping ERA5.")
        print("         Register at https://cds.climate.copernicus.eu/")
        print("         Get your Personal Access Token at https://cds.climate.copernicus.eu/profile")
        print("         Then: export CDS_API_KEY=<personal_access_token>")
        return

    # Write ~/.cdsapirc if not present.
    # New CDS API (post Sep-2024): single PAT, URL without /v2.
    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        # Gracefully handle if someone passes the old uid:token format
        token = api_key.split(":", 1)[-1] if ":" in api_key else api_key
        cdsapirc.write_text(
            f"url: https://cds.climate.copernicus.eu/api\nkey: {token}\n"
        )
        print(f"    Wrote {cdsapirc}")

    try:
        import cdsapi  # type: ignore
    except ImportError:
        pip_install("cdsapi")
        import cdsapi  # type: ignore

    out_file = dest / "era5_eu_2020_2021.nc"
    if out_file.exists():
        print(f"    [exists] {out_file.name}")
        mark_done(status, key)
        return

    print("    Submitting ERA5 CDS request (may queue for minutes to hours) ...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "convective_available_potential_energy",
                "total_column_water_vapour",
                "surface_solar_radiation_downwards",
            ],
            "year":  ["2020", "2021"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  ["00:00", "06:00", "12:00", "18:00"],
            "area":  [72, -15, 35, 45],   # N, W, S, E — Europe bbox
            "format": "netcdf",
        },
        str(out_file),
    )

    mark_done(status, key)
    print(f"  [done] ERA5 EU subset -> {out_file}")


def download_rainbench(dest: Path, status: dict, dry_run: bool):
    """
    RainBench — ~20 GB
    Satellite + radar precipitation maps + ERA5-derived TS
    Source: Google Cloud Storage (access requires registration)

    RainBench data is NOT on Zenodo. It is hosted on Google Cloud Storage
    (bucket: gs://aaai_release) and requires requesting access via Google Form.

    Manual steps:
      1. Request access: https://github.com/FrontierDevelopmentLab/PyRain
      2. Once granted, install gsutil: pip install gsutil
      3. Download: gsutil -m cp -r gs://aaai_release/ {dest}
         Available resolutions: 5.625deg and 1.40625deg
    """
    key = "rainbench"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download RainBench (~20 GB) to {dest}")
        return

    print("  [MANUAL] RainBench requires access request — not auto-downloadable.")
    print("    1. Request access at: https://github.com/FrontierDevelopmentLab/PyRain")
    print("    2. Once granted:")
    print(f"       gsutil -m cp -r gs://aaai_release/ {dest}")
    print("  Skipping RainBench.")


# ---------------------------------------------------------------------------
# Mobility & Traffic
# ---------------------------------------------------------------------------

def download_gluonts_electricity(dest: Path, status: dict, dry_run: bool):
    """
    GluonTS Electricity — ~0.5 GB
    Classic electricity demand TS, TS-only (100% visual mask)
    Source: GluonTS Python library
    """
    key = "gluonts_electricity"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download GluonTS Electricity (~0.5 GB) to {dest}")
        return

    try:
        from gluonts.dataset.repository import get_dataset  # type: ignore
    except ImportError:
        pip_install("gluonts[arrow]")
        from gluonts.dataset.repository import get_dataset  # type: ignore

    print("    Downloading GluonTS electricity dataset ...")
    # GluonTS downloads to ~/.mxnet/gluon-ts/datasets/ by default; we copy to dest
    ds = get_dataset("electricity", regenerate=False)
    src = Path(ds.path)
    out_dir = dest / "electricity"
    if not out_dir.exists():
        import shutil
        shutil.copytree(str(src), str(out_dir))
        print(f"    Copied to {out_dir}")

    mark_done(status, key)
    print(f"  [done] GluonTS Electricity")


def _print_manual_download_notice(name: str, url: str, dest: Path, notes: str = ""):
    """Print a clearly formatted notice for datasets requiring manual download."""
    print(f"""
  [MANUAL REQUIRED] {name}
  ─────────────────────────────────────────────────────────────
  This dataset requires registration or a manual download step.
  Steps:
    1. Visit: {url}
    2. Register / request access if required
    3. Download all files to:
         {dest}
  {notes}
  After download, run:
    python scripts/download_phase1.py --mark-done {name.lower().replace(' ', '_')}
  ─────────────────────────────────────────────────────────────
""")


def download_i24_wavex(dest: Path, status: dict, dry_run: bool):
    """
    I-24 WaveX — ~50 GB
    High-fidelity camera observations + radar TS (30s cadence)
    Source: https://i24motion.org/wavex (requires account + token)
    """
    key = "i24_wavex"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download I-24 WaveX (~50 GB) to {dest}")
        return

    # WaveX uses a token-based download portal. Once you have the token:
    # Uncomment and fill in the token below, then re-run.
    #
    # token = os.environ.get("I24_WAVEX_TOKEN")
    # if not token:
    #     _print_manual_download_notice(...)
    #     return
    #
    # run(["wget", "--header", f"Authorization: Bearer {token}",
    #      "-r", "-np", "-nH", "--cut-dirs=2",
    #      "-P", str(dest), "https://i24motion.org/wavex/data/"])

    _print_manual_download_notice(
        name="I-24 WaveX",
        url="https://arxiv.org/html/2408.00941v1",
        dest=dest,
        notes=(
            "Follow the data access instructions in the paper appendix.\n"
            "  Set env var I24_WAVEX_TOKEN=<token> to enable automated download."
        ),
    )


def download_i24_motion(dest: Path, status: dict, dry_run: bool):
    """
    I-24 MOTION subset — ~100 GB
    6-month window, 2 camera arrays (~10 cameras), full resolution
    Source: https://i24motion.org/data (requires DUA agreement)
    """
    key = "i24_motion"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download I-24 MOTION subset (~100 GB) to {dest}")
        return

    _print_manual_download_notice(
        name="I-24 MOTION",
        url="https://i24motion.org/data",
        dest=dest,
        notes=(
            "Sign the Data Use Agreement (DUA) at the link above.\n"
            "  Subset: 6-month window, camera arrays CAM_01–CAM_10 only.\n"
            "  Use the portal's selective download to stay within ~100 GB."
        ),
    )


def download_marvel(dest: Path, status: dict, dry_run: bool):
    """
    MARVEL — ~30 GB
    CCTV images/video + pedestrian/vehicle counts + road network graph (Trento)
    Source: https://marvel-project.eu/ (academic request)
    """
    key = "marvel"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download MARVEL (~30 GB) to {dest}")
        return

    _print_manual_download_notice(
        name="MARVEL",
        url="https://marvel-project.eu/",
        dest=dest,
        notes=(
            "Submit an academic data access request via the project website.\n"
            "  Expected content: video/, counts/, graph/ subdirectories."
        ),
    )


# ---------------------------------------------------------------------------
# Solar / PV
# ---------------------------------------------------------------------------

def download_skippd(dest: Path, status: dict, dry_run: bool):
    """
    Source: https://huggingface.co/datasets/solarbench/SKIPPD
    """
    key = "skippd"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download SKIPP'D (~31 GB) to {dest}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pip_install("huggingface_hub")
        from huggingface_hub import snapshot_download

    repo_id = "solarbench/SKIPPD"
    print(f"    Downloading dataset {repo_id} to {dest} ...")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
    )

    mark_done(status, key)
    print(f"  [done] SKIPP'D")


def download_girasol_direct(dest: Path, status: dict):
    """
    Fallback direct download for Girasol dataset from DataDryad.
    Parses HTML page to extract download links without external dependencies.
    """
    try:
        import requests
        from tqdm import tqdm
        import re
        import time
    except ImportError:
        print(f"    [error] requests or tqdm not available")
        return

    dataset_url = "https://datadryad.org/dataset/doi:10.5061/dryad.zcrjdfn9m"
    print(f"    Fetching dataset page: {dataset_url}")
    
    # Set User-Agent to avoid 403 Forbidden
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    resp = requests.get(dataset_url, timeout=30, headers=headers)
    resp.raise_for_status()
    
    # Extract download links using regex
    # Pattern: <a class="js-individual-dl" href="/downloads/file_stream/XXXXX">...</a>
    # with filename inside or as text content
    file_links = []
    
    # Match: <a ... href="/downloads/file_stream/XXXXX">...filename.zip</a>
    # The filename is after any icon tags and before the closing </a>
    pattern = r'<a[^>]*href="(/downloads/file_stream/\d+)"[^>]*>(?:<i[^>]*></i>)?([^\<]+\.zip)'
    matches = re.finditer(pattern, resp.text, re.IGNORECASE)
    
    for match in matches:
        href = match.group(1)
        filename = match.group(2).strip()
        
        # Construct full URL
        if not href.startswith("http"):
            url = f"https://datadryad.org{href}"
        else:
            url = href
        
        file_links.append((filename, url))
    
    # Also try to find README.txt
    pattern_readme = r'<a[^>]*href="(/downloads/file_stream/\d+)"[^>]*>(?:<i[^>]*></i>)?(README\.txt)'
    matches = re.finditer(pattern_readme, resp.text, re.IGNORECASE)
    for match in matches:
        href = match.group(1)
        filename = match.group(2).strip()
        url = f"https://datadryad.org{href}" if not href.startswith("http") else href
        if (filename, url) not in file_links:
            file_links.append((filename, url))
    
    # Deduplicate by filename
    seen = set()
    unique_files = []
    for filename, url in file_links:
        if filename not in seen:
            seen.add(filename)
            unique_files.append((filename, url))
    
    print(f"    Found {len(unique_files)} files to download")
    
    if not unique_files:
        print(f"    [warning] No files found via regex patterns")
        return
    
    # Track overall progress
    pbar_overall = tqdm(total=len(unique_files), desc="Overall progress", unit="file")
    
    for i, (filename, url) in enumerate(unique_files, 1):
        out_path = dest / filename
        if out_path.exists():
            pbar_overall.update(1)
            continue
        
        pbar_overall.set_description(f"[{i}/{len(unique_files)}] {filename[:40]}")
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(out_path, "wb") as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"  {filename}", leave=False) as bar:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                                bar.update(len(chunk))
                    else:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"    [error] Failed to download {filename}: {e}")
            if out_path.exists():
                out_path.unlink()
        finally:
            pbar_overall.update(1)
        
        # Small delay between downloads to avoid rate limiting
        if i < len(unique_files):
            time.sleep(0.5)
    
    pbar_overall.close()
    print(f"  [done] Girasol (direct download)")
    mark_done(status, "girasol")


def download_girasol(dest: Path, status: dict, dry_run: bool):
    """
    Girasol — ~16 GB
    IR + visible fisheye images + Global Solar Irradiance TS
    Source: DataDryad dataset doi:10.5061/dryad.zcrjdfn9m
    Paper: https://www.researchgate.net/publication/349874802
    """
    key = "girasol"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download Girasol (~16 GB) to {dest}")
        return

    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        pip_install("requests", "tqdm")
        import requests
        from tqdm import tqdm

    # DataDryad API endpoint for the Girasol dataset
    # Dataset URL: https://datadryad.org/dataset/doi:10.5061/dryad.zcrjdfn9m
    # The full DOI (including "doi:" prefix) must be URL-encoded in the path:
    #   ":" → %3A, "/" → %2F
    dryad_api = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.zcrjdfn9m"

    print(f"    Fetching DataDryad dataset metadata for Girasol ...")
    resp = requests.get(dryad_api, timeout=30)
    
    # Fallback to direct download of all daily zip files if API fails
    if resp.status_code != 200:
        print(f"    [note] API unavailable, using direct download approach ...")
        download_girasol_direct(dest, status)
        return
    
    dataset = resp.json()

    # Extract file download URLs from the dataset metadata
    files = dataset.get("files", [])
    if not files:
        print(f"    [error] No files found in DataDryad dataset, using direct download...")
        download_girasol_direct(dest, status)
        return

    for file_info in files:
        filename = file_info.get("path", "")
        download_url = file_info.get("downloadHref", "")
        
        if not filename or not download_url:
            continue
        
        out_path = dest / filename
        if out_path.exists():
            print(f"    [exists] {filename}")
            continue
        
        size_bytes = file_info.get("size", 0)
        size_mb = size_bytes / 1e6
        print(f"    Downloading {filename} ({size_mb:.0f} MB) ...")
        
        with requests.get(download_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f, tqdm(
                total=size_bytes, unit="B", unit_scale=True, desc=filename
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

    mark_done(status, key)
    print(f"  [done] Girasol")


# ---------------------------------------------------------------------------
# NSRDB invalid-point cache — avoids re-requesting ocean/out-of-domain points
# ---------------------------------------------------------------------------

_NSRDB_SKIP_CACHE = Path("data/nsrdb_skip_points.json")


def _load_nsrdb_skip_points() -> set:
    if _NSRDB_SKIP_CACHE.exists():
        return {(round(p[0], 1), round(p[1], 1)) for p in json.loads(_NSRDB_SKIP_CACHE.read_text())}
    return set()


def _save_nsrdb_skip_points(skip: set) -> None:
    _NSRDB_SKIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _NSRDB_SKIP_CACHE.write_text(json.dumps(sorted(skip)))


def download_goes16_nsrdb(dest: Path, status: dict, dry_run: bool):
    """
    GOES-16 ABI + NSRDB multimodal dataset — ~100 GB

    Downloads GOES-16 ABI CONUS imagery and NSRDB irradiance time series for a
    0.5° CONUS grid, then aligns them into per-station Zarr stores ready for
    Vision-Time FM training.

    Output layout:
      {dest}/
        GOES16/{year}/{doy:03d}/{hour:02d}/*.nc   <- raw MCMIPC NetCDF files
        NSRDB/nsrdb_{year}_{lat}_{lon}.csv        <- irradiance TS per station
        paired/{lat}_{lon}.zarr/                  <- aligned (image, irradiance)

    Each paired Zarr store (Vision-Time FM Task 2.2 schema):
      images:          (T, 3, 64, 64) uint16  — C02 (visible), C09 (mid-IR), C13 (clean-IR)
      irradiance:      (T, 6) float32         — ghi, dni, dhi, wind_speed, air_temp, pressure
      timestamps:      (T,) int64             — Unix seconds UTC
      image_available: (T,) bool              — False when GOES scan is missing

    Sources:
      - AWS S3: s3://noaa-goes16 (public, no auth)
      - NREL NSRDB v4: https://nsrdb.nrel.gov (API key required)

    Setup:
      export NREL_API_KEY=<your_key>  (register at https://developer.nrel.gov/signup/)
    """
    key = "goes16_nsrdb"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return

    goes_dest   = dest / "GOES16"
    nsrdb_dest  = dest / "NSRDB"
    paired_dest = dest / "paired"
    ensure_dir(goes_dest)
    ensure_dir(nsrdb_dest)
    ensure_dir(paired_dest)

    if dry_run:
        print(f"  [dry-run] Would download GOES-16 ABI (~90 GB) to {goes_dest}")
        print(f"  [dry-run] Would download NSRDB (~10 GB) to {nsrdb_dest}")
        print(f"  [dry-run] Would build paired Zarr stores (~3 GB) to {paired_dest}")
        return

    year = "2021"
    _download_goes16_abi(goes_dest, year)

    nrel_key = os.environ.get("NREL_API_KEY")
    if not nrel_key:
        print("\n  [WARN] NREL_API_KEY not set. Skipping NSRDB download and pairing.")
    else:
        print("    Downloading NSRDB irradiance data for CONUS (2021) ...")
        _download_nsrdb(nsrdb_dest, nrel_key, year=year)
        print("    Building paired multimodal Zarr stores ...")
        _pair_goes16_nsrdb(goes_dest, nsrdb_dest, paired_dest, year=year)

    mark_done(status, key)
    print(f"  [done] GOES-16 + NSRDB -> paired multimodal dataset at {paired_dest}")


def _download_goes16_abi(dest: Path, year: str = "2021") -> None:
    """
    Download GOES-16 ABI MCMIPC (CONUS) NetCDF files from the public AWS S3 bucket.
    Samples every 6th scan (~hourly) to stay within ~90 GB.
    Adjust STEP=1 for full 10-min cadence (~300 GB).
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from boto3.s3.transfer import TransferConfig
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
    except ImportError:
        pip_install("boto3")
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from boto3.s3.transfer import TransferConfig
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

    from tqdm import tqdm as _tqdm

    print("    Downloading GOES-16 ABI (CONUS, 2021, MCMIPC, hourly sample) ...")

    MAX_WORKERS = 32
    STEP        = 6       # every 6th scan ≈ hourly; set to 1 for full 10-min cadence
    bucket      = "noaa-goes16"
    product     = "ABI-L2-MCMIPC"

    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, max_pool_connections=MAX_WORKERS),
        region_name="us-east-1",
    )
    transfer_cfg = TransferConfig(
        multipart_threshold=50 * 1024 * 1024,
        multipart_chunksize=16 * 1024 * 1024,
        max_concurrency=10,
    )

    _created_dirs: set = set()
    _dir_lock = threading.Lock()

    def ensure_out_dir(p: Path):
        parent = p.parent
        with _dir_lock:
            if parent not in _created_dirs:
                parent.mkdir(parents=True, exist_ok=True)
                _created_dirs.add(parent)

    def _list_hour(doy: int, hour: int) -> list:
        prefix = f"{product}/{year}/{doy:03d}/{hour:02d}/"
        paginator = s3.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            files.extend(page.get("Contents", []))
        return [f for i, f in enumerate(files) if i % STEP == 0]

    def _download_one(obj: dict) -> tuple:
        try:
            s3_key   = obj["Key"]
            fname    = Path(s3_key).name
            out_path = dest / year / f"{obj['doy']:03d}" / f"{obj['hour']:02d}" / fname
            ensure_out_dir(out_path)
            if out_path.exists():
                return (f"SKIP:{fname}", False)
            s3.download_file(bucket, s3_key, str(out_path), Config=transfer_cfg)
            return (fname, True)
        except Exception as e:
            fname_safe = Path(obj.get("Key", "unknown")).name
            return (f"FAILED:{fname_safe}:{str(e)[:60]}", False)

    # Phase 1: parallel S3 listing
    print("    [Phase 1] Listing S3 objects ...")
    all_objects = []
    all_tasks   = [(doy, hour) for doy in range(1, 366) for hour in range(24)]

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_list_hour, d, h): (d, h) for d, h in all_tasks}
        pbar_list = _tqdm(total=len(futures), desc="Listing hours", unit="hour")
        for future in as_completed(futures):
            doy, hour = futures[future]
            try:
                objs = future.result()
                for obj in objs:
                    obj["doy"] = doy
                    obj["hour"] = hour
                all_objects.extend(objs)
            except Exception as e:
                print(f"    [warn] Failed to list DOY {doy} hour {hour}: {e}")
            pbar_list.update(1)
        pbar_list.close()

    print(f"    Found {len(all_objects)} objects.")

    # Phase 2: parallel downloads
    print(f"    [Phase 2] Downloading files (MAX_WORKERS={MAX_WORKERS}) ...")
    downloaded = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_download_one, obj): obj["Key"] for obj in all_objects}
        pbar_dl = _tqdm(total=len(futures), desc="GOES-16", unit="file")
        for future in as_completed(futures):
            try:
                result, is_new = future.result()
                if result.startswith("SKIP:"):
                    skipped += 1
                elif result.startswith("FAILED:"):
                    failed += 1
                    pbar_dl.write(f"    [warn] {result}")
                else:
                    downloaded += 1
                pbar_dl.set_description(f"GOES-16 (D:{downloaded} S:{skipped} F:{failed})")
            except Exception as e:
                failed += 1
                pbar_dl.write(f"    [warn] {e}")
            pbar_dl.update(1)
        pbar_dl.close()

    print(f"    GOES-16 complete: {downloaded} new + {skipped} existing + {failed} failed -> {dest}")


def _download_nsrdb(dest: Path, api_key: str, year: str = "2021") -> None:
    """
    Download NSRDB PSM v4 data for a 0.5° CONUS grid (~10 GB).

    Points returning HTTP 400 (ocean / out-of-domain) are persisted in
    data/nsrdb_skip_points.json so they are never re-requested on resume.
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        pip_install("requests", "tqdm")
        import requests
        from tqdm import tqdm

    # 0.5° grid over CONUS: lat 24.5–49.5°N, lon -124.5–-67°W
    lats = [round(24.5 + i * 0.5, 1) for i in range(51)]
    lons = [round(-124.5 + j * 0.5, 1) for j in range(116)]

    base_url = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
    params_template = {
        "api_key":      api_key,
        "wkt":          "",
        "names":        year,
        "leap_day":     "false",
        "interval":     "60",
        "utc":          "true",
        "email":        os.environ.get("NREL_EMAIL", "user@example.com"),
        "affiliation":  "Research",
        "mailing_list": "false",
        "reason":       "Research",
        "attributes":   "ghi,dni,dhi,wind_speed,air_temperature,surface_pressure",
    }

    # Load persisted skip-list (ocean / out-of-domain points found in prior runs)
    skip_pts  = _load_nsrdb_skip_points()
    new_skips: set = set()

    total_points = len(lats) * len(lons)
    pre_skipped  = sum(1 for lat in lats for lon in lons if (lat, lon) in skip_pts)
    print(f"    {total_points} grid points ({pre_skipped} pre-skipped from cache) ...")

    pbar = tqdm(total=total_points, desc="NSRDB", unit="point")
    for lat in lats:
        for lon in lons:
            if (lat, lon) in skip_pts:
                pbar.update(1)
                continue

            fname = dest / f"nsrdb_{year}_{lat:.1f}_{lon:.1f}.csv"
            if fname.exists():
                pbar.update(1)
                continue

            pbar.set_description(f"NSRDB ({lat:.1f}°N, {lon:.1f}°W)")
            params = {**params_template, "wkt": f"POINT({lon} {lat})"}
            try:
                r = requests.get(base_url, params=params, timeout=30)
                if r.status_code == 200:
                    fname.write_bytes(r.content)
                elif r.status_code == 400:
                    # Ocean or outside NSRDB domain — never retry
                    new_skips.add((lat, lon))
                else:
                    pbar.write(f"      [warn] {lat},{lon}: HTTP {r.status_code}")
            except Exception as e:
                pbar.write(f"      [warn] {lat},{lon}: {e}")
            pbar.update(1)
            time.sleep(1.1)  # NREL rate limit: 1 req/s

            if len(new_skips) % 50 == 0 and new_skips:
                _save_nsrdb_skip_points(skip_pts | new_skips)

    pbar.close()
    if new_skips:
        _save_nsrdb_skip_points(skip_pts | new_skips)
        print(f"    Cached {len(new_skips)} ocean/invalid points to {_NSRDB_SKIP_CACHE}.")


def _pair_goes16_nsrdb(
    goes_dest: Path,
    nsrdb_dest: Path,
    paired_dest: Path,
    year: str = "2021",
    patch_size: int = 64,
    channels: tuple = ("CMI_C02", "CMI_C09", "CMI_C13"),
    batch_size: int = 100,
) -> None:
    """
    Align GOES-16 ABI imagery with NSRDB irradiance CSVs into per-station Zarr stores.

    Uses a SCAN-CENTRIC approach: iterates over GOES-16 files once per batch and
    extracts patches for all stations in the batch simultaneously.

      Naive approach:   O(N_stations x T_nsrdb) file opens  <- 35M opens ~500 h
      This approach:    O(ceil(N / batch_size) x T_goes)    <- ~35k opens  ~2 h

    Memory per batch: batch_size x T x 3 x patch_size^2 x 2 bytes ~200 MB/station.
    Default batch_size=100 needs ~20 GB RAM peak. Lower it on smaller machines.

    New dependencies: zarr, netCDF4, pyproj, pandas (auto-installed if missing).
    """
    try:
        import numpy as np
        import zarr
        import netCDF4 as nc4
        import pandas as pd
        from pyproj import CRS, Transformer
        from tqdm import tqdm
        from collections import defaultdict
    except ImportError:
        pip_install("numpy", "zarr", "netCDF4", "pyproj", "pandas", "tqdm")
        import numpy as np
        import zarr
        import netCDF4 as nc4
        import pandas as pd
        from pyproj import CRS, Transformer
        from tqdm import tqdm
        from collections import defaultdict

    from datetime import datetime, timedelta, timezone

    IRRAD_COLS = ["GHI", "DNI", "DHI", "Wind Speed", "Temperature", "Pressure"]

    # -- Read GOES-16 projection from the first available NetCDF
    nc_files = sorted(goes_dest.glob(f"{year}/**/*.nc"))
    if not nc_files:
        print("    [warn] No GOES-16 NetCDF files found; skipping pairing.")
        return

    with nc4.Dataset(nc_files[0]) as ds:
        pv    = ds.variables["goes_imager_projection"]
        sat_h = float(pv.perspective_point_height)
        lon_0 = float(pv.longitude_of_projection_origin)
        sweep = str(pv.sweep_angle_axis)
        x_rad = np.array(ds.variables["x"][:])  # scan angles (rad), shape (ncols,)
        y_rad = np.array(ds.variables["y"][:])  # shape (nrows,)

    crs_geos = CRS.from_proj4(
        f"+proj=geos +h={sat_h} +lon_0={lon_0} +sweep={sweep} +ellps=GRS80"
    )
    to_geos = Transformer.from_crs("EPSG:4326", crs_geos, always_xy=True)

    half  = patch_size // 2
    nrows = len(y_rad)
    ncols = len(x_rad)

    def _latlon_to_pixel(lat: float, lon: float):
        x_m, y_m = to_geos.transform(lon, lat)
        col = int(np.argmin(np.abs(x_rad - x_m / sat_h)))
        row = int(np.argmin(np.abs(y_rad - y_m / sat_h)))
        if row < half or row >= nrows - half or col < half or col >= ncols - half:
            return None
        return row, col

    # -- Build sorted GOES timestamp index
    print("    Building GOES-16 file index ...")
    ts_unix:  list = []
    nc_paths: list = []
    for nc_path in nc_files:
        fname = nc_path.name
        try:
            s_idx  = fname.index("_s") + 2
            ts_str = fname[s_idx : s_idx + 13]  # YYYYDDDHHMMSSs
            dt = datetime(int(ts_str[0:4]), 1, 1,
                          int(ts_str[7:9]), int(ts_str[9:11]),
                          tzinfo=timezone.utc)
            dt += timedelta(days=int(ts_str[4:7]) - 1)
            ts_unix.append(dt.timestamp())
            nc_paths.append(nc_path)
        except Exception:
            continue

    order      = sorted(range(len(ts_unix)), key=lambda i: ts_unix[i])
    ts_arr     = np.array([ts_unix[i] for i in order], dtype=np.float64)
    sorted_ncs = [nc_paths[i] for i in order]
    print(f"    Indexed {len(ts_arr)} GOES-16 scans.")

    def _nearest_goes_idx(row_ts: float) -> int:
        """Return index into sorted_ncs for nearest scan, or -1 if >10 min away."""
        idx      = int(np.searchsorted(ts_arr, row_ts))
        best_idx = idx
        if idx > 0 and (
            idx >= len(ts_arr)
            or abs(ts_arr[idx - 1] - row_ts) < abs(ts_arr[idx] - row_ts)
        ):
            best_idx = idx - 1
        if best_idx >= len(ts_arr) or abs(ts_arr[best_idx] - row_ts) > 600:
            return -1
        return best_idx

    # -- Load all valid stations from downloaded NSRDB CSVs
    nsrdb_csvs = sorted(nsrdb_dest.glob(f"nsrdb_{year}_*.csv"))
    if not nsrdb_csvs:
        print("    [warn] No NSRDB CSVs found; skipping pairing.")
        return

    print(f"    Loading {len(nsrdb_csvs)} NSRDB stations ...")
    all_stations = []
    for csv_path in tqdm(nsrdb_csvs, desc="Loading CSVs", unit="station"):
        stem_parts = csv_path.stem.split("_")
        try:
            lat, lon = float(stem_parts[2]), float(stem_parts[3])
        except (IndexError, ValueError):
            continue
        if (paired_dest / f"{lat:.1f}_{lon:.1f}.zarr").exists():
            continue  # already done in a previous run
        pixel = _latlon_to_pixel(lat, lon)
        if pixel is None:
            continue
        try:
            df = pd.read_csv(csv_path, skiprows=2)
            df["timestamp"] = pd.to_datetime(
                df[["Year", "Month", "Day", "Hour", "Minute"]]
            ).dt.tz_localize("UTC")
            df = df.set_index("timestamp").sort_index()
        except Exception as e:
            tqdm.write(f"      [warn] Cannot parse {csv_path.name}: {e}")
            continue

        unix_ts = (df.index.astype(np.int64) // 10 ** 9).to_numpy()
        irrad   = np.zeros((len(unix_ts), len(IRRAD_COLS)), dtype=np.float32)
        for j, col_name in enumerate(IRRAD_COLS):
            if col_name in df.columns:
                irrad[:, j] = df[col_name].to_numpy(dtype=np.float32, na_value=0.0)

        # Pre-compute nearest GOES file index per NSRDB row (O(T log N_goes))
        goes_row_idx = np.array([_nearest_goes_idx(t) for t in unix_ts], dtype=np.int32)

        all_stations.append({
            "lat": lat, "lon": lon, "pixel": pixel,
            "unix_ts": unix_ts, "irrad": irrad, "goes_row_idx": goes_row_idx,
        })

    if not all_stations:
        print("    [info] All stations already paired; nothing to do.")
        return

    # -- Scan-centric extraction: one pass over GOES files per batch
    n_batches   = (len(all_stations) + batch_size - 1) // batch_size
    mem_est_gb  = batch_size * 8760 * 3 * patch_size * patch_size * 2 / 1e9
    print(f"    {len(all_stations)} stations -> {n_batches} batch(es) "
          f"(batch_size={batch_size}, ~{mem_est_gb:.1f} GB RAM/batch)")

    for b_idx in range(n_batches):
        batch = all_stations[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(f"\n    [Batch {b_idx + 1}/{n_batches}] {len(batch)} stations")

        # Allocate in-memory image buffers for this batch
        for s in batch:
            T = len(s["unix_ts"])
            s["images"]          = np.zeros((T, 3, patch_size, patch_size), dtype=np.uint16)
            s["image_available"] = np.zeros(T, dtype=bool)

        # Reverse map: goes_file_idx -> [(station_in_batch_idx, nsrdb_row_idx)]
        reverse_map: dict = defaultdict(list)
        for si, s in enumerate(batch):
            for row_idx, gi in enumerate(s["goes_row_idx"]):
                if gi >= 0:
                    reverse_map[gi].append((si, row_idx))

        # One sequential pass over needed GOES files -- each opened exactly ONCE
        needed = sorted(reverse_map.keys())
        for gi in tqdm(needed, desc=f"  Batch {b_idx + 1} scans", unit="scan"):
            try:
                with nc4.Dataset(sorted_ncs[gi]) as ds:
                    bands = [
                        np.ma.filled(ds.variables[ch][:], 0).astype(np.uint16)
                        for ch in channels
                    ]
            except Exception as e:
                tqdm.write(f"      [warn] {sorted_ncs[gi].name}: {e}")
                continue

            for si, row_idx in reverse_map[gi]:
                p_row, p_col = batch[si]["pixel"]
                for ci, band in enumerate(bands):
                    batch[si]["images"][row_idx, ci] = (
                        band[p_row - half : p_row + half, p_col - half : p_col + half]
                    )
                batch[si]["image_available"][row_idx] = True

        # Write Zarr stores and release memory before the next batch
        for s in batch:
            T        = len(s["unix_ts"])
            out_zarr = paired_dest / f"{s['lat']:.1f}_{s['lon']:.1f}.zarr"
            store    = zarr.open_group(str(out_zarr), mode="w")
            store.create_dataset("images",          data=s["images"],
                                 chunks=(min(T, 168), 3, patch_size, patch_size))
            store.create_dataset("irradiance",      data=s["irrad"],
                                 chunks=(min(T, 8760), len(IRRAD_COLS)))
            store.create_dataset("image_available", data=s["image_available"],
                                 chunks=(min(T, 8760),))
            store.create_dataset("timestamps",      data=s["unix_ts"],
                                 chunks=(min(T, 8760),))
            store.attrs.update({
                "lat": s["lat"], "lon": s["lon"], "year": year,
                "channels": list(channels), "patch_size": patch_size,
                "irradiance_columns": IRRAD_COLS, "schema": "vision_time_fm_v1",
            })
            del s["images"], s["image_available"]  # free RAM before next batch

    print(f"    Paired Zarr stores written to {paired_dest}")


def download_solarnet(dest: Path, status: dict, dry_run: bool):
    """
    SolarNet — ~50 GB
    Sky camera imagery + pyranometer irradiance TS (Folsom, CA, 2014-2016)
    Source: Zenodo record 2826939
    DOI: https://doi.org/10.5281/zenodo.2826939
    """
    key = "solarnet"
    if is_done(status, key):
        print(f"  [skip] {key} already downloaded")
        return
    ensure_dir(dest)
    if dry_run:
        print(f"  [dry-run] Would download SolarNet (~50 GB) to {dest}")
        return

    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        pip_install("requests", "tqdm")
        import requests
        from tqdm import tqdm

    zenodo_record = "2826939"
    zenodo_api = f"https://zenodo.org/api/records/{zenodo_record}"

    print(f"    Fetching Zenodo record {zenodo_record} for SolarNet ...")
    resp = requests.get(zenodo_api, timeout=30)
    resp.raise_for_status()
    record = resp.json()

    for file_info in record["files"]:
        filename = file_info["key"]
        url = file_info["links"]["self"]
        out_path = dest / filename
        if out_path.exists():
            print(f"    [exists] {filename}")
            continue
        size_mb = file_info["size"] / 1e6
        print(f"    Downloading {filename} ({size_mb:.0f} MB) ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f, tqdm(
                total=file_info["size"], unit="B", unit_scale=True, desc=filename
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

    mark_done(status, key)
    print(f"  [done] SolarNet")


# ---------------------------------------------------------------------------
# Domain runners
# ---------------------------------------------------------------------------

def run_meteorology(status: dict, dry_run: bool):
    dest_base = STORAGE_LAYOUT["meteorology"]
    print("\n=== Meteorology & Earth Observation (~290 GB) ===")
    download_meteonet(dest_base / "meteonet",       status, dry_run)
    download_earthnet2021(dest_base / "earthnet2021", status, dry_run)
    download_era5(dest_base / "era5",               status, dry_run)
    download_rainbench(dest_base / "rainbench",     status, dry_run)


def run_mobility(status: dict, dry_run: bool):
    dest_base = STORAGE_LAYOUT["mobility"]
    print("\n=== Mobility & Traffic (~200 GB) ===")
    download_i24_wavex(dest_base / "i24_wavex",     status, dry_run)
    download_i24_motion(dest_base / "i24_motion",   status, dry_run)
    download_marvel(dest_base / "marvel",           status, dry_run)
    download_gluonts_electricity(dest_base / "electricity", status, dry_run)


def run_solar(status: dict, dry_run: bool):
    dest_base = STORAGE_LAYOUT["solar"]
    print("\n=== Solar / PV (~160 GB) ===")
    download_skippd(dest_base / "skippd",           status, dry_run)
    download_girasol(dest_base / "girasol",         status, dry_run)
    download_goes16_nsrdb(dest_base / "goes16_nsrdb", status, dry_run)
    download_solarnet(dest_base / "solarnet",       status, dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 dataset downloader — Meteorology, Mobility, Solar (~650 GB)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without downloading")
    parser.add_argument("--domain", choices=["meteorology", "mobility", "solar"], help="Download a single domain")
    parser.add_argument("--all", action="store_true", help="Download all three domains")
    parser.add_argument("--data-root", type=Path, default=None, help="Override DATA_ROOT (default: data/raw)")
    parser.add_argument(
        "--mark-done", metavar="DATASET_KEY",
        help="Manually mark a dataset as done (use after manual download)"
    )
    args = parser.parse_args()

    if args.data_root:
        global DATA_ROOT
        DATA_ROOT = args.data_root
        for k in STORAGE_LAYOUT:
            STORAGE_LAYOUT[k] = DATA_ROOT / k

    if args.dry_run and not args.domain and not args.all and not args.mark_done:
        dry_run_summary()
        return

    status = load_status()

    if args.mark_done:
        mark_done(status, args.mark_done)
        print(f"Marked '{args.mark_done}' as done in {STATUS_FILE}")
        return

    if args.dry_run:
        print("[DRY RUN MODE — no files will be downloaded]\n")

    if args.domain == "meteorology" or args.all:
        run_meteorology(status, args.dry_run)

    if args.domain == "mobility" or args.all:
        run_mobility(status, args.dry_run)

    if args.domain == "solar" or args.all:
        run_solar(status, args.dry_run)

    if not args.domain and not args.all:
        parser.print_help()


if __name__ == "__main__":
    main()

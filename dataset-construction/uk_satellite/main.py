import os
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
from pyproj import Transformer
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import numcodecs
import ocf_blosc2

# Register the blosc2 codec
numcodecs.registry.register_codec(ocf_blosc2.Blosc2)

# Load Hugging Face token from environment / .env file
load_dotenv()

def compile_local_dataset():
    # 1. Define Local Dataset Scope (Adjust these to scale your download)
    output_filename = "uk_pv_local_paired_dataset.nc"
    num_pv_systems = 100
    start_time = "2019-01-01T00:00:00"
    end_time = "2020-12-31T23:30:00"
    patch_radius = 16  # Results in a 32x32 pixel satellite crop (approx. 32km x 32km)
    time_batch_size = 128  # Reduced batch size to cap peak RAM on 16GB machines (was 2000).
    hf_repo_id = "openclimatefix/uk_pv"
    hf_repo_type = "dataset"
    pv_data_folder = "30_minutely"
    pv_timestamp_frequency = "30min"
    generation_multiplier = 2.0  # Convert Wh in a 30-minute interval to average Watts.
    hf_token = os.environ.get("HF_TOKEN") or True

    print("--- Step 1: Loading PV Metadata and Yield from Hugging Face ---")
    # Load the published CSV/parquet files directly from the dataset repository.
    try:
        metadata_path = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type=hf_repo_type,
            filename="metadata.csv",
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to download the gated openclimatefix/uk_pv dataset. Set HF_TOKEN to a Hugging Face access token with dataset permission, or run huggingface-cli login and accept the dataset terms."
        ) from exc

    pv_metadata_df = pd.read_csv(metadata_path)

    # Download and load all yield data files for 2019 and 2020
    pv_yield_dfs = []
    years_to_download = [2019, 2020]
    for year in years_to_download:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            parquet_filename = f"{pv_data_folder}/year={year}/month={month_str}/data.parquet"
            print(f" -> Downloading yield data for {year}-{month_str}...")
            try:
                yield_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    repo_type=hf_repo_type,
                    filename=parquet_filename,
                    token=hf_token,
                )
                month_df = pd.read_parquet(yield_path, columns=["ss_id", "datetime_GMT", "generation_Wh"])
                pv_yield_dfs.append(month_df)
            except Exception as exc:
                print(f"      Warning: Could not download {parquet_filename}: {exc}")

    if not pv_yield_dfs:
        raise RuntimeError("No yield data could be downloaded.")
    
    print(" -> Concatenating yield data...")
    pv_yield_df = pd.concat(pv_yield_dfs, ignore_index=True)

    # Clean up timestamps in the yield data.
    pv_yield_df["datetime_GMT"] = pd.to_datetime(pv_yield_df["datetime_GMT"])
    
    # Select the target PV system IDs to download (most active first)
    active_pv_ids = pv_yield_df["ss_id"].value_counts().index
    target_pv_ids = [pid for pid in active_pv_ids if pid in pv_metadata_df["ss_id"].values][:num_pv_systems]
    filtered_metadata = pv_metadata_df[pv_metadata_df["ss_id"].isin(target_pv_ids)]
    
    # Generate the continuous target timestamp array (30-minute frequency, daytime only).
    all_timestamps = pd.date_range(start=start_time, end=end_time, freq=pv_timestamp_frequency)
    target_timestamps = all_timestamps[all_timestamps.indexer_between_time("08:00", "16:00")]
    print(f"Targeting {len(target_pv_ids)} PV systems across {len(target_timestamps)} timestamps.")

    print("\n--- Step 2: Initializing Cloud Satellite Stream ---")
    # Establish anonymous connection to the OCF EUMETSAT HRV Zarr bucket
    fs = gcsfs.GCSFileSystem(token='anon')
    
    sat_datasets = []
    for year in years_to_download:
        zarr_bucket_path = f'gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{year}_hrv.zarr'
        print(f" -> Opening satellite stream for {year}...")
        mapper = fs.get_mapper(zarr_bucket_path)
        ds = xr.open_zarr(mapper, consolidated=True, chunks={})
        sat_datasets.append(ds)
    
    print(" -> Combining satellite streams...")
    sat_ds = xr.concat(sat_datasets, dim="time", join="override")
    
    # Identify the primary data variable inside the satellite dataset dynamically
    sat_var_name = list(sat_ds.data_vars)[0]

    print("\n--- Step 3: Setting Up Coordinate Projection Transformer ---")
    # Extract geostationary projection attributes from metadata to handle coordinate warping
    sat_crs_attrs = sat_ds.attrs.get('projection', {
        'proj': 'geos', 'lon_0': 9.5, 'h': 35785831, 'x_0': 0, 'y_0': 0, 'a': 6378169, 'rf': 295.488065897014
    })
    proj_string = (
        f"+proj={sat_crs_attrs['proj']} +lon_0={sat_crs_attrs['lon_0']} +h={sat_crs_attrs['h']} "
        f"+x_0={sat_crs_attrs['x_0']} +y_0={sat_crs_attrs['y_0']} +a={sat_crs_attrs['a']} +rf={sat_crs_attrs['rf']}"
    )
    # Transformer from GPS (EPSG:4326) to EUMETSAT Geostationary space
    geo_transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    print("\n--- Step 4: Extracting and Aligning Paired Batches (Chunked by Plant) ---")
    import shutil
    parts_dir = "paired_dataset_parts"
    os.makedirs(parts_dir, exist_ok=True)

    # --- Precompute once, reused across all plants (huge speedup) ---
    sat_var = sat_ds[sat_var_name]
    n_x = sat_ds.sizes["x_geostationary"]
    n_y = sat_ds.sizes["y_geostationary"]
    x_coords = sat_ds.x_geostationary.values
    y_coords = sat_ds.y_geostationary.values

    # Nearest satellite-time index for each target timestamp. Same for every plant,
    # so compute it ONCE instead of re-running sel(method='nearest') 100 times.
    print(" -> Precomputing nearest satellite time indices...")
    time_idx = sat_ds.indexes["time"].get_indexer(target_timestamps, method="nearest")
    n_t = len(target_timestamps)

    # Group yield once instead of boolean-masking the full frame per plant.
    yield_groups = pv_yield_df.set_index("datetime_GMT").groupby("ss_id")

    patch_size = 2 * patch_radius
    import time as _time

    # --- Precompute each plant's clamped pixel window once ---
    # (raw negative slice starts would silently wrap and corrupt the crop)
    plant_info = []  # (pv_id, x0, y0)
    for pv_id in target_pv_ids:
        meta = filtered_metadata[filtered_metadata["ss_id"] == pv_id].iloc[0]
        sat_x, sat_y = geo_transformer.transform(meta["longitude_rounded"], meta["latitude_rounded"])
        x_idx = int(np.abs(x_coords - sat_x).argmin())
        y_idx = int(np.abs(y_coords - sat_y).argmin())
        x0 = max(0, min(x_idx - patch_radius, n_x - patch_size))
        y0 = max(0, min(y_idx - patch_radius, n_y - patch_size))
        plant_info.append((pv_id, x0, y0))

    # Global bounding box covering every plant's patch -> we download this ONE
    # region per time-batch and slice all 100 patches out of it locally. This is
    # the ~100x win: each satellite frame is fetched once, not once-per-plant.
    bx0 = min(x0 for _, x0, _ in plant_info)
    bx1 = max(x0 + patch_size for _, x0, _ in plant_info)
    by0 = min(y0 for _, _, y0 in plant_info)
    by1 = max(y0 + patch_size for _, _, y0 in plant_info)
    region = sat_var.isel(
        x_geostationary=slice(bx0, bx1),
        y_geostationary=slice(by0, by1),
    ).squeeze(drop=True).transpose("time", "y_geostationary", "x_geostationary")
    print(f" -> Shared download window: {by1-by0} x {bx1-bx0} px covering {len(plant_info)} plants")

    # Per-plant patch buffers live on disk as .npy memmaps. They double as the
    # resume checkpoint: data already written survives a crash/kill.
    memmaps = {}
    for pv_id, _, _ in plant_info:
        mpath = os.path.join(parts_dir, f"pv_{pv_id}.patches.npy")
        mode = "r+" if os.path.exists(mpath) else "w+"
        memmaps[pv_id] = np.lib.format.open_memmap(
            mpath, mode=mode, dtype=np.float32, shape=(n_t, patch_size, patch_size)
        )

    # Resume marker: how many leading time-batches are already fully written.
    progress_path = os.path.join(parts_dir, "_progress.txt")
    done_batches = 0
    if os.path.exists(progress_path):
        try:
            done_batches = int(open(progress_path).read().strip())
        except Exception:
            done_batches = 0

    n_batches = (n_t + time_batch_size - 1) // time_batch_size
    print(f" -> Streaming {n_batches} time-batches of {time_batch_size} frames "
          f"(resuming from batch {done_batches})...")

    for b, start in enumerate(range(0, n_t, time_batch_size)):
        if b < done_batches:
            continue
        end = min(start + time_batch_size, n_t)
        t0 = _time.time()
        # ONE download for all plants in this time slice
        block = region.isel(time=time_idx[start:end]).values  # (tb, Y, X)
        block = np.ascontiguousarray(block, dtype=np.float32)
        for pv_id, x0, y0 in plant_info:
            ly, lx = y0 - by0, x0 - bx0
            memmaps[pv_id][start:end] = block[:, ly:ly + patch_size, lx:lx + patch_size]
        for mm in memmaps.values():
            mm.flush()
        with open(progress_path, "w") as f:
            f.write(str(b + 1))
        dt = _time.time() - t0
        eta = dt * (n_batches - b - 1)
        print(f"      batch {b+1}/{n_batches} ({end-start} frames) in {dt:.0f}s "
              f"| ~{eta/60:.1f} min left total")

    # --- Assemble per-plant NetCDF parts from the completed memmaps ---
    print(" -> Assembling per-plant NetCDF part files...")
    for pv_id, _, _ in plant_info:
        part_filename = os.path.join(parts_dir, f"pv_{pv_id}.nc")
        if os.path.exists(part_filename):
            continue
        try:
            system_yield = yield_groups.get_group(pv_id).reindex(target_timestamps)
            gen = system_yield["generation_Wh"].values
        except KeyError:
            gen = np.full(n_t, np.nan)
        pv_dataset_node = xr.Dataset(
            data_vars={
                "pv_generation": (["time"], (gen * generation_multiplier).astype(np.float32)),
                "satellite_hrv": (["time", "y_patch", "x_patch"], np.asarray(memmaps[pv_id])),
            },
            coords={
                "time": target_timestamps,
                "y_patch": np.arange(-patch_radius, patch_radius),
                "x_patch": np.arange(-patch_radius, patch_radius),
            },
        ).expand_dims(pv_id=[pv_id])
        tmp_filename = part_filename + ".tmp"
        pv_dataset_node.to_netcdf(tmp_filename)
        os.replace(tmp_filename, part_filename)

    # Drop memmap refs + delete raw patch buffers now that parts are written
    for pv_id in list(memmaps):
        memmaps[pv_id].flush()
    del memmaps
    import gc
    gc.collect()
    for pv_id, _, _ in plant_info:
        mpath = os.path.join(parts_dir, f"pv_{pv_id}.patches.npy")
        if os.path.exists(mpath):
            os.remove(mpath)
    if os.path.exists(progress_path):
        os.remove(progress_path)

    print("\n--- Step 5: Merging and Saving Dataset to Local Storage ---")
    part_files = [os.path.join(parts_dir, f"pv_{pid}.nc") for pid in target_pv_ids if os.path.exists(os.path.join(parts_dir, f"pv_{pid}.nc"))]
    print(f"Merging {len(part_files)} part files...")
    if part_files:
        final_dataset = xr.open_mfdataset(part_files, combine="nested", concat_dim="pv_id")
        final_dataset.to_netcdf(output_filename)
        print(f"Success! Paired dataset compiled and downloaded to local path: '{os.path.abspath(output_filename)}'")
        # Clean up temporary part files
        try:
            shutil.rmtree(parts_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {parts_dir}: {e}")
    else:
        print("No part files found to merge.")

if __name__ == "__main__":
    compile_local_dataset()
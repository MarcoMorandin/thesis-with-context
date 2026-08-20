import xarray as xr
import numpy as np
import pandas as pd
from PIL import Image
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_patch(args):
    pv_id, patch, pv_val, t_str, t_iso, out_img_dir = args
    patch = np.nan_to_num(patch, nan=0.0)
    patch = np.clip(patch, 0.0, 1.0)
    img_arr = (patch * 255).astype(np.uint8)
    
    dataset_site_name = f"uk_pv_{pv_id}"
    site_dir = os.path.join(out_img_dir, dataset_site_name)
    # the dir is created beforehand
        
    img_filename = f"{t_str}.png"
    img_path_rel = f"images/{dataset_site_name}/{img_filename}"
    img_path_abs = os.path.join(site_dir, img_filename)
    
    img = Image.fromarray(img_arr)
    img.save(img_path_abs)
    
    return {
        'dataset': 'uk_pv',
        'site_id': str(pv_id),
        'station_id': 'uk',
        'camera_id': 'hrv',
        'timestamp_utc': t_iso,
        'power_w': float(pv_val) if not np.isnan(pv_val) else None,
        'image_path': img_path_rel
    }

def main():
    out_dir = '/Volumes/SSD/standardized-dataset'
    out_img_dir = os.path.join(out_dir, 'images')
    out_num_dir = os.path.join(out_dir, 'numerical')
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_num_dir, exist_ok=True)
    
    print("Processing Solar Satellite data...")
    solar_num_dir = '/Volumes/SSD/solar-satellite/refactored/numerical'
    solar_img_dir = '/Volumes/SSD/solar-satellite/refactored/images'
    
    solar_dfs = []
    for file in os.listdir(solar_num_dir):
        if file.endswith('.parquet') and file != 'all.parquet':
            try:
                df = pd.read_parquet(os.path.join(solar_num_dir, file))
                solar_dfs.append(df)
            except Exception as e:
                pass
                
    if solar_dfs:
        solar_df = pd.concat(solar_dfs, ignore_index=True)
    else:
        solar_df = pd.read_parquet(os.path.join(solar_num_dir, 'all.parquet'))
        
    print(f"Loaded {len(solar_df)} records from solar satellite.")
    
    print("Copying solar satellite images...")
    for site_folder in os.listdir(solar_img_dir):
        src_folder = os.path.join(solar_img_dir, site_folder)
        dst_folder = os.path.join(out_img_dir, site_folder)
        if os.path.isdir(src_folder) and not os.path.exists(dst_folder):
            shutil.copytree(src_folder, dst_folder)
                
    print("Processing UK Data...")
    uk_nc_file = '/Volumes/SSD/uk-data/uk_pv_local_paired_dataset.nc'
    ds = xr.open_dataset(uk_nc_file)
    
    pv_ids = ds['pv_id'].values
    times = ds['time'].values
    hrv_data = ds['satellite_hrv'].values
    pv_data = ds['pv_generation'].values
    
    time_strs = pd.to_datetime(times).strftime('%Y-%m-%dT%H-%M-%SZ').values
    time_iso = pd.to_datetime(times).strftime('%Y-%m-%dT%H:%M:%SZ').values
    
    valid_mask = ~np.isnan(hrv_data).all(axis=(2, 3))
    valid_indices = np.argwhere(valid_mask)
    print(f"Found {len(valid_indices)} valid image patches in UK data to extract.")
    
    for pv_id in pv_ids:
        os.makedirs(os.path.join(out_img_dir, f"uk_pv_{pv_id}"), exist_ok=True)
    
    tasks = []
    for pv_idx, time_idx in valid_indices:
        tasks.append((
            pv_ids[pv_idx],
            hrv_data[pv_idx, time_idx],
            pv_data[pv_idx, time_idx],
            time_strs[time_idx],
            time_iso[time_idx],
            out_img_dir
        ))

    uk_records = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for res in tqdm(executor.map(process_patch, tasks), total=len(tasks), desc="Extracting UK images"):
            uk_records.append(res)
        
    print(f"Created {len(uk_records)} UK records.")
    uk_df = pd.DataFrame(uk_records)
    
    print("Combining datasets...")
    combined_df = pd.concat([solar_df, uk_df], ignore_index=True)
    
    print("Saving combined numerical dataset to parquet...")
    combined_df.to_parquet(os.path.join(out_num_dir, 'all.parquet'), index=False)
    
    print("Done! Standardized dataset created at:", out_dir)

if __name__ == "__main__":
    main()

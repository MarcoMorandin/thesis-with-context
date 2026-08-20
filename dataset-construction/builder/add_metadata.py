import pandas as pd
import requests
import json
import os
import time
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def get_nrel_pvdaq_metadata(site_ids):
    site_meta = {}
    for site_id in site_ids:
        url = f"https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/system_metadata/{site_id}_system_metadata.json"
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 404:
                    print(f"Warning: NREL metadata not found for site {site_id}")
                    break
                if response.status_code == 429:
                    print(f"Rate limited by NREL S3. Retrying in {10 * (attempt + 1)}s...")
                    time.sleep(10 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                
                site = data.get('Site', {})
                system = data.get('System', {})
                site_meta[str(site_id)] = {
                    'latitude': float(site.get('latitude')) if site.get('latitude') else None,
                    'longitude': float(site.get('longitude')) if site.get('longitude') else None,
                    'power_w': float(system.get('power')) * 1000 if system.get('power') else None
                }
                break
            except Exception as e:
                print(f"Warning: Failed to fetch NREL metadata for {site_id} (Attempt {attempt+1}/5): {e}")
                time.sleep(5)
                
    return site_meta

def get_uk_pv_metadata():
    try:
        print("Downloading UK PV metadata from Hugging Face...")
        metadata_path = hf_hub_download(
            repo_id="openclimatefix/uk_pv",
            repo_type="dataset",
            filename="metadata.csv",
        )
        df_meta = pd.read_csv(metadata_path)
        site_meta = {}
        for _, row in df_meta.iterrows():
            site_id_str = str(row['ss_id'])
            site_meta[site_id_str] = {
                'latitude': float(row['latitude_rounded']),
                'longitude': float(row['longitude_rounded']),
                'power_w': float(row['kWp']) * 1000 if pd.notna(row['kWp']) else None
            }
        return site_meta
    except Exception as e:
        print(f"Warning: Failed to fetch UK PV metadata: {e}")
        return {}

def fetch_uk_pv_yield(target_site_ids):
    print("Downloading UK PV yield data from Hugging Face...")
    pv_yield_dfs = []
    hf_repo_id = "openclimatefix/uk_pv"
    hf_repo_type = "dataset"
    pv_data_folder = "30_minutely"
    
    target_site_ids = set([int(sid) for sid in target_site_ids])
    
    for year in [2019, 2020]:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            parquet_filename = f"{pv_data_folder}/year={year}/month={month_str}/data.parquet"
            try:
                yield_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    repo_type=hf_repo_type,
                    filename=parquet_filename,
                )
                month_df = pd.read_parquet(yield_path, columns=["ss_id", "datetime_GMT", "generation_Wh"])
                month_df = month_df[month_df['ss_id'].isin(target_site_ids)]
                pv_yield_dfs.append(month_df)
            except Exception as exc:
                pass
                
    if not pv_yield_dfs:
        print("Warning: Could not fetch yield data.")
        return pd.DataFrame()
        
    df_yield = pd.concat(pv_yield_dfs, ignore_index=True)
    df_yield["datetime_GMT"] = pd.to_datetime(df_yield["datetime_GMT"], utc=True)
    df_yield["power_w"] = df_yield["generation_Wh"] * 2.0  # Convert Wh in 30m to W
    df_yield["site_id"] = df_yield["ss_id"].astype(str)
    return df_yield[['site_id', 'datetime_GMT', 'power_w']]

def fetch_weather_open_meteo(lat, lon, start_date_str, end_date_str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "temperature_2m,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,cloudcover,windspeed_10m,precipitation",
        "timezone": "UTC"
    }
    
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print(f"Rate limited by Open-Meteo. Retrying in {15 * (attempt + 1)}s...")
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            
            hourly = data.get("hourly", {})
            if not hourly:
                print(f"Warning: No hourly data returned from Open-Meteo for lat:{lat} lon:{lon} dates:{start_date_str} to {end_date_str}")
                return pd.DataFrame()
                
            df_weather = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"], utc=True),
                "temperature_2m": hourly.get("temperature_2m", []),
                "shortwave_radiation": hourly.get("shortwave_radiation", []),
                "direct_radiation": hourly.get("direct_radiation", []),
                "diffuse_radiation": hourly.get("diffuse_radiation", []),
                "direct_normal_irradiance": hourly.get("direct_normal_irradiance", []),
                "cloudcover": hourly.get("cloudcover", []),
                "windspeed_10m": hourly.get("windspeed_10m", []),
                "precipitation": hourly.get("precipitation", [])
            })
            return df_weather
        except Exception as e:
            print(f"Error fetching Open-Meteo for lat:{lat} lon:{lon} (Attempt {attempt+1}/5): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print("Response text:", e.response.text)
            time.sleep(10)
    print(f"Failed to fetch weather from Open-Meteo for lat:{lat} lon:{lon} after 5 attempts.")
    return pd.DataFrame()

def main():
    parquet_path = "/Volumes/SSD/standardized-dataset/numerical/all.parquet"
    out_path = "/Volumes/SSD/standardized-dataset/numerical/all.parquet"
    
    print(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df['site_id'] = df['site_id'].astype(str)
    
    # Clean any duplicate or corrupted columns from previous runs
    to_drop = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    if to_drop:
        df = df.drop(columns=to_drop)
        
    weather_cols = [
        'temperature_2m', 'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
        'direct_normal_irradiance', 'cloudcover', 'windspeed_10m', 'precipitation'
    ]
    for c in weather_cols:
        if c not in df.columns:
            df[c] = np.nan
    
    # 1. Fetch missing uk_pv power_w
    uk_site_ids = df[df['dataset'] == 'uk_pv']['site_id'].unique()
    uk_yield_df = fetch_uk_pv_yield(uk_site_ids)
    if not uk_yield_df.empty:
        print("Merging missing UK PV power_w...")
        # Since df could have solar-satellite power_w correct, we update only uk_pv
        df_uk = df[df['dataset'] == 'uk_pv'].copy()
        df_uk.drop(columns=['power_w'], inplace=True, errors='ignore')
        df_uk = df_uk.merge(uk_yield_df, left_on=['site_id', 'timestamp_utc'], right_on=['site_id', 'datetime_GMT'], how='left')
        if 'datetime_GMT' in df_uk.columns:
            df_uk.drop(columns=['datetime_GMT'], inplace=True)
            
        print("Interpolating short gaps and dropping unfixable missing UK PV power_w...")
        df_uk = df_uk.sort_values(['site_id', 'timestamp_utc'])
        df_uk['power_w'] = df_uk.groupby('site_id')['power_w'].transform(lambda x: x.interpolate(method='linear', limit=3))
        df_uk = df_uk.dropna(subset=['power_w'])
            
        df_solar = df[df['dataset'] != 'uk_pv'].copy()
        df = pd.concat([df_solar, df_uk], ignore_index=True)
    
    print("Fetching NREL PVDAQ metadata...")
    nrel_site_ids = df[df['dataset'] == 'goes_pvdaq']['site_id'].unique()
    nrel_meta = get_nrel_pvdaq_metadata(nrel_site_ids)
    
    print("Fetching UK PV metadata...")
    uk_meta = get_uk_pv_metadata()
    
    # Add installed power and coordinates columns
    installed_power = []
    latitudes = []
    longitudes = []
    for idx, row in df.iterrows():
        dataset = row['dataset']
        site_id = str(row['site_id'])
        
        power, lat, lon = None, None, None
        if dataset == 'goes_pvdaq':
            info = nrel_meta.get(site_id)
            if info:
                power = info.get('power_w')
                lat = info.get('latitude')
                lon = info.get('longitude')
        elif dataset == 'uk_pv':
            info = uk_meta.get(site_id)
            if info:
                power = info.get('power_w')
                lat = info.get('latitude')
                lon = info.get('longitude')
            
        installed_power.append(power)
        latitudes.append(lat)
        longitudes.append(lon)
        
    df['installed_power_w'] = installed_power
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    
    print("Preparing to fetch weather data...")
    groups = df.groupby(['dataset', 'site_id'])
    weather_dfs = []
    
    for (dataset, site_id), group in tqdm(groups, desc="Fetching weather per site"):
        # Skip if this site already has weather data
        if group['temperature_2m'].notna().any():
            continue
            
        start_ts = group['timestamp_utc'].min()
        end_ts = group['timestamp_utc'].max()
        
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue
            
        start_date_str = start_ts.strftime('%Y-%m-%d')
        end_date_str = end_ts.strftime('%Y-%m-%d')
        
        lat, lon = None, None
        site_id_str = str(site_id)
        if dataset == 'goes_pvdaq':
            info = nrel_meta.get(site_id_str)
            if info:
                lat, lon = info.get('latitude'), info.get('longitude')
        elif dataset == 'uk_pv':
            info = uk_meta.get(site_id_str)
            if info:
                lat, lon = info.get('latitude'), info.get('longitude')
                
        if lat is not None and lon is not None:
            df_w = fetch_weather_open_meteo(lat, lon, start_date_str, end_date_str)
            if not df_w.empty:
                df_w['dataset'] = dataset
                df_w['site_id'] = site_id_str
                weather_dfs.append(df_w)
            
    if weather_dfs:
        all_weather = pd.concat(weather_dfs, ignore_index=True)
        print("Merging weather data...")
        df = df.sort_values('timestamp_utc')
        all_weather = all_weather.sort_values('time')
        
        all_weather['site_id'] = all_weather['site_id'].astype(str)
        
        df_needs_weather = df[df['temperature_2m'].isna()].copy()
        df_has_weather = df[df['temperature_2m'].notna()].copy()
        
        df_needs_weather = df_needs_weather.drop(columns=[c for c in weather_cols if c in df_needs_weather.columns])
        
        merged_needs = pd.merge_asof(
            df_needs_weather, 
            all_weather, 
            left_on='timestamp_utc', 
            right_on='time',
            by=['dataset', 'site_id'],
            direction='nearest'
        )
        
        if 'time' in merged_needs.columns:
            merged_needs.drop(columns=['time'], inplace=True)
            
        df = pd.concat([df_has_weather, merged_needs], ignore_index=True)
        
    print(f"Saving enriched dataset to {out_path}...")
    df.to_parquet(out_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()

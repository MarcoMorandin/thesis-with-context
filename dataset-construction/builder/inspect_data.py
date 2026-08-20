import pandas as pd
import xarray as xr
import os

print("\n=== UK DATA ===")
uk_nc_file = '/Volumes/SSD/uk-data/uk_pv_local_paired_dataset.nc'
if os.path.exists(uk_nc_file):
    ds = xr.open_dataset(uk_nc_file)
    print("Variables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))
    for var in ds.data_vars:
        print(f"{var} dims:", ds[var].dims)
        print(f"{var} shape:", ds[var].shape)
    
    # print a bit of the first variable to see structure
    print(ds.head(2))
else:
    print(f"File not found: {uk_nc_file}")

print("\n=== SOLAR SATELLITE ===")
solar_num_dir = '/Volumes/SSD/solar-satellite/refactored/numerical'
solar_files = os.listdir(solar_num_dir)
if len(solar_files) > 0:
    for file in solar_files:
        if file.endswith('.parquet'):
            try:
                solar_df = pd.read_parquet(os.path.join(solar_num_dir, file), engine='pyarrow')
                print(f"File: {file}")
                print("Columns:", solar_df.columns.tolist())
                print("Head (1):\n", solar_df.head(1).to_dict('records'))
                break
            except Exception as e:
                print(f"Failed to read {file}: {e}")
                try:
                    solar_df = pd.read_parquet(os.path.join(solar_num_dir, file), engine='fastparquet')
                    print(f"File: {file} (fastparquet)")
                    print("Columns:", solar_df.columns.tolist())
                    print("Head (1):\n", solar_df.head(1).to_dict('records'))
                    break
                except Exception as e2:
                    print(f"Failed to read {file} with fastparquet: {e2}")

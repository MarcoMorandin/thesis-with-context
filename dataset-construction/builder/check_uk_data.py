import xarray as xr
import numpy as np
import pandas as pd

uk_nc_file = '/Volumes/SSD/uk-data/uk_pv_local_paired_dataset.nc'
ds = xr.open_dataset(uk_nc_file)

hrv = ds['satellite_hrv'].values
print("HRV min:", np.nanmin(hrv))
print("HRV max:", np.nanmax(hrv))
print("HRV mean:", np.nanmean(hrv))
print("HRV shape:", hrv.shape)
print("NaN count:", np.isnan(hrv).sum())

pv = ds['pv_generation'].values
print("PV min:", np.nanmin(pv))
print("PV max:", np.nanmax(pv))
print("PV mean:", np.nanmean(pv))
print("PV shape:", pv.shape)
print("NaN count:", np.isnan(pv).sum())

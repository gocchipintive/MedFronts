import xarray as xr
import numpy as np
import regionmask
import cartopy.feature as cfeature

# --- Open dataset ---
ds = xr.open_dataset("RESULTS/chl_surface_mean.nc")

# Use 2D coordinates
lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)

#set to nan lon < -5.7
ds = ds.where(lon > -5.7)
#set to nan where lat 42 and lon < 0
ds = ds.where(np.logical_not(np.logical_and(lat > 42,lon < 0)))

#save to new netcdf file
ds.to_netcdf("RESULTS/chl_surface_mean.nc_masked.nc")

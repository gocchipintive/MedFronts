import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import seaborn as sns
import numpy as np

var = "HI_5_10"
var = "HI_gt10"
var = "P_l" #total chlorophyll

if var == "HI_5_10":
    # Open the NetCDF file
    filename = "RESULTS/frequency_HI_5_10_regridded.nc"
    cmap = cmocean.cm.rain
    ds = xr.open_dataset(filename)
    # Select the variable of interest
    da = ds["HI"]
elif var == "HI_gt10":
    # Open the NetCDF file
    filename = "RESULTS/frequency_HI_gt10_regridded.nc"
    cmap = cmocean.cm.matter
    ds = xr.open_dataset(filename)
    # Select the variable of interest
    da = ds["HI"]
elif var == "P_l":
    ds = xr.open_dataset("RESULTS/chl_surface_mean.nc")
    da = ds.P_l
    cmap = "viridis"


import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Create figure
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add coastlines and features
ax.coastlines(resolution="10m", linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="white")

# Plot the data
if var == "HI_5_10":
    #construct 2d array of lon and lat
    lon2d, lat2d = np.meshgrid(ds.lon.values, ds.lat.values)
    #set to nan lon < -5.7
    ds = ds.where(lon2d > -5.7)
    da = da.where(lon2d > -5.7)
    #set to nan where lat 42 and lon < 0
    ds = ds.where(np.logical_not(np.logical_and(lat2d > 42,lon2d < 0)))
    da = da.where(np.logical_not(np.logical_and(lat2d > 42,lon2d < 0)))
    im = ax.pcolormesh(lon2d, lat2d, ds.HI.values*100, cmap=cmap, vmin=0, vmax=80, transform=ccrs.PlateCarree())
elif var == "HI_5_10" or var == "HI_gt10":
    #construct 2d array of lon and lat
    lon2d, lat2d = np.meshgrid(ds.lon.values, ds.lat.values)
    #set to nan lon < -5.7
    ds = ds.where(lon2d > -5.7)
    da = da.where(lon2d > -5.7)
    #set to nan where lat 42 and lon < 0
    ds = ds.where(np.logical_not(np.logical_and(lat2d > 42,lon2d < 0)))
    da = da.where(np.logical_not(np.logical_and(lat2d > 42,lon2d < 0)))
    im = ax.pcolormesh(lon2d, lat2d, ds.HI.values*100, cmap=cmap, vmin=0, vmax=60, transform=ccrs.PlateCarree())
elif var == "P_l":
    #construct 2d array of lon and lat
    lon2d, lat2d = np.meshgrid(ds.lon.values, ds.lat.values)
    #set to nan lon < -5.7
    ds = ds.where(lon2d > -5.7)
    da = da.where(lon2d > -5.7)
    #set to nan where lat 42 and lon < 0
    ds = ds.where(np.logical_not(np.logical_and(lat2d > 42,lon2d < 0)))
    da = da.where(np.logical_not(np.logical_and(lat2d > 42,lon2d < 0)))
    #use log scale in colorbar
    im = ax.pcolormesh(ds.lon.values, ds.lat.values, ds.P_l.values, norm='log', vmin=0.05, vmax=0.5, cmap=cmap, transform=ccrs.PlateCarree())
#add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, aspect=16, shrink=0.5)
#add lon lat grid and ticks
gls = ax.gridlines(draw_labels=True, color="none")
gls.top_labels=False   # suppress top labels
gls.right_labels=False # suppress right labels
if var == "P_l":
    #cbar.set_label(r"Chl $[mg/m^3]$", fontsize=15)
    #st cbar ticks to 0.01, 0.1, 0.5, 1
    cbar.set_ticks([0.05, 0.1, 0.5, 1])
    ax.set_title(r"Mean Chl-a $[mg/m^3]$", fontsize=20)
elif var == "HI_5_10":
    ax.set_title("Weak Fronts Frequency [%]", fontsize=20)
elif var == "HI_gt10":
    ax.set_title("Strong Fronts Frequency [%]", fontsize=20)
cbar.ax.tick_params(labelsize=12)
#show tick labels only on left and bottom


if var == "HI_5_10":
    fig.savefig('FIGS/HI_5_10_frequency.png', dpi=600)
elif var == "HI_gt10":
    fig.savefig('FIGS/HI_gt10_frequency.png', dpi=600)
elif var == "P_l":
    fig.savefig('FIGS/chl_surface_mean.png', dpi=600)

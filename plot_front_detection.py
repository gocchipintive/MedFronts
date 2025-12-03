import xarray as xr
import tarfile
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

which = 'april'
#which = 'june'
if which == 'june':
    sst_file = "/g100_work/OGS23_PRACE_IT/plazzari/NECCTON/Neccton_hindcast1999_2022_v22/wrkdir/MODEL/FORCINGS/2020/06/T20200601-12:00:00.nc"
    tg_file  = "/g100_scratch/userexternal/gocchipi/HI_index/TG_MedSea_2020-6-1.nc"
elif which == 'april':
    sst_file = "/g100_work/OGS23_PRACE_IT/plazzari/NECCTON/Neccton_hindcast1999_2022_v22/wrkdir/MODEL/FORCINGS/2020/04/T20200401-12:00:00.nc"
    tg_file  = "/g100_scratch/userexternal/gocchipi/HI_index/TG_MedSea_2020-4-1.nc"
sst_varname = "votemper"

ds_sst = xr.open_dataset(sst_file)
if 'deptht' in ds_sst[sst_varname].dims:
    ds_sst = ds_sst[sst_varname].isel(deptht=0).squeeze("time_counter")
elif 'depth' in ds_sst[sst_varname].dims:
    ds_sst = ds_sst[sst_varname].isel(depth=0).squeeze("time_counter")
else:
    ds_sst = ds_sst[sst_varname].squeeze("time_counter")
#where sst <=0 set to nan
ds_sst = ds_sst.where(ds_sst > 0)

if which == 'june':
    hi_file = "/g100_scratch/userexternal/gocchipi/HI_index/DATA/2020/06/HI20200601-12:00:00.nc"
elif which == 'april':
    hi_file = "/g100_scratch/userexternal/gocchipi/HI_index/DATA/2020/04/HI20200401-12:00:00.nc"
hi_varname = "HI"
ds_hi = xr.open_dataset(hi_file)
ds_hi = ds_hi[hi_varname].squeeze("time_counter")

chl_file = "/g100_work/OGS_test2528/plazzari/Neccton_hindcast1999_2022/v22/wrkdir/MODEL/AVE_FREQ_1_tar/2020/P_l.tar"
chl_varname = "P_l"

#extrect daily file from tar
tar = tarfile.open(chl_file, "r")
if which == 'june':
    member = tar.getmember("ave.20200601-12:00:00.P_l.nc")
elif which == 'april':
    member = tar.getmember("ave.20200401-12:00:00.P_l.nc")
f = tar.extractfile(member)
ds_chl = xr.open_dataset(f)
ds_chl = ds_chl[chl_varname].isel(depth=0).squeeze("time")


#load tg gradient file
ds_tg = xr.open_dataset(tg_file)
tg = ds_tg['tg'].squeeze("time")
flag_tg = ds_tg['flag'].squeeze("time")
#put to nan tg where flag_tg !=0
tg = tg.where(flag_tg == 0)
#extract threshold_MedSea from ds_tg global attributes
threshold = ds_tg.attrs['threshold_MedSea']
mask_tg = xr.where(tg > threshold, 1, 0)

#restrict x dimension of sst and hi to -len(ds_chl.lon) to match chl grid
ds_sst = ds_sst.isel(x=slice(-len(ds_chl.lon), None))
ds_hi = ds_hi.isel(x=slice(-len(ds_chl.lon), None))

#mask outside med
lon, lat = np.meshgrid(ds_chl.lon.values, ds_chl.lat.values)
ds_sst = ds_sst.where(lon > -5.7)
ds_hi = ds_hi.where(lon > -5.7)
ds_chl = ds_chl.where(lon > -5.7)
ds_sst = ds_sst.where(np.logical_not(np.logical_and(lat > 42,lon < 0)))
ds_hi = ds_hi.where(np.logical_not(np.logical_and(lat > 42,lon < 0)))
ds_chl = ds_chl.where(np.logical_not(np.logical_and(lat > 42,lon < 0)))

#create a mask returning 1 where HI>5 and HI<10, else 0
mask_weak_fronts = xr.where((ds_hi > 5) & (ds_hi <= 10), 1, 0)
mask_strong_fronts = xr.where(ds_hi > 10, 1, 0)

# create a figure with 3 subplots: sst, hi, chl in a column using platecarree projection and viridis colormap
fig, axs = plt.subplots(4, 1, figsize=(10, 15), subplot_kw={'projection': ccrs.PlateCarree()})
# Plot SST
cmap_sst = cmocean.cm.thermal
im1 = axs[0].pcolormesh(ds_chl.lon, ds_chl.lat, ds_sst, cmap=cmap_sst, transform=ccrs.PlateCarree())
axs[0].set_title("Sea Surface Temperature (°C)"+f" at {ds_sst.time_counter.values.astype(str)[:10]}")
cbar1 = plt.colorbar(im1, ax=axs[0], orientation='vertical', pad=0.02, aspect=16, shrink=0.5)
cbar1.set_label("°C")
axs[0].coastlines(resolution="10m", linewidth=0.8)
axs[0].add_feature(cfeature.BORDERS, linewidth=0.5)
axs[0].add_feature(cfeature.LAND, facecolor="lightgray")
axs[0].add_feature(cfeature.OCEAN, facecolor="white")
#add yellow contours for weak fronts
contour1 = axs[0].contour(ds_chl.lon, ds_chl.lat, mask_weak_fronts, levels=[0.5], colors='yellow', linewidths=1, transform=ccrs.PlateCarree())
#add red contours for strong fronts
contour2 = axs[0].contour(ds_chl.lon, ds_chl.lat, mask_strong_fronts, levels=[0.5], colors='red', linewidths=1, transform=ccrs.PlateCarree())

# Plot HI
cmap_hi = cmocean.cm.rain
im2 = axs[1].pcolormesh(ds_chl.lon, ds_chl.lat, ds_hi, cmap=cmap_hi, vmin=0, vmax=10, transform=ccrs.PlateCarree())
axs[1].set_title("Heterogeneity Index (HI)")
cbar2 = plt.colorbar(im2, ax=axs[1], orientation='vertical', pad=0.02, aspect=16, shrink=0.5)
cbar2.set_label("HI")
axs[1].coastlines(resolution="10m", linewidth=0.8)
axs[1].add_feature(cfeature.BORDERS, linewidth=0.5)
axs[1].add_feature(cfeature.LAND, facecolor="lightgray")
axs[1].add_feature(cfeature.OCEAN, facecolor="white")
#add yellow contours for weak fronts
contour3 = axs[1].contour(ds_chl.lon, ds_chl.lat, mask_weak_fronts, levels=[0.5], colors='yellow', linewidths=1, transform=ccrs.PlateCarree())
#add red contours for strong fronts
contour4 = axs[1].contour(ds_chl.lon, ds_chl.lat, mask_strong_fronts, levels=[0.5], colors='red', linewidths=1, transform=ccrs.PlateCarree())

# Plot Chlorophyll
cmap_chl = "viridis"
im3 = axs[2].pcolormesh(ds_chl.lon, ds_chl.lat, ds_chl, norm='log', vmin=0.05, vmax=1, cmap=cmap_chl, transform=ccrs.PlateCarree())
axs[2].set_title("Surface Chlorophyll-a (mg/m³)")
cbar3 = plt.colorbar(im3, ax=axs[2], orientation='vertical', pad=0.02, aspect=16, shrink=0.5)
cbar3.set_label("mg/m³")
axs[2].coastlines(resolution="10m", linewidth=0.8)
axs[2].add_feature(cfeature.BORDERS, linewidth=0.5)
axs[2].add_feature(cfeature.LAND, facecolor="lightgray")
axs[2].add_feature(cfeature.OCEAN, facecolor="white")
#add yellow contours for weak fronts
contour5 = axs[2].contour(ds_chl.lon, ds_chl.lat, mask_weak_fronts, levels=[0.5], colors='yellow', linewidths=1, transform=ccrs.PlateCarree())
#add red contours for strong fronts
contour6 = axs[2].contour(ds_chl.lon, ds_chl.lat, mask_strong_fronts, levels=[0.5], colors='red', linewidths=1, transform=ccrs.PlateCarree())

# Plot TG  and contours of tg_mask
cmap_tg = cmocean.cm.balance
im4 = axs[3].pcolormesh(ds_tg.lon, ds_tg.lat, tg, cmap=cmap_tg, vmin=0, vmax=threshold*2, transform=ccrs.PlateCarree())
axs[3].set_title(f"Temperature Gradient with threshold {threshold} °C/km")
#cbar4
cbar4 = plt.colorbar(im4, ax=axs[3], orientation='vertical', pad=0.02, aspect=16, shrink=0.5)
cbar4.set_label("°C/km")
axs[3].coastlines(resolution="10m", linewidth=0.8)
axs[3].add_feature(cfeature.BORDERS, linewidth=0.5)
axs[3].add_feature(cfeature.LAND, facecolor="lightgray")
axs[3].add_feature(cfeature.OCEAN, facecolor="white")
#add contour for tg_mask
contour7 = axs[3].contour(ds_tg.lon, ds_tg.lat, mask_tg, levels=[0.5], colors='black', linewidths=1, transform=ccrs.PlateCarree())


#adjust layout
plt.tight_layout()
#save figure
plt.savefig(f"sst_hi_chl_fronts_{ds_sst.time_counter.values.astype(str)[:10]}.png", dpi=300)

tar.close()
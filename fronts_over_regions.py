import xarray as xr
import geopandas as gpd
import regionmask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

file1 = "RESULTS/frequency_HI_5_10_regridded.nc"
file2 = "RESULTS/frequency_HI_gt10_regridded.nc"

ds1 = xr.open_dataset(file1)
ds2 = xr.open_dataset(file2)

var_name = 'HI'
da1 = ds1[var_name]
da2 = ds2[var_name]


# Load region shapefile
shapefile_path = "MEOW/meow_ecos.shp"
gdf = gpd.read_file(shapefile_path)
name_col = 'ECOREGION' if 'ECOREGION' in gdf.columns else 'name'

# Mediterranean ecoregions (names as in the shapefile)
med_ecoregions = [
    'Adriatic Sea',
    'Aegean Sea',
    'Alboran Sea',
    'Ionian Sea',
    'Levantine Sea',
    'Western Mediterranean',
    'Tunisian Plateau/Gulf of Sidra'
]
region_flags = {
    'Adriatic Sea': 224,
    'Levantine Sea': 225,
    'Ionian Sea': 227,
    'Aegean Sea': 228,
    'Alboran Sea': 229,
    'Western Mediterranean': 230,         # adjust based on your shapefile
    'Tunisian Plateau/Gulf of Sidra': 226  # adjust based on your shapefile
}
gdf_med = gdf[gdf[name_col].isin(med_ecoregions)].copy()

#region mask dataset
regions = regionmask.from_geopandas(gdf_med, names=name_col, name='MEOW_Mediterranean')

# Create a meshgrid of lon/lat points
lon = da1['nav_lon']
lat = da1['nav_lat']

mask2d = regions.mask(lon, lat)  # boolean mask per region

#compute means inside regions
region_means = {}
for region_name, flag in region_flags.items():
    region_mask = mask2d == flag
    mean1 = da1.where(region_mask).mean().item()
    mean2 = da2.where(region_mask).mean().item()
    remainder = 1 - (mean1 + mean2)
    region_means[region_name] = (mean1, mean2, remainder)

#plot means in a bar inside each region
fig = plt.figure(figsize=(10, 3))
ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
#add boundaries lat max 45.97917 min 30.1875, lon max 36.29167 min -8.875
ax.set_extent([-10, 40, 30, 46], crs=ccrs.PlateCarree())

#Prepare centroids for plotting
gdf_med['centroid'] = gdf_med.geometry.centroid
gdf_med['x'] = gdf_med.centroid.x
gdf_med['y'] = gdf_med.centroid.y

# Colors for the bars (nofront, weak, strong)
colors = ['#c49c94','#bcbd22', '#9467bd']

# Plot ecoregion boundaries
gdf_med.boundary.plot(ax=ax, color='black', transform=ccrs.PlateCarree())

# Scaling factor to convert fraction to map height (degrees latitude)
scale = 3
text_offset = 1.2

for idx, row in gdf_med.iterrows():
    x = row['x']
    y = row['y']
    # Shift Adriatic and Ionian bars slightly south
    if row[name_col] in ['Adriatic Sea', 'Ionian Sea']:
        y -= 0.8
    # Reorder values: remainder, 5-10, >10
    values = region_means[row[name_col]]
    values = (values[2], values[0], values[1])
    
    bottom = y
    for v, c in zip(values, colors):
        height = v * scale
        rect = Rectangle((x - 0.5, bottom), 1.0, height, facecolor=c, edgecolor='k', alpha=1,
                         transform=ccrs.PlateCarree())
        ax.add_patch(rect)
        bottom += height
    # Add ticks/labels: fraction at the top of each section
    cum_height = y
    for v in values:
        label_y = cum_height + v*scale/2  # place label at center of section
        ax.text(
            x + text_offset, label_y, f"{v:.2f}",
            ha='left', va='center', fontsize=8, color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            transform=ccrs.PlateCarree()
        )
        cum_height += v*scale

# Add gridlines and labels
gls = ax.gridlines(draw_labels=True, color="none")
gls.top_labels=False   # suppress top labels
gls.right_labels=False # suppress right labels
# Add coastlines and features
ax.coastlines(resolution="10m", linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="white")

#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude')
ax.set_title('Mediterranean Ecoregions: Front Frequency', fontsize=20)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#c49c94', edgecolor='k', label='No Fronts'),
    Patch(facecolor='#bcbd22', edgecolor='k', label='Weak Fronts'),
    Patch(facecolor='#9467bd', edgecolor='k', label='Strong Fronts')
]
ax.legend(handles=legend_elements, loc='upper right')

fig.tight_layout()
fig.savefig('FIGS/mediterranean_fronts_over_regions.png', dpi=300)

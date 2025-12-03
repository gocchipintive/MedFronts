import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
ds = xr.open_dataset('RESULTS/longitudinal_chl_over_fronts.nc')

# Extract variables
nofronts = ds['chl_no_fronts'].values
weak     = ds['chl_weak_fronts'].values
strong   = ds['chl_strong_fronts'].values

lon = ds['lon'].values

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)

ax.plot(lon, nofronts, label='No Fronts',c='#c49c94', linestyle='-')
ax.plot(lon, weak, label='Weak Fronts',c='#bcbd22', linestyle='-')
ax.plot(lon, strong, label='Strong Fronts',c='#9467bd', linestyle='-')

#ax.set_yscale('log')

ax.set_xlabel('Longitude')
ax.set_ylabel('Chlorophyll [mg m$^{-3}$]')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig('FIGS/longitudinal_mean_chl_fronts.png', dpi=300)

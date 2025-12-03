import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
ds = xr.open_dataset('RESULTS/longitudinal_mean_P_groups.nc')

# Extract variables
P1l = ds['P1l'].values
P2l = ds['P2l'].values
P3l = ds['P3l'].values
P4l = ds['P4l'].values

lon = ds['lon'].values

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)

ax.plot(lon, P1l, label='P1',c='k', linestyle='-')
ax.plot(lon, P2l, label='P2',c='k', linestyle='--')
ax.plot(lon, P3l, label='P3',c='k', linestyle='-.')
ax.plot(lon, P4l, label='P4',c='k', linestyle=':')

ax.set_xlabel('Longitude')
ax.set_ylabel('Chlorophyll [mg m$^{-3}$]')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig('FIGS/longitudinal_mean_PFTs.png', dpi=300)

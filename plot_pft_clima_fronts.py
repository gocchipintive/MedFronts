import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
filename = '/g100_scratch/userexternal/gocchipi/HI_index/RESULTS/daily_climatology_ecoregions.nc'
ds = xr.open_dataset(filename)

P1l_mean = ds["P1l_ecoregion_mean"] # (time, ecoregion, front strength)
P2l_mean = ds["P2l_ecoregion_mean"] # (time, ecoregion, front strength)
P3l_mean = ds["P3l_ecoregion_mean"] # (time, ecoregion, front strength)
P4l_mean = ds["P4l_ecoregion_mean"] # (time, ecoregion, front strength)

P1l_std = ds["P1l_ecoregion_std"] # (time, ecoregion, front strength)
P2l_std = ds["P2l_ecoregion_std"] # (time, ecoregion, front strength)
P3l_std = ds["P3l_ecoregion_std"] # (time, ecoregion, front strength)
P4l_std = ds["P4l_ecoregion_std"] # (time, ecoregion, front strength)

time = ds["dayofyear"] # (time)

MEDITERRANEAN_ECOREGIONS = [
    'Adriatic Sea',
    'Aegean Sea',
    'Alboran Sea',
    'Ionian Sea',
    'Levantine Sea',
    'Western Mediterranean',
    'Tunisian Plateau/Gulf of Sidra'
]

# Define colors for each front strenght
colors = ['#c49c94','#bcbd22','#9467bd']
fronts = ['No Fronts', 'Weak Fronts', 'Strong Fronts']

# define a figure with 4 columns and 7 rows
fig, axs = plt.subplots(7, 4, figsize=(20, 25), sharex=True, sharey=True)

#plot each ecoregion in a different row and each pft in a different column
#in each subplot plot the mean and the std as a shaded area for each front strength
for i, ecoregion in enumerate(MEDITERRANEAN_ECOREGIONS):
    for j, (P_mean, P_std, pft) in enumerate(zip([P1l_mean, P2l_mean, P3l_mean, P4l_mean],
                                                 [P1l_std, P2l_std, P3l_std, P4l_std],
                                                 ['P1', 'P2', 'P3', 'P4'])):
        ax = axs[i, j]
        for k in range(3): # for each front strength
            mean = P_mean[:, i, k]
            std = P_std[:, i, k]
            ax.plot(time, mean, label=fronts[k], color=colors[k],alpha=0.9)
            ax.fill_between(time, mean - std, mean + std, color=colors[k], alpha=0.1)
        
        ax.set_title(f'{ecoregion} - {pft}')
        ax.set_ylabel(r'CHL [mg m$^{-3}$]')
        if i == 6:
            ax.set_xlabel('Day of Year')
        if i == 0 and j == 0:
            ax.legend(loc='upper right')

fig.tight_layout()

fig.savefig('FIGS/climatology_pft_fronts_ecoregions.png', dpi=300)


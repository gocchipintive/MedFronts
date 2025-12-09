#!/usr/bin/env python3
"""
Reads daily_climatology_ecoregions.nc, computes mean over dayofyear,
normalizes P1l–P4l relative to their sum per region and front,
and plots stacked bar charts per region showing front classes.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# === INPUT FILE ===
INFILE = "RESULTS/daily_climatology_ecoregions_P.nc"
OUTFIG = "FIGS/pft_front_region_bars.png"

# === SETTINGS ===
PFTS = ["P1l", "P2l", "P3l", "P4l"]
PFTS_labels = ['Diatoms', 'Nanoflagellates', 'Picoplankton', 'Dinoflagellates']
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red
FRONT_LABELS = ["no", "weak", "strong"]

# === LOAD DATA ===
ds = xr.open_dataset(INFILE)

ecoregions = ds["ecoregion"].values
fronts = ds["front"].values  # ["no","weak","strong"]
dayofyear = ds["dayofyear"].values

# === COMPUTE MEANS OVER TIME ===
mean_data = {}
for pft in PFTS:
    var_name = f"{pft}_ecoregion_mean"
    # mean over dayofyear → shape: (ecoregion, front)
    mean_data[pft] = ds[var_name].mean(dim="dayofyear", skipna=True)

# === NORMALIZE PER REGION AND FRONT ===
# stack PFTs along new dimension to simplify normalization
pft_stack = xr.concat([mean_data[p] for p in PFTS], dim="pft")
pft_stack = pft_stack.assign_coords(pft=PFTS)  # shape: (pft, ecoregion, front)

# compute normalization factor (sum across PFTs)
pft_sum = pft_stack.sum(dim="pft", skipna=True)
normalized = (pft_stack / pft_sum) * 100.0  # percent contribution

# === PLOT ===
nregions = len(ecoregions)
fig, axes = plt.subplots(1, nregions, figsize=(2.2 * nregions, 5), sharey=True)
if nregions == 1:
    axes = [axes]

for i, region in enumerate(ecoregions):
    ax = axes[i]
    data_region = normalized.sel(ecoregion=region)

    # For each front, plot stacked bars
    bottoms = np.zeros(len(FRONT_LABELS))
    for lab, pft, color in zip(PFTS_labels, PFTS, COLORS):
        vals = data_region.sel(front=FRONT_LABELS).sel(pft=pft).values
        ax.bar(FRONT_LABELS, vals, bottom=bottoms, color=color, edgecolor='black', label=lab)
        bottoms += np.nan_to_num(vals)

    ax.set_title(str(region), fontsize=9, weight='bold')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Relative Chl [%]' if i == 0 else "")
    ax.set_xticklabels(FRONT_LABELS, rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# === LEGEND AND LAYOUT ===
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=14)
#fig.suptitle("Normalized PFT Composition by Front and Ecoregion", fontsize=12, weight='bold', y=1.05)
plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig(OUTFIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to {OUTFIG}")

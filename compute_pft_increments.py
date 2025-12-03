#!/usr/bin/env python3
"""
Compute seasonal percent changes for each PFT (P1l–P4l)
and for their total (P1l+P2l+P3l+P4l)
from daily climatology of ecoregions and fronts.
Also computes significance flags for weak/strong fronts vs no front.
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import regionmask
from scipy import stats

# ---- Input ----
INFILE = "RESULTS/daily_climatology_ecoregions.nc"
OUTDIR = "RESULTS/"
SEASONS = {
    "JFM": (1, 90),
    "AMJ": (91, 181),
    "JAS": (182, 273),
    "OND": (274, 365)
}
#PFTS = ["P1l", "P2l", "P3l", "P4l"]
PFTS = ["N1p", "N3n"]

# ---- Load dataset ----
ds = xr.open_dataset(INFILE)
ecoregions = ds["ecoregion"].values
fronts = ds["front"].values

# ---- Spatial mask for weighting ----
SPTFILE = "RESULTS/chl_surface_mean.nc"
ds_spt = xr.open_dataset(SPTFILE)
lat = ds_spt["lat"].values
lon = ds_spt["lon"].values
SHAPEFILE = "MEOW/meow_ecos.shp"
MEDITERRANEAN_ECOREGIONS = [
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
    'Western Mediterranean': 230, 
    'Tunisian Plateau/Gulf of Sidra': 226 
}

def build_region_mask(lat, lon, shapefile=SHAPEFILE):
    gdf = gpd.read_file(shapefile)
    name_col = 'ECOREGION' if 'ECOREGION' in gdf.columns else 'name'
    gdf_med = gdf[gdf[name_col].isin(MEDITERRANEAN_ECOREGIONS)].copy()
    regions = regionmask.from_geopandas(gdf_med, names=name_col, name='MEOW_Mediterranean')
    mask = regions.mask(np.array(lon), np.array(lat))
    return mask, list(regions.names)

region_mask, region_names = build_region_mask(lat, lon)
nregions = len(region_names)
n_ens = np.zeros(nregions, dtype=int)
region_mask_np = np.asarray(region_mask) if hasattr(region_mask,"values") else np.asarray(region_mask)
region_mask_int = np.full(region_mask_np.shape, -999, dtype=np.int32)
finite_mask = np.isfinite(region_mask_np)
region_mask_int[finite_mask] = region_mask_np[finite_mask].astype(np.int32)

for r, rname in enumerate(region_names):
    region_flag = region_flags[rname]
    region_full = (region_mask_int == region_flag).astype(np.uint8)
    n_ens[r] = np.nansum(region_full)

n_ens = np.tile(n_ens, (4, 1))  # shape (4,7) to match seasons

# ---- Helper functions ----
def doy_range_mask(doys, start, end):
    return (doys >= start) & (doys <= end)

def compute_seasonal_means(var_data, doys):
    season_means = []
    for name, (start, end) in SEASONS.items():
        mask = doy_range_mask(doys, start, end)
        mean = var_data.sel(dayofyear=doys[mask]).mean(dim="dayofyear", skipna=True)
        mean = mean.expand_dims(season=[name])
        season_means.append(mean)
    return xr.concat(season_means, dim="season")

# ---- Add TOTAL variable (sum of P1l–P4l) ----
ds["Total_ecoregion_mean"] = sum([ds[f"{p}_ecoregion_mean"] for p in PFTS])
# Assuming std of sum ≈ sqrt(sum of variances) (independent assumption)
ds["Total_ecoregion_std"] = np.sqrt(sum([ds[f"{p}_ecoregion_std"] ** 2 for p in PFTS]))

ALL_VARS = PFTS + ["Total"]

# ---- Process each PFT and Total ----
for pft in ALL_VARS:
    var_name = f"{pft}_ecoregion_mean"
    std_name = f"{pft}_ecoregion_std"
    data = ds[var_name]
    data_std = ds[std_name]

    doy_vals = ds["dayofyear"].values
    seasonal = compute_seasonal_means(data, doy_vals)
    seasonal_std = compute_seasonal_means(data_std, doy_vals)

    # Extract fronts
    no_front = seasonal.sel(front="no")
    weak_front = seasonal.sel(front="weak")
    strong_front = seasonal.sel(front="strong")

    no_front_std = seasonal_std.sel(front="no")
    weak_front_std = seasonal_std.sel(front="weak")
    strong_front_std = seasonal_std.sel(front="strong")

    # Percent change
    weak_pct = ((weak_front - no_front) / no_front) * 100
    strong_pct = ((strong_front - no_front) / no_front) * 100

    # z-score for significance
    weak_z = np.abs(weak_front - no_front) / np.sqrt(
        np.power(weak_front_std, 2) / n_ens + np.power(no_front_std, 2) / n_ens
    )
    strong_z = np.abs(strong_front - no_front) / np.sqrt(
        np.power(strong_front_std, 2) / n_ens + np.power(no_front_std, 2) / n_ens
    )
    df_ = n_ens + n_ens - 2

    # p-values
    weak_p = 2 * (1 - stats.t.cdf(np.abs(weak_z), df=df_))
    strong_p = 2 * (1 - stats.t.cdf(np.abs(strong_z), df=df_))

    # Significance flags
    weak_sig_flag = (weak_p < 0.05)
    strong_sig_flag = (strong_p < 0.05)

    # Transpose to ecoregion × season
    weak_pct = weak_pct.transpose("ecoregion", "season")
    strong_pct = strong_pct.transpose("ecoregion", "season")
    weak_sig_flag = xr.DataArray(weak_sig_flag, dims=weak_pct.dims)
    strong_sig_flag = xr.DataArray(strong_sig_flag, dims=strong_pct.dims)

    # Combine into one DataFrame: 8 value columns + 8 flag columns
    data_arr = np.hstack([
        weak_pct.values,
        weak_sig_flag.values.T.astype(int),
        strong_pct.values,
        strong_sig_flag.values.T.astype(int)
    ])

    columns = ([f"weak_{s}" for s in SEASONS.keys()] +
               [f"weak_sig_{s}" for s in SEASONS.keys()] +
               [f"strong_{s}" for s in SEASONS.keys()] +
               [f"strong_sig_{s}" for s in SEASONS.keys()])

    df = pd.DataFrame(index=ecoregions, data=data_arr, columns=columns)

    out_csv = f"{OUTDIR}/{pft}_seasonal_percent_change.csv"
    df.to_csv(out_csv, float_format="%.3f")
    print(f"[DONE] Saved {out_csv}")

ds.close()

#!/usr/bin/env python3
"""
MPI-parallel script to compute daily climatology (day-of-year 1..365)
of P1l-P4l classified by HI fronts (no / weak / strong) per Mediterranean ecoregion.

Output NetCDF contains for each variable:
  <var>_ecoregion_mean(dayofyear, ecoregion, front)
  <var>_ecoregion_std(dayofyear, ecoregion, front)
"""
#debug rutine
import psutil, os, tracemalloc
def mem_usage(tag=""):
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1024**2  # MB
    print(f"[R{rank}] {tag} -- RSS: {rss:.1f} MB", flush=True)

import os
import re
import tarfile
from glob import glob
import numpy as np
import xarray as xr
from mpi4py import MPI
from scipy.spatial import cKDTree as KDTree
import geopandas as gpd
import regionmask

# ---------------- Settings ----------------
BASE = "/g100_work/OGS_test2528/plazzari/Neccton_hindcast1999_2022/v22/wrkdir/MODEL/AVE_FREQ_1_tar"
HI_BASE = "/g100_scratch/userexternal/gocchipi/HI_index/DATA"
SHAPEFILE = "MEOW/meow_ecos.shp"
#VARS = ["P1l", "P2l", "P3l", "P4l"]
VARS = ["N1p", "N3n"]
OUTFILE = "/g100_scratch/userexternal/gocchipi/HI_index/RESULTS/daily_climatology_ecoregions.nc"
FRONT_THRESHOLDS = [5, 10]  # HI thresholds: <5 no front, 5-10 weak, >=10 strong
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
    'Western Mediterranean': 230, # adjust based on your shapefile 
    'Tunisian Plateau/Gulf of Sidra': 226 # adjust based on your shapefile 
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------- Helpers ----------------
date8_re = re.compile(r"(?<!\d)(19|20)\d{6}")

def log(msg):
    print(f"[R{rank}] {msg}", flush=True)

def list_tar_files(var):
    years = sorted(glob(BASE + "/*"))
    tars = []
    for y in years:
        matches = sorted(glob(f"{y}/{var}.tar"))
        tars.extend(matches)
    return sorted(tars)

def extract_yyyymmdd_from_name(name):
    m = date8_re.search(name)
    if m: return m.group(0)
    m2 = re.search(r"(19|20)\d{2}[-_]\d{2}[-_]\d{2}", name)
    if m2: return m2.group(0).replace("-", "").replace("_", "")
    m3 = re.search(r"(?<!\d)(\d{8})(?!\d)", name)
    if m3: return m3.group(1)
    return None

def sample_grid(tar_path, var):
    with tarfile.open(tar_path, "r") as tar:
        members = sorted([m for m in tar.getmembers() if m.name.endswith(".nc")], key=lambda m: m.name)
        f = tar.extractfile(members[0])
        ds = xr.open_dataset(f, decode_times=False)
        lat_keys = [k for k in ds.coords if re.search("(^|_)lat($|_)", k)]
        lon_keys = [k for k in ds.coords if re.search("(^|_)lon($|_)", k)]
        lat = ds[lat_keys[0]].values
        lon = ds[lon_keys[0]].values
        ds.close()
        return lat.size, lon.size, lat, lon

def regrid_hi_to_pl(chl_ds, hi_ds, var):
    """Nearest-neighbor KDTree: map HI(nav_lon,nav_lat) to P_l grid and return chl_vals, hi_on_pl."""
    chl = chl_ds[var].squeeze()
    if "depth" in chl.dims:
        chl = chl.isel(depth=0)
    elif "deptht" in chl.dims:
        chl = chl.isel(deptht=0)

    lat_pl = chl_ds["lat"].values
    lon_pl = chl_ds["lon"].values

    # HI coords & values
    if ("nav_lat" in hi_ds.coords) and ("nav_lon" in hi_ds.coords):
        nav_lat = hi_ds["nav_lat"].values
        nav_lon = hi_ds["nav_lon"].values
    else:
        # fallback: try any lat/lon coords
        lat_keys = [k for k in hi_ds.coords if "lat" in k]
        lon_keys = [k for k in hi_ds.coords if "lon" in k]
        nav_lat = hi_ds[lat_keys[0]].values
        nav_lon = hi_ds[lon_keys[0]].values

    hi_varname = list(hi_ds.data_vars)[0]
    hi_vals = hi_ds[hi_varname].squeeze().values

    lat_min, lat_max = float(np.min(lat_pl)), float(np.max(lat_pl))
    lon_min, lon_max = float(np.min(lon_pl)), float(np.max(lon_pl))

    valid = np.isfinite(nav_lat) & np.isfinite(nav_lon) & np.isfinite(hi_vals)
    bbox = (nav_lat >= lat_min) & (nav_lat <= lat_max) & (nav_lon >= lon_min) & (nav_lon <= lon_max)
    sel = valid & bbox
    if not np.any(sel):
        raise RuntimeError("No HI points found inside P_l bounding box.")

    pts_hi = np.column_stack((nav_lon[sel].ravel(), nav_lat[sel].ravel()))
    vals_hi = hi_vals[sel].ravel()

    tree = KDTree(pts_hi)
    lon2d, lat2d = np.meshgrid(lon_pl, lat_pl)
    pts_pl = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    _, idx = tree.query(pts_pl, k=1)
    hi_on_pl = vals_hi[idx].reshape(lon2d.shape)

    return chl.values, hi_on_pl

def build_region_mask(lat, lon, shapefile=SHAPEFILE):
    gdf = gpd.read_file(shapefile)
    name_col = 'ECOREGION' if 'ECOREGION' in gdf.columns else 'name'
    gdf_med = gdf[gdf[name_col].isin(MEDITERRANEAN_ECOREGIONS)].copy()
    regions = regionmask.from_geopandas(gdf_med, names=name_col, name='MEOW_Mediterranean')
    mask = regions.mask(np.array(lon), np.array(lat))  # returns values 0..n-1 or NaN
    return mask, list(regions.names)

import gc
import math

def sizeof(obj):
    """Return size in MB of a numpy array or bytes-like object; fallback to 0."""
    try:
        return (obj.nbytes / 1024.0**2)
    except Exception:
        try:
            return (sys.getsizeof(obj) / 1024.0**2)
        except Exception:
            return 0.0

def print_array_info(name, arr):
    if arr is None:
        log(f"{name}: None")
        return
    try:
        s = arr.shape
        dt = getattr(arr, "dtype", type(arr))
        nbytes = sizeof(arr)
        log(f"{name}: shape={s}, dtype={dt}, size={nbytes:.2f} MB")
    except Exception as e:
        log(f"{name}: (info failed) {e}")

def process_tar_files_daily(tar_list, nlat, nlon, lat_vals, lon_vals, region_mask, nregions, var,
                            chunk_rows=40):
    """
    Memory-conservative implementation with chunking and debug prints.

    - chunk_rows: number of latitude rows to process at once (tweak: 20-80).
    """
    import gc
    # streaming accumulators (float64 for sums/sumsq; counts int64)
    daily_sum = np.zeros((365, nregions, 3), dtype=np.float64)
    daily_sumsq = np.zeros((365, nregions, 3), dtype=np.float64)
    daily_cnt = np.zeros((365, nregions, 3), dtype=np.int64)

    # ensure region_mask is numpy (use values if xarray)
    if hasattr(region_mask, "values"):
        region_mask_np = np.asarray(region_mask.values)
    else:
        region_mask_np = np.asarray(region_mask)

    # if your region_mask uses flags like 224..230, keep them
    unique_flags = np.unique(region_mask_np[~np.isnan(region_mask_np)])
    log(f"region_mask unique flags (sample): {unique_flags[:10]}; dtype={region_mask_np.dtype}")

    # convert to integer array if floats (NaNs present)
    # keep NaN as -999 to avoid accidental equality
    region_mask_int = np.full(region_mask_np.shape, -999, dtype=np.int32)
    finite_mask = np.isfinite(region_mask_np)
    region_mask_int[finite_mask] = region_mask_np[finite_mask].astype(np.int32)

    log(f"region_mask_int created. shape {region_mask_int.shape}.")
    mem_usage("after region_mask conversion")
    print_array_info("region_mask_int", region_mask_int)

    for tar_path in tar_list:
        log(f"opening {tar_path}")
        mem_usage(f"Before opening {os.path.basename(tar_path)}")
        with tarfile.open(tar_path, "r") as tar:
            members = sorted([m for m in tar.getmembers() if m.name.endswith(".nc")], key=lambda m: m.name)
            for member in members:
                yyyymmdd = extract_yyyymmdd_from_name(member.name)
                if yyyymmdd is None:
                    continue
                month = int(yyyymmdd[4:6]); day = int(yyyymmdd[6:8])
                if month == 2 and day == 29:
                    continue
                doy = int((np.datetime64(f"2001-{month:02d}-{day:02d}") - np.datetime64("2001-01-01")).astype(int)) + 1
                yyyy = yyyymmdd[:4]
                hi_file = f"{HI_BASE}/{yyyy}/{yyyymmdd[4:6]}/HI{yyyymmdd}-12:00:00.nc"
                if not os.path.exists(hi_file):
                    log(f"[WARN] missing HI {hi_file}; skipping {member.name}")
                    continue

                f_chl = tar.extractfile(member)
                if f_chl is None:
                    log(f"[WARN] extractfile returned None for {member.name}")
                    continue

                # OPEN datasets
                ds_chl = xr.open_dataset(f_chl, decode_times=False)
                ds_hi = xr.open_dataset(hi_file, decode_times=False)
                lon_min = np.min(ds_chl.lon.values) 
                ds_hi = ds_hi.where(ds_hi.nav_lon >= lon_min, drop=True)
                mem_usage("  After open_dataset")

                # get surface arrays as numpy (and downcast to float32 to save memory)
                try:
                    chl_da = ds_chl[var]
                except Exception as e:
                    log(f"[ERROR] var {var} not in {member.name}: {e}")
                    ds_chl.close(); ds_hi.close(); continue

                # select surface
                if "depth" in chl_da.dims:
                    chl_da = chl_da.isel(depth=0)
                elif "deptht" in chl_da.dims:
                    chl_da = chl_da.isel(deptht=0)
                chl_vals = np.asarray(chl_da.squeeze().values, dtype=np.float32)

                # HI: prefer 'HI' variable
                if "HI" in ds_hi:
                    hi_da = ds_hi["HI"].squeeze()
                else:
                    hi_da = list(ds_hi.data_vars)[0]
                    hi_da = ds_hi[hi_da].squeeze()
                hi_on_pl = np.asarray(hi_da.values, dtype=np.float32)

                # close quickly to free file handles and xarray internals
                ds_chl.close()
                ds_hi.close()
                mem_usage("  After extracting arrays (numpy)")

                # debug shapes/dtypes and memory
                print_array_info("chl_vals", chl_vals)
                print_array_info("hi_on_pl", hi_on_pl)
                mem_usage("  after array info")

                # ensure arrays are 2D and match dims: try to reconcile shapes
                if chl_vals.ndim != 2 or hi_on_pl.ndim != 2:
                    log(f"[WARN] unexpected dims chl {chl_vals.ndim} hi {hi_on_pl.ndim} -> skipping")
                    del chl_vals, hi_on_pl
                    gc.collect()
                    continue

                if chl_vals.shape != hi_on_pl.shape:
                    # attempt simple cropping based on lon_min constraint you used earlier
                    # if shapes differ only by columns/rows, try to align using min-lon selection
                    log(f"[WARN] shape mismatch chl {chl_vals.shape} vs hi {hi_on_pl.shape}. Trying simple align.")
                    # attempt align by min-lon assumption: if one array has extra columns on right/left
                    # If shapes not alignable, skip to be safe
                    if chl_vals.shape[0] == hi_on_pl.shape[0]:
                        mincols = min(chl_vals.shape[1], hi_on_pl.shape[1])
                        chl_vals = chl_vals[:, :mincols]
                        hi_on_pl = hi_on_pl[:, :mincols]
                        log(f"  aligned by truncating columns to {mincols}")
                    elif chl_vals.shape[1] == hi_on_pl.shape[1]:
                        minrows = min(chl_vals.shape[0], hi_on_pl.shape[0])
                        chl_vals = chl_vals[:minrows, :]
                        hi_on_pl = hi_on_pl[:minrows, :]
                        log(f"  aligned by truncating rows to {minrows}")
                    else:
                        log("  cannot align shapes -> skip")
                        del chl_vals, hi_on_pl
                        gc.collect()
                        continue

                mem_usage("  After alignment")
                print_array_info("chl_vals (aligned)", chl_vals)
                print_array_info("hi_on_pl (aligned)", hi_on_pl)

                # Build front masks (bool numpy arrays). Keep them as uint8 (0/1) for fast math
                mask_no = (np.isfinite(hi_on_pl) & (hi_on_pl < FRONT_THRESHOLDS[0])).astype(np.uint8)
                mask_weak = (np.isfinite(hi_on_pl) & (hi_on_pl >= FRONT_THRESHOLDS[0]) & (hi_on_pl < FRONT_THRESHOLDS[1])).astype(np.uint8)
                mask_strong = (np.isfinite(hi_on_pl) & (hi_on_pl >= FRONT_THRESHOLDS[1])).astype(np.uint8)
                masks = [mask_no, mask_weak, mask_strong]

                # debug counts
                log(f"  front counts (no/weak/strong): {mask_no.sum()}/{mask_weak.sum()}/{mask_strong.sum()}")
                mem_usage("  After creating masks")

                # process in row-chunks to avoid large intermediate booleans
                nrows = chl_vals.shape[0]
                for row0 in range(0, nrows, chunk_rows):
                    row1 = min(nrows, row0 + chunk_rows)
                    chl_chunk = chl_vals[row0:row1, :]
                    # Precompute finite mask for chunk
                    finite_chunk = np.isfinite(chl_chunk).astype(np.uint8)

                    for r, rname in enumerate(region_names):
                        region_flag = region_flags[rname]
                        # derive region selection as uint8 chunk (so we don't allocate full boolean many times)
                        region_full = (region_mask_int == region_flag).astype(np.uint8)
                        region_chunk = region_full[row0:row1, :]

                        # small speed: if region_chunk has no True skip
                        if region_chunk.sum() == 0:
                            continue

                        # combine with front masks chunk and finite mask
                        for i, fmask in enumerate(masks):
                            fmask_chunk = fmask[row0:row1, :]
                            sel_chunk = region_chunk & fmask_chunk & finite_chunk  # uint8 mask
                            cnt = int(sel_chunk.sum())
                            if cnt == 0:
                                continue
                            # compute sum and sumsq using broadcasting with mask (cast sel_chunk to bool then multiply)
                            # convert to float64 for accumulation to reduce rounding error
                            sel_bool = sel_chunk.astype(bool)
                            vals = chl_chunk[sel_bool].astype(np.float64)  # small array: only True elements in chunk
                            s = np.nansum(vals)
                            ssq = np.nansum(vals * vals)
                            idx = doy - 1
                            daily_sum[idx, r, i] += s
                            daily_sumsq[idx, r, i] += ssq
                            daily_cnt[idx, r, i] += cnt
                            # free
                            del vals, sel_bool
                    # free chunk temporaries
                    del chl_chunk, finite_chunk, region_chunk
                    gc.collect()

                # free whole arrays for this file and force GC
                del chl_vals, hi_on_pl, mask_no, mask_weak, mask_strong
                gc.collect()
                mem_usage("  After processing arrays & GC")

    # end tar loop
    mem_usage("End process_tar_files_daily")
    return daily_sum, daily_sumsq, daily_cnt


# ---------------- Main ----------------
if rank == 0:
    tracemalloc.start()

if rank == 0:
    tar_files = list_tar_files(VARS[0])
    if not tar_files:
        raise SystemExit("No tar files found.")
    nlat, nlon, lat_vals, lon_vals = sample_grid(tar_files[0], VARS[0])
    region_mask, region_names = build_region_mask(lat_vals, lon_vals)
    nregions = len(region_names)
    # dayofyear coordinate
    dayofyear = np.arange(1, 366)
    log(f"Found grid and {nregions} regions: {region_names}")
else:
    tar_files = None
    nlat = nlon = None
    lat_vals = lon_vals = None
    region_mask = None
    nregions = None
    region_names = None
    dayofyear = None

# Broadcast metadata to all ranks
tar_files = comm.bcast(tar_files, root=0)
nlat = comm.bcast(nlat, root=0)
nlon = comm.bcast(nlon, root=0)
lat_vals = comm.bcast(lat_vals, root=0)
lon_vals = comm.bcast(lon_vals, root=0)
region_mask = comm.bcast(region_mask, root=0)
nregions = comm.bcast(nregions, root=0)
region_names = comm.bcast(region_names, root=0)
dayofyear = comm.bcast(dayofyear, root=0)

# Prepare output dataset on root
if rank == 0:
    ds_out = xr.Dataset(coords={
        "dayofyear": ("dayofyear", dayofyear),
        "ecoregion": ("ecoregion", region_names),
        "front": ("front", ["no", "weak", "strong"])
    })

# Process each variable: each rank computes daily_stack for the tarfiles it is assigned
for var in VARS:
    tar_files = list_tar_files(var)
    if not tar_files:
        raise SystemExit(f"No tar files found for Var {var}.")
    # distribute tar files round-robin
    my_tars = tar_files[rank::size]
    log(f"Var {var} - processing {len(my_tars)} tar(s) on this rank")
    local_sum, local_sumsq, local_cnt = process_tar_files_daily(my_tars, nlat, nlon, lat_vals, lon_vals, region_mask, nregions, var)

    if rank == 0:
        total_sum = np.zeros_like(local_sum)
        total_sumsq = np.zeros_like(local_sumsq)
        total_cnt = np.zeros_like(local_cnt)
    else:
        total_sum = None
        total_sumsq = None
        total_cnt = None

    comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_sumsq, total_sumsq, op=MPI.SUM, root=0)
    comm.Reduce(local_cnt, total_cnt, op=MPI.SUM, root=0)

    if rank == 0:
        # compute mean and std (avoid divide by zero)
        mean_clim = np.full_like(total_sum, np.nan)
        std_clim = np.full_like(total_sum, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_clim = total_sum / total_cnt
            var_clim  = (total_sumsq / total_cnt) - (mean_clim ** 2)
            # numerical negative rounding tolerance
            var_clim = np.where(var_clim < 0, 0.0, var_clim)
            std_clim = np.sqrt(var_clim)
        
        ds_out[f"{var}_ecoregion_mean"] = (("dayofyear","ecoregion","front"), mean_clim)
        ds_out[f"{var}_ecoregion_std"] = (("dayofyear","ecoregion","front"), std_clim)

        log(f"[DONE] Processed variable {var}")
    # sync before next variable
    comm.Barrier()

# Root writes NetCDF
if rank == 0:
    # add metadata
    ds_out.attrs["description"] = "Daily climatology (dayofyear 1..365) per MEOW ecoregion and front class"
    ds_out.attrs["front_classes"] = "no (<5), weak (5-10), strong (>=10)"
    ds_out.to_netcdf(OUTFILE)
    log(f"[DONE] Saved daily climatology to {OUTFILE}")

# finalize
comm.Barrier()
if rank == 0:
    log("ALL DONE")


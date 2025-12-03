#!/usr/bin/env python3
"""
MPI-parallel script to compute longitudinal (zonal) means of P_l
classified by HI fronts (no / weak / strong). Fixes filename parsing
and regrids curvilinear HI (nav_lat/nav_lon) to the P_l rectilinear grid
using nearest-neighbour KDTree.

Output: NetCDF with variables:
 - chl_no_fronts(lon)
 - chl_weak_fronts(lon)
 - chl_strong_fronts(lon)
"""
import os
import re
import tarfile
from glob import glob
import numpy as np
import xarray as xr
from mpi4py import MPI
from scipy.spatial import cKDTree as KDTree

# ---------------- Settings ----------------
BASE = "/g100_work/OGS_test2528/plazzari/Neccton_hindcast1999_2022/v22/wrkdir/MODEL/AVE_FREQ_1_tar"
HI_BASE = "/g100_scratch/userexternal/gocchipi/HI_index/DATA"
VAR = "P_l"
OUTFILE = "/g100_scratch/userexternal/gocchipi/HI_index/RESULTS/longitudinal_chl_over_fronts.nc"
# -------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    print(f"[rank {rank}] {msg}", flush=True)

# ---------------- Helpers ----------------
def list_tar_files(base):
    years = sorted(glob(base + "/*"))
    tars = []
    for y in years:
        matches = sorted(glob(f"{y}/{VAR}.tar"))
        tars.extend(matches)
    return sorted(tars)

def sample_grid(tar_path, var=VAR):
    """Return (nlat, nlon, lat_vals, lon_vals) taken from first file inside tar."""
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".nc")]
        members = sorted(members, key=lambda m: m.name)
        if not members:
            raise RuntimeError("No .nc members in sample tar.")
        f = tar.extractfile(members[0])
        ds = xr.open_dataset(f, decode_times=False)
        # find lat/lon coords robustly
        # prefer 1D 'lat','lon' keys, fallback to first coord containing 'lat'/'lon'
        lat_keys = [k for k in ds.coords if re.search("(^|_)lat($|_)", k)]
        lon_keys = [k for k in ds.coords if re.search("(^|_)lon($|_)", k)]
        if lat_keys and lon_keys:
            lat = ds[lat_keys[0]].values
            lon = ds[lon_keys[0]].values
        else:
            raise KeyError("Could not find lat/lon coords in sample file.")
        ds.close()
        # nlat/nlon: if lat/lon 1D -> sizes; if 2D -> shape
        if lat.ndim == 1 and lon.ndim == 1:
            return lat.size, lon.size, lat, lon
        elif lat.ndim == 2:
            return lat.shape[0], lat.shape[1], lat, lon
        else:
            return lat.shape[0], lon.shape[1], lat, lon

date8_re = re.compile(r"(?<!\d)(19|20)\d{6}")  # matches YYYYMMDD (starting 19xx or 20xx)

def extract_yyyymmdd_from_name(name):
    """
    Extract YYYYMMDD from a filename or member.name robustly.
    Returns string 'YYYYMMDD' or None if not found.
    """
    # 1) look for YYYYMMDD digits
    m = date8_re.search(name)
    if m:
        return m.group(0)
    # 2) fallback: look for patterns like YYYY-MM-DD or YYYY_MM_DD
    m2 = re.search(r"(19|20)\d{2}[-_]\d{2}[-_]\d{2}", name)
    if m2:
        s = m2.group(0)
        return s.replace("-", "").replace("_", "")
    # 3) fallback: try to find any 8-digit sequence
    m3 = re.search(r"(?<!\d)(\d{8})(?!\d)", name)
    if m3:
        return m3.group(1)
    return None

def regrid_hi_to_pl(chl_ds, hi_ds, var=VAR):
    """KDTree NN: map HI(nav_lon,nav_lat) onto chl's rectilinear lat/lon grid.

    Returns: chl_vals (2D), hi_on_pl (2D), lon_pl (1D), lat_pl (1D)
    """
    # --- chl grid & data (surface) ---
    if var not in chl_ds:
        raise KeyError(f"{var} not found in chl file.")
    chl = chl_ds[var]
    if "depth" in chl.dims:
        chl = chl.isel(depth=0)
    elif "deptht" in chl.dims:
        chl = chl.isel(deptht=0)
    chl = chl.squeeze()

    # pick rectilinear lat/lon of chl: prefer 1D arrays 'lat' and 'lon'
    if "lat" in chl_ds.coords and "lon" in chl_ds.coords:
        lat_pl = chl_ds["lat"].values
        lon_pl = chl_ds["lon"].values
    else:
        # attempt nav_lat/nav_lon or other names
        lat_candidates = [k for k in chl_ds.coords if "lat" in k]
        lon_candidates = [k for k in chl_ds.coords if "lon" in k]
        if lat_candidates and lon_candidates:
            lat_pl = chl_ds[lat_candidates[0]].values
            lon_pl = chl_ds[lon_candidates[0]].values
        else:
            raise KeyError("Could not find lat/lon coords in P_l file.")

    chl_vals = chl.values

    # --- HI data (curvilinear) ---
    if "nav_lat" not in hi_ds.coords or "nav_lon" not in hi_ds.coords:
        # attempt other names
        lat_keys = [k for k in hi_ds.coords if "lat" in k]
        lon_keys = [k for k in hi_ds.coords if "lon" in k]
        if "nav_lat" in hi_ds.coords:
            nav_lat = hi_ds["nav_lat"].values
            nav_lon = hi_ds["nav_lon"].values
        elif lat_keys and lon_keys:
            nav_lat = hi_ds[lat_keys[0]].values
            nav_lon = hi_ds[lon_keys[0]].values
        else:
            raise KeyError("HI dataset has no nav_lat/nav_lon coords.")
    else:
        nav_lat = hi_ds["nav_lat"].values
        nav_lon = hi_ds["nav_lon"].values

    hi_varname = None
    for c in ("HI", "hi", "HI_index"):
        if c in hi_ds:
            hi_varname = c
            break
    if hi_varname is None:
        # fallback to first data var
        hi_varname = list(hi_ds.data_vars)[0]
    hi_vals_full = hi_ds[hi_varname].squeeze().values

    # bounding box select on nav coords
    lat_min, lat_max = float(np.min(lat_pl)), float(np.max(lat_pl))
    lon_min, lon_max = float(np.min(lon_pl)), float(np.max(lon_pl))

    # build mask for candidate HI points inside bounding box and finite nav coords and finite hi values
    valid_nav = np.isfinite(nav_lat) & np.isfinite(nav_lon) & np.isfinite(hi_vals_full)
    bbox_mask = (nav_lat >= lat_min) & (nav_lat <= lat_max) & (nav_lon >= lon_min) & (nav_lon <= lon_max)
    sel_mask = valid_nav & bbox_mask

    if not np.any(sel_mask):
        raise RuntimeError(f"No HI grid points fall inside P_l bbox: lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")

    pts_hi = np.column_stack((nav_lon[sel_mask].ravel(), nav_lat[sel_mask].ravel()))
    vals_hi = hi_vals_full[sel_mask].ravel()
    if pts_hi.shape[0] == 0:
        raise RuntimeError("No HI points after bbox & finite filtering.")

    # build KDTree from HI subset
    tree = KDTree(pts_hi)

    # build target points (chl rectilinear grid). If lat_pl/lon_pl are 1D -> meshgrid
    if lat_pl.ndim == 1 and lon_pl.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_pl, lat_pl)
    elif lat_pl.ndim == 2 and lon_pl.ndim == 2:
        # if chl somehow provided 2D coords, use them directly
        lon2d, lat2d = lon_pl, lat_pl
    else:
        # try to build meshgrid if possible
        try:
            lon2d, lat2d = np.meshgrid(lon_pl, lat_pl)
        except Exception:
            raise RuntimeError("Unable to construct chl target grid (lat/lon shapes).")

    pts_pl = np.column_stack((lon2d.ravel(), lat2d.ravel()))

    # query tree
    _, idx = tree.query(pts_pl, k=1)
    hi_on_pl = vals_hi[idx].reshape(lon2d.shape)

    return chl_vals, hi_on_pl, lon_pl, lat_pl

def process_tar_files(tar_list, nlat, nlon, lat_vals, lon_vals, var=VAR):
    """Accumulate sums/counts per class (no/weak/strong fronts)."""
    sum_arr = np.zeros((3, nlat, nlon), dtype=np.float64)
    cnt_arr = np.zeros((3, nlat, nlon), dtype=np.int64)

    for tar_path in tar_list:
        year = os.path.basename(os.path.dirname(tar_path))
        log(f"Processing tar: {tar_path} (year {year})")
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".nc")]
            members = sorted(members, key=lambda m: m.name)
            log(f"  Found {len(members)} members")

            for member in members:
                fname = os.path.basename(member.name)
                # robust date extraction
                yyyymmdd = extract_yyyymmdd_from_name(member.name)
                if yyyymmdd is None:
                    yyyymmdd = extract_yyyymmdd_from_name(fname)
                if yyyymmdd is None:
                    # try alternative: look for 8-digit anywhere in fname
                    log(f"    [WARN] Could not extract date from member name '{member.name}' -> skipping")
                    continue

                yyyy, mm, dd = yyyymmdd[:4], yyyymmdd[4:6], yyyymmdd[6:8]
                hi_file = f"{HI_BASE}/{yyyy}/{mm}/HI{yyyymmdd}-12:00:00.nc"

                # debug
                log(f"    Member: {member.name}  -> date {yyyymmdd}  -> HI path {hi_file}")

                # open chl member
                f = tar.extractfile(member)
                if f is None:
                    log("    extractfile returned None -> skipping")
                    continue
                ds_chl = xr.open_dataset(f, decode_times=False)

                # check HI file exists
                if not os.path.exists(hi_file):
                    log(f"    [WARN] HI file missing: {hi_file} -> skipping this day")
                    ds_chl.close()
                    continue

                ds_hi = xr.open_dataset(hi_file, decode_times=False)

                try:
                    chl_vals, hi_on_pl, _, _ = regrid_hi_to_pl(ds_chl, ds_hi, var)
                except Exception as e:
                    log(f"    [ERROR] Regridding failed for {member.name}: {e}")
                    ds_chl.close()
                    ds_hi.close()
                    continue

                ds_chl.close()
                ds_hi.close()

                # classification masks
                no_mask = (hi_on_pl < 5)
                weak_mask = (hi_on_pl >= 5) & (hi_on_pl < 10)
                strong_mask = (hi_on_pl >= 10)

                # ensure shapes match
                if chl_vals.shape != hi_on_pl.shape:
                    # try transpose fallback
                    if chl_vals.T.shape == hi_on_pl.shape:
                        chl_vals = chl_vals.T
                        log("    transposed chl to match HI shape")
                    else:
                        log(f"    [WARN] shape mismatch chl {chl_vals.shape} vs hi {hi_on_pl.shape} -> skipping")
                        continue

                # accumulate sums & counts
                for i, mask in enumerate([no_mask, weak_mask, strong_mask]):
                    valid = np.isfinite(chl_vals) & mask
                    sum_arr[i] += np.where(valid, chl_vals, 0.0)
                    cnt_arr[i] += valid.astype(np.int64)

                # debug counts and ranges
                n_no = int(np.sum(no_mask))
                n_weak = int(np.sum(weak_mask))
                n_strong = int(np.sum(strong_mask))
                log(f"    {fname}: chl mean={np.nanmean(chl_vals):.6g}, HI(min/max)={np.nanmin(hi_on_pl):.3g}/{np.nanmax(hi_on_pl):.3g}, counts(no/weak/strong)={n_no}/{n_weak}/{n_strong}")

    return sum_arr, cnt_arr

# ---------------- Main ----------------
if rank == 0:
    tar_files = list_tar_files(BASE)
    if not tar_files:
        raise SystemExit("No tar files found in BASE.")
    nlat, nlon, lat_vals, lon_vals = sample_grid(tar_files[0])
    log(f"Sample grid: nlat={nlat}, nlon={nlon}")
else:
    tar_files = None
    nlat = nlon = lat_vals = lon_vals = None

# Broadcast metadata
tar_files = comm.bcast(tar_files, root=0)
nlat = comm.bcast(nlat, root=0)
nlon = comm.bcast(nlon, root=0)
lat_vals = comm.bcast(lat_vals, root=0)
lon_vals = comm.bcast(lon_vals, root=0)

log(f"Total tars: {len(tar_files)}; this rank will process {len(tar_files[rank::size])} tars")

# Distribute tar files to ranks (round-robin)
my_tars = tar_files[rank::size]

# Local accumulators
local_sum, local_cnt = process_tar_files(my_tars, nlat, nlon, lat_vals, lon_vals, VAR)

# Reduce to root
if rank == 0:
    total_sum = np.zeros_like(local_sum)
    total_cnt = np.zeros_like(local_cnt)
else:
    total_sum = None
    total_cnt = None

comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
comm.Reduce(local_cnt, total_cnt, op=MPI.SUM, root=0)

# Root: finalize and save
if rank == 0:
    log("Reduction done; computing means.")
    with np.errstate(divide="ignore", invalid="ignore"):
        means = total_sum / total_cnt
    means[~np.isfinite(means)] = np.nan

    # print some diagnostics
    for i, name in enumerate(["no", "weak", "strong"]):
        total_count = np.nansum(total_cnt[i])
        log(f"Total non-NaN count for class '{name}': {int(total_count)}")

    # Average over latitude axis -> (3, nlon)
    # if means shape is (3, nlat, nlon) this collapses axis=1
    zonal_means = np.nanmean(means, axis=1)

    # prepare longitude coordinate (use lon_vals if 1D)
    if isinstance(lon_vals, np.ndarray) and lon_vals.ndim == 1:
        lon_coord = lon_vals
    else:
        # flatten if 2D or fallback to range
        try:
            lon_coord = np.asarray(lon_vals).ravel()
            # if too many points, pick unique sorted
            lon_coord = np.unique(lon_coord)
        except Exception:
            lon_coord = np.arange(zonal_means.shape[1])

    ds_out = xr.Dataset(
        {
            "chl_no_fronts": (("lon",), zonal_means[0, :]),
            "chl_weak_fronts": (("lon",), zonal_means[1, :]),
            "chl_strong_fronts": (("lon",), zonal_means[2, :]),
        },
        coords={"lon": lon_coord},
    )

    ds_out.to_netcdf(OUTFILE)
    log(f"[DONE] Saved output to {OUTFILE}")

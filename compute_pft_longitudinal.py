import os
import tarfile
from glob import glob
import numpy as np
import xarray as xr
from mpi4py import MPI

# ---------------- Settings ----------------
BASE = "/g100_work/OGS_test2528/plazzari/Neccton_hindcast1999_2022/v22/wrkdir/MODEL/AVE_FREQ_1_tar"
VARS = ["P1l", "P2l", "P3l", "P4l"]
OUTFILE = "/g100_scratch/userexternal/gocchipi/HI_index/RESULTS/longitudinal_mean_P_groups.nc"
# -------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------- Helpers ----------------
def list_tar_files(base, var):
    years = sorted(glob(base + "/*"))
    tars = []
    for y in years:
        matches = sorted(glob(f"{y}/{var}.tar"))
        tars.extend(matches)
    return sorted(tars)

def sample_grid(tar_path, var):
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".nc")]
        members = sorted(members, key=lambda m: m.name)
        f = tar.extractfile(members[0])
        ds = xr.open_dataset(f)
        lat = ds["lat"].values
        lon = ds["lon"].values
        ds.close()
        return len(lat), len(lon), lat, lon

def process_tar_files(tar_list, nlat, nlon, var):
    sum_arr = np.zeros((nlat, nlon), dtype=np.float64)
    cnt_arr = np.zeros((nlat, nlon), dtype=np.int64)

    for tar_path in tar_list:
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".nc")]
            members = sorted(members, key=lambda m: m.name)

            for member in members:
                f = tar.extractfile(member)
                if f is None:
                    continue
                ds = xr.open_dataset(f, decode_times=False)

                # select surface
                if "depth" in ds[var].dims:
                    da = ds[var].isel(depth=0).squeeze()
                elif "deptht" in ds[var].dims:
                    da = ds[var].isel(deptht=0).squeeze()
                else:
                    da = ds[var].squeeze()

                data = da.values
                ds.close()

                valid = np.isfinite(data)
                sum_arr += np.where(valid, data, 0.0)
                cnt_arr += valid.astype(np.int64)

    return sum_arr, cnt_arr

# ---------------- Main ----------------
results = {}
for var in VARS:
    if rank == 0:
        tar_files = list_tar_files(BASE, var)
        if not tar_files:
            raise SystemExit(f"No tar files found for {var}")
        nlat, nlon, lat_vals, lon_vals = sample_grid(tar_files[0], var)
    else:
        tar_files = None
        nlat = nlon = lat_vals = lon_vals = None

    # broadcast
    tar_files = comm.bcast(tar_files, root=0)
    nlat = comm.bcast(nlat, root=0)
    nlon = comm.bcast(nlon, root=0)
    lat_vals = comm.bcast(lat_vals, root=0)
    lon_vals = comm.bcast(lon_vals, root=0)

    # distribute
    my_tars = tar_files[rank::size]

    # local accumulators
    local_sum, local_cnt = process_tar_files(my_tars, nlat, nlon, var)

    # global accumulators
    if rank == 0:
        total_sum = np.zeros_like(local_sum)
        total_cnt = np.zeros_like(local_cnt)
    else:
        total_sum = None
        total_cnt = None

    comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_cnt, total_cnt, op=MPI.SUM, root=0)

    if rank == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = total_sum / total_cnt
        mean[~np.isfinite(mean)] = np.nan

        # average over latitude
        zonal_mean = np.nanmean(mean, axis=0)  # shape (nlon,)
        results[var] = zonal_mean

# ---------------- Save ----------------
if rank == 0:
    ds_out = xr.Dataset(
        {var: (("lon",), results[var]) for var in VARS},
        coords={"lon": lon_vals},
    )
    ds_out.to_netcdf(OUTFILE)
    print(f"Saved longitudinal means for {VARS} to {OUTFILE}")

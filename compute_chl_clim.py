
#!/usr/bin/env python3
import tarfile
from glob import glob
import numpy as np
import xarray as xr
from mpi4py import MPI
from netCDF4 import Dataset

# ---------------- Settings ----------------
BASE = "/g100_work/OGS_test2528/plazzari/Neccton_hindcast1999_2022/v22/wrkdir/MODEL/AVE_FREQ_1_tar"
VAR = "P_l"
OUTFILE = "/g100_scratch/userexternal/gocchipi/HI_index/RESULTS/chl_surface_mean.nc"
# -------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------- Helpers ----------------
def list_tar_files(base):
    """List all tar files (one per year)."""
    years = sorted(glob(base + "/*"))
    tars = []
    for y in years:
        matches = sorted(glob(f"{y}/{VAR}.tar"))
        tars.extend(matches)
    return sorted(tars)

def sample_grid(tar_path, var=VAR):
    """Get grid shape and coordinates from first file in tar."""
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".nc")]
        members = sorted(members, key=lambda m: m.name)
        f = tar.extractfile(members[0])
        ds = xr.open_dataset(f)
        lat = ds["lat"].values
        lon = ds["lon"].values
        ds.close()
        return len(lat), len(lon), lat, lon

def process_tar_files(tar_list, nlat, nlon, var=VAR):
    """Accumulate sums and counts over all files in tar_list."""
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
                # Select surface layer
                if "depth" in ds[var].dims:
                    da = ds[var].isel(depth=0).squeeze("time")
                elif "deptht" in ds[var].dims:
                    da = ds[var].isel(deptht=0).squeeze("time")
                else:
                    da = ds[var].squeeze("time")

                data = da.values
                valid = np.isfinite(data)
                sum_arr += np.where(valid, data, 0.0)
                cnt_arr += valid.astype(np.int64)
                ds.close()
    return sum_arr, cnt_arr

# ---------------- Main ----------------
if rank == 0:
    tar_files = list_tar_files(BASE)
    if not tar_files:
        raise SystemExit("No tar files found!")
    nlat, nlon, lat_vals, lon_vals = sample_grid(tar_files[0])
else:
    tar_files = None
    nlat = nlon = lat_vals = lon_vals = None

# Broadcast metadata
tar_files = comm.bcast(tar_files, root=0)
nlat = comm.bcast(nlat, root=0)
nlon = comm.bcast(nlon, root=0)
lat_vals = comm.bcast(lat_vals, root=0)
lon_vals = comm.bcast(lon_vals, root=0)

# Distribute tar files: one year per rank (round-robin if more years than ranks)
my_tars = tar_files[rank::size]

# Local accumulators
local_sum, local_cnt = process_tar_files(my_tars, nlat, nlon, VAR)

# Reduce across ranks
if rank == 0:
    total_sum = np.zeros_like(local_sum)
    total_cnt = np.zeros_like(local_cnt)
else:
    total_sum = None
    total_cnt = None

comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
comm.Reduce(local_cnt, total_cnt, op=MPI.SUM, root=0)

# Rank 0: compute mean and write NetCDF
if rank == 0:
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total_sum / total_cnt
    mean[~np.isfinite(mean)] = np.nan

    nc_out = Dataset(OUTFILE, "w", format="NETCDF4")
    nc_out.createDimension("lat", nlat)
    nc_out.createDimension("lon", nlon)
    lat_var = nc_out.createVariable("lat", "f4", ("lat",))
    lon_var = nc_out.createVariable("lon", "f4", ("lon",))
    mean_var = nc_out.createVariable(VAR, "f4", ("lat", "lon"), zlib=True, complevel=4)

    lat_var[:] = lat_vals
    lon_var[:] = lon_vals
    mean_var[:] = mean.astype(np.float32)
    nc_out.close()
    print(f"Mean field saved to {OUTFILE}")

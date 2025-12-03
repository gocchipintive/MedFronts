#compute_fronts_frequency.py
from mpi4py import MPI
import xarray as xr
import numpy as np
import glob
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -----------------------------
# User parameters
# -----------------------------
indir = "/g100_scratch/userexternal/gocchipi/HI_index/DATA/"
varname = "HI"

outdir = "/g100_scratch/userexternal/gocchipi/HI_index/RESULTS/"
os.makedirs(outdir, exist_ok=True)

# -----------------------------
# Collect file list (rank 0)
# -----------------------------
if rank == 0:
    # recursive glob to get all files
    files = sorted(glob.glob(os.path.join(indir, "*/*/*.nc")))
    print(f"[rank 0] Found {len(files)} files.")
else:
    files = None

# Scatter files across ranks
files = comm.bcast(files, root=0)
my_files = files[rank::size]  # each rank processes a slice

# -----------------------------
# Processing function
# -----------------------------
def process_file(fname, varname="HI"):
    ds = xr.open_dataset(fname)
    da = ds[varname]  # (time, lat, lon)

    # Boolean masks
    mask_5_10 = (da >= 5) & (da < 10)
    mask_gt10 = (da >= 10)

    # Count occurrences along time dimension
    count_5_10 = mask_5_10.sum(dim="time_counter", skipna=True)
    count_gt10 = mask_gt10.sum(dim="time_counter", skipna=True)
    count_total = (~da.isnull()).sum(dim="time_counter")

    ds.close()
    return count_5_10, count_gt10, count_total

# -----------------------------
# Local accumulation
# -----------------------------
local_5_10 = None
local_gt10 = None
local_total = None

for f in my_files:
    c1, c2, ctot = process_file(f, varname)
    if local_5_10 is None:
        local_5_10 = c1
        local_gt10 = c2
        local_total = ctot
    else:
        local_5_10 = local_5_10 + c1
        local_gt10 = local_gt10 + c2
        local_total = local_total + ctot

# -----------------------------
# Gather results on root
# -----------------------------
all_5_10 = comm.gather(local_5_10, root=0)
all_gt10 = comm.gather(local_gt10, root=0)
all_total = comm.gather(local_total, root=0)

if rank == 0:
    # Reduce by summation
    total_5_10 = all_5_10[0]
    total_gt10 = all_gt10[0]
    total_count = all_total[0]

    for i in range(1, size):
        total_5_10 = total_5_10 + all_5_10[i]
        total_gt10 = total_gt10 + all_gt10[i]
        total_count = total_count + all_total[i]

    # Compute frequencies
    freq_5_10 = total_5_10 / total_count
    freq_gt10 = total_gt10 / total_count

    # Save to NetCDF
    freq_5_10.to_netcdf(os.path.join(outdir, "frequency_HI_5_10.nc"))
    freq_gt10.to_netcdf(os.path.join(outdir, "frequency_HI_gt10.nc"))

    print("[rank 0] Finished. Output written in:", outdir)

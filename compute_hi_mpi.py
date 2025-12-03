# compute_hi_mpi.py
from mpi4py import MPI
import os
import sys
import glob
import xarray as xr
import tarfile
import heterogeneity_index as HI
import numpy as np
from typing import Sequence, Any, cast
from scipy import stats
from scipy.ndimage import binary_dilation
# ---------------------------
# User settings
# ---------------------------
PATH = "/g100_work/OGS23_PRACE_IT/plazzari/NECCTON/Neccton_hindcast1999_2022_v22/wrkdir/MODEL/FORCINGS/"
VARNAME = "votemper"
FILE_GLOB = "????/??/T*.nc"
OUTDIR = "/g100_scratch/userexternal/gocchipi/HI_index/DATA/"
WSIZE = 7

# ---------------------------
# MPI setup
# ---------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg, root_only=False):
    if (not root_only) or (root_only and rank == 0):
        print(f"[rank {rank}] {msg}", flush=True)


def compute_coefficient_hi(
    components: xr.Dataset | Sequence[np.ndarray | xr.DataArray],
    coefficients: dict[str, float],
    quantile_target: float = 0.95,
    hi_limit: float = 9.5,
    bins: np.ndarray | None = None,
    **kwargs: Any,
) -> float:
    """
    Compute the normalization coefficient for the HI using NumPy instead of xhistogram,
    safely handling NaNs.

    Returns np.nan if no valid HI values exist.
    """
    from heterogeneity_index import apply_coefficients
    COMPONENTS_NAMES = ["stdev", "skew", "bimod"]  # same as in heterogeneity_index.components
    # Convert xarray.Dataset to a tuple of arrays
    if isinstance(components, xr.Dataset):
        components = tuple(components[name] for name in COMPONENTS_NAMES)
    components = cast(Sequence[np.ndarray | xr.DataArray], components)

    # Compute un-normalized HI
    hi = apply_coefficients(components, coefficients)

    # Flatten and remove NaNs
    hi_values = np.ravel(hi.values if isinstance(hi, xr.DataArray) else hi)
    hi_values = hi_values[np.isfinite(hi_values)]

    if hi_values.size == 0:
        return np.nan  # nothing to compute

    # Histogram bins
    if bins is None:
        bins = np.linspace(0.0, 80.0, 801)

    # Compute histogram using NumPy
    hist, bin_edges = np.histogram(hi_values, bins=bins, density=False)

    # Convert histogram to a cumulative distribution function (CDF)
    cdf = np.cumsum(hist).astype(float)
    cdf /= cdf[-1]  # normalize to [0,1]

    # Find HI value corresponding to quantile_target
    idx = np.searchsorted(cdf, quantile_target, side='left')
    current_hi = bin_edges[idx] if idx < len(bin_edges) else bin_edges[-1]

    # Compute normalization coefficient
    coef = hi_limit / current_hi if current_hi > 0 else np.nan

    return coef


# ---------------------------
# HI computation
# ---------------------------
def compute_hi_lat_lon(data, varname, wsize=7,ocean_mask=None):
    var_data = data[varname]
    # tranform the 0 in nan
    var_data = var_data.where(var_data != 0)
    #keep only surface layer if depth is present
    if "deptht" in var_data.dims:
        var_data = var_data.isel(deptht=0)
    if "lon" in var_data.dims and "lat" in var_data.dims:
        components = HI.compute_components_xarray(
            input_field=var_data,
            window_size={"lon": wsize, "lat": wsize},
            bins_shift=False,
        )
        #apply ocean mask to components to remove coastal artifacts
        components = components.where(ocean_mask)
        coefficients = HI.compute_coefficients_components(components)
        #coefficients["HI"] = HI.compute_coefficient_hi(components.dropna(dim="lat",how="any").dropna(dim="lon",how="any"), coefficients)
        coefficients["HI"] = compute_coefficient_hi(components, coefficients)
   
    elif "longitude" in var_data.dims and "latitude" in var_data.dims:
        components = HI.compute_components_xarray(
            input_field=var_data,
            window_size={"longitude": wsize, "latitude": wsize},
            bins_shift=False,
        )
        coefficients = HI.compute_coefficients_components(components)
        #coefficients["HI"] = HI.compute_coefficient_hi(components.dropna(dim="latitude",how="any").dropna(dim="longitude",how="any"), coefficients)
        coefficients["HI"] = compute_coefficient_hi(components, coefficients)

    elif "x" in var_data.dims and "y" in var_data.dims:
        components = HI.compute_components_xarray(
            input_field=var_data,
            window_size={"x": wsize, "y": wsize},
            bins_shift=False,
        )
        coefficients = HI.compute_coefficients_components(components)
        #coefficients["HI"] = HI.compute_coefficient_hi(components.dropna(dim="y",how="any").dropna(dim="x",how="any"), coefficients)
        coefficients["HI"] = compute_coefficient_hi(components, coefficients)
    else:
        raise ValueError("The input data must contain lon/lat or longitude/latitude or x/y dims.")

    hi = HI.apply_coefficients(components, coefficients)
    return hi

# ---------------------------
# Helpers
# ---------------------------
def safe_makedirs(path):
    if rank == 0:
        os.makedirs(path, exist_ok=True)
    comm.Barrier()

def split_work(items, n_parts):
    L = len(items)
    base = L // n_parts
    rem = L % n_parts
    chunks = []
    start = 0
    for i in range(n_parts):
        extra = 1 if i < rem else 0
        end = start + base + extra
        chunks.append(items[start:end])
        start = end
    return chunks
#mask coordinates with depth smaller than Xm
def mask_shallow(ds: xr.Dataset):
    """
    Create a mask for shallow coastal areas by dilating land points of 2 grid points.
    """
    
    temp = ds.votemper.isel(time_counter=0, deptht=0)
    temp = temp.where(temp != 0)
    # Build ocean mask (True = ocean, False = land)
    land_mask = temp.isnull()

    # Dilate land mask to "grow" it by N grid points (coastal band)
    # Here we expand land by 2 grid cells in all directions
    dilated_land = binary_dilation(land_mask, iterations=2)

    return ~dilated_land


def build_output_path(infile):
    """
    Place output under OUTDIR/yyyy/mm/, preserving year/month structure.
    Replace 'T' with 'HI' if present, otherwise append '_HI'.
    """
    rel = os.path.relpath(infile, PATH).lstrip("./")
    parts = rel.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"Unexpected path structure for {infile}")
    year, month, basename = parts[0], parts[1], parts[2]

    if "T" in basename:
        outname = basename.replace("T", "HI")
    else:
        root, ext = os.path.splitext(basename)
        outname = f"{root}_HI{ext}"

    outdir = os.path.join(OUTDIR, year, month)
    outpath = os.path.join(outdir, outname)
    return outdir, outpath

def process_file(filename,ocean_mask=None):
    outdir, outpath = build_output_path(filename)
    os.makedirs(outdir, exist_ok=True)

    if os.path.exists(outpath):
        log(f"Exists, skipping: {outpath}")
        return "skipped"

    try:
        with xr.open_dataset(filename) as ds:
            if VARNAME not in ds.variables and VARNAME not in ds.data_vars:
                raise KeyError(f"Variable '{VARNAME}' not found in {filename}")
            
            hi = compute_hi_lat_lon(ds, varname=VARNAME, wsize=WSIZE, ocean_mask=ocean_mask)

            tmp_out = outpath + f".rank{rank}.tmp"
            hi.to_netcdf(tmp_out)
            os.replace(tmp_out, outpath)

        log(f"Wrote: {outpath}")
        return "ok"
    except Exception as e:
        log(f"ERROR on {filename}: {e}")
        return f"error: {e}"

# ---------------------------
# Main
# ---------------------------
def main():
    if rank == 0:
        file_list = sorted(glob.glob(os.path.join(PATH, FILE_GLOB)))
        if not file_list:
            print("No files matched. Check PATH/FILE_GLOB.", flush=True)
        #compute the shallow mask from the first file only once
        ocean_mask = mask_shallow(xr.open_dataset(file_list[0], decode_times=False))
    else:
        file_list  = None
        ocean_mask = None

    file_list = comm.bcast(file_list, root=0)
    ocean_mask = comm.bcast(ocean_mask, root=0)
    safe_makedirs(OUTDIR)

    chunks = split_work(file_list, size)
    my_files = chunks[rank]

    log(f"Received {len(my_files)} files.")

    n_ok = n_skip = n_err = 0
    for i, f in enumerate(my_files, 1):
        status = process_file(f,ocean_mask=ocean_mask)
        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skip += 1
        else:
            n_err += 1
        if i % 10 == 0 or i == len(my_files):
            log(f"Progress {i}/{len(my_files)} (ok={n_ok}, skip={n_skip}, err={n_err})")

    totals = comm.gather((n_ok, n_skip, n_err), root=0)
    if rank == 0:
        tot_ok = sum(t[0] for t in totals)
        tot_skip = sum(t[1] for t in totals)
        tot_err = sum(t[2] for t in totals)
        print(f"[SUMMARY] files={len(file_list)} ok={tot_ok} skipped={tot_skip} errors={tot_err}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[rank {rank}] FATAL: {e}", file=sys.stderr, flush=True)
        try:
            MPI.COMM_WORLD.Abort(1)
        except Exception:
            pass
        raise

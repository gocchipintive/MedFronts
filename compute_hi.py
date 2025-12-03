import tarfile
import heterogeneity_index as HI
import xarray as xr
import glob

def compute_hi_lat_lon(data,varname,wsize=7):
    """
    Compute the heterogeneity index (HI) for a given variable in a dataset.
    
    Parameters:
    data (xarray.Dataset): The input dataset containing the variable.
    varname (str): The name of the variable for which to compute the HI.
    wsize (int): The window size for the HI computation. Default is 7.
    
    Returns:
    xarray.DataArray: The computed HI values.
    """
    var_data = data[varname]
    if "lon" in var_data.dims and "lat" in var_data.dims:
        components = HI.compute_components_xarray(input_field=var_data,window_size={"lon": wsize, "lat": wsize},bins_shift=False)
    elif "longitude" in var_data.dims and "latitude" in var_data.dims:
        components = HI.compute_components_xarray(input_field=var_data,window_size={"longitude": wsize, "latitude": wsize},bins_shift=False)
    else:
        raise ValueError("The input data must contain 'lon' and 'lat' or 'longitude' and 'latitude' dimensions.")

    coefficients = HI.compute_coefficients_components(components)
    coefficients["HI"] = HI.compute_coefficient_hi(components.dropna(dim="lat",how="any").dropna(dim="lon",how="any"), coefficients)
    hi = HI.apply_coefficients(components, coefficients)

    return hi

def open_tar(filename):
    tar = tarfile.open(filename)
    for im,member in enumerate(tar.getmembers()):
        print(f'Processing {member.get_info()["name"]}',f' {im+1}/{len(tar.getmembers())*100}%')
        f = tar.extractfile(member)
        ds = xr.open_dataset(f)
    return ds

path = '/g100_work/OGS23_PRACE_IT/plazzari/NECCTON/Neccton_hindcast1999_2022_v22/wrkdir/MODEL/FORCINGS/'
varname = 'votemper'

filenames = glob.glob(path+'??/??/T*.nc')
for filename in filenames:
    ds = xr.open_dataset(filename)
    
    hi = compute_hi_lat_lon(ds,varname=varname,wsize=7)
    
    #save to netcdf
    outdir = "/g100_scratch/userexternal/gocchipi/HI_index/DATA/"
    outname = filename.replace('ave','HI')
    
    hi.to_netcdf(outdir+outname)


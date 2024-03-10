import xarray as xr
import os
import pandas as pd
import zarr

def generate_noleap_date_range(start_date, end_date):
    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date)
        next_day = current_date + pd.Timedelta(days=1)
        if next_day.month == 2 and next_day.day == 29:
            next_day += pd.Timedelta(days=1)
        current_date = next_day

    return date_list

def ensure_zarr_structure_exists(zarr_store, variable_list, chunk_shape):
    for var in variable_list:
        zarr_file = os.path.join(zarr_store, var)
        if not os.path.exists(zarr_file):
            arr = zarr.open_array(zarr_file, mode='w', shape=(0,), chunks=(chunk_shape,), dtype='float32', fill_value=0)
            arr.attrs['_ARRAY_DIMENSIONS'] = ['time']

def preprocess_and_split_dataset(filename, frequent_vars, other_vars):
    try:
        ds = xr.open_dataset(filename)
        freq_ds = ds[frequent_vars]
        other_ds = ds[other_vars]
        return freq_ds, other_ds
    except Exception as e:
        print(f"Error preprocessing file {filename}: {e}")
        return None, None

def append_to_zarr(ds, zarr_group_path, chunk_shape):
    if ds is not None and ds.variables:
        zarr_group = zarr.open_group(zarr_group_path, mode='a')

        for var in ds.data_vars:
            var_data = ds[var].load()  # Ensure data is loaded into memory

            # Check if the variable already exists in the group
            if var in zarr_group:
                # Append data to the existing array within the group
                zarr_array = zarr_group[var]
                # Calculate the new shape, extending only along the 'time' dimension
                new_shape = list(zarr_array.shape)
                new_shape[0] += var_data.shape[0]  # Assuming 'time' is the first dimension
                zarr_array.resize(new_shape)

                # Append the new data
                zarr_array[-var_data.shape[0]:] = var_data.values
            else:
                # Create a new array for the variable within the group
                # The shape and chunks should match the data dimensions
                zarr_group.create_dataset(var, data=var_data.values, shape=var_data.shape, chunks=chunk_shape, dtype=var_data.dtype)



def process_file(filename, zarr_store, frequent_vars, other_vars, freq_chunk_shape, other_chunk_shape):
    print(f'Processing file: {filename}')
    freq_ds, other_ds = preprocess_and_split_dataset(filename, frequent_vars, other_vars)
    append_to_zarr(freq_ds, os.path.join(zarr_store, 'frequent_vars'), freq_chunk_shape)
    append_to_zarr(other_ds, os.path.join(zarr_store, 'other_vars'), other_chunk_shape)

if __name__ == "__main__":
    netcdf_dir = '/tmpdata/i.e215.I2000Clm50SpGs.hw_production.02/'
    zarr_path = '/tmpdata/zarr/i.e215.I2000Clm50SpGs.hw_production.02'

    start_date = pd.to_datetime('1986-01-01')
    end_date = pd.to_datetime('1986-12-31')
    date_range = generate_noleap_date_range(start_date, end_date)
    netcdf_filenames = [os.path.join(netcdf_dir, f"i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{date.strftime('%Y-%m-%d')}-00000.nc") for date in date_range]

    core_vars = ['TSA', 'TSA_R', 'TSA_U', 'WBA', 'WBA_R', 'WBA_U']
    other_vars = [var for var in xr.open_dataset(netcdf_filenames[0]).data_vars if var not in core_vars]

    yearly_chunk_size = 24 * 365  # 8760 hours for a non-leap year

    ensure_zarr_structure_exists(os.path.join(zarr_path, 'frequent_vars'), core_vars, yearly_chunk_size)
    ensure_zarr_structure_exists(os.path.join(zarr_path, 'other_vars'), other_vars, yearly_chunk_size)

    for filename in netcdf_filenames:
        process_file(filename, zarr_path, core_vars, other_vars, (yearly_chunk_size, len(core_vars)), (yearly_chunk_size, 1))

import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import yaml

import psutil
import time
from datetime import datetime

################ Constants ################
# Define global constants
LAT_LEN: int = 192
LON_LEN: int = 288
fahrenheit_threshold: float = 90
kelvin_threshold: float = 305.372


###########################################

def time_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_dt = datetime.now()  # Capture the datetime at start

        result = func(*args, **kwargs)

        end_time = time.time()
        end_dt = datetime.now()  # Capture the datetime at end

        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the output string
        time_str = f"{int(hours):02d}h:{int(minutes):02d}m:{seconds:.2f}s"
        print(
            f"{func.__name__} started at {start_dt.strftime('%Y-%m-%d %H:%M:%S')} and finished at {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{func.__name__} executed in {time_str}")

        return result

    return wrapper


def log_file_status(log_file_path, file_path, status):
    ensure_directory_exists(log_file_path)
    """Logs the status of each file."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')


def looks_like_file(path):
    # A simple heuristic: if the last segment after a split on the OS separator has a dot, it might be a file.
    return '.' in os.path.basename(path)


def ensure_directory_exists(file_path):
    # Extract the directory part of the file path
    if looks_like_file(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path

    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory does not exist, create it
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    # else:
    #     print(f"Directory '{directory}' already exists.")


def round_to_nearest_hour(time_values):
    """Rounds cftime.DatetimeNoLeap objects to the nearest hour."""
    rounded_times = []
    for time_value in time_values:
        year, month, day, hour = time_value.year, time_value.month, time_value.day, time_value.hour
        minute = time_value.minute

        if minute >= 30 and hour < 23:
            hour += 1
        elif minute >= 30 and hour == 23:
            new_day = cftime.DatetimeNoLeap(year, month, day) + cftime.timedelta(days=1)
            year, month, day = new_day.year, new_day.month, new_day.day
            hour = 0

        rounded_time = cftime.DatetimeNoLeap(year, month, day, hour)
        rounded_times.append(rounded_time)

    return np.array(rounded_times)


def substitute_variables(data, parent_data=None):
    if parent_data is None:
        parent_data = data  # Initial call uses the root of the data

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.format(**parent_data)
            else:
                substitute_variables(value, parent_data)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                data[i] = item.format(**parent_data)
            else:
                substitute_variables(item, parent_data)


def load_config(config_file):
    """Loads configuration parameters from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Perform variable substitution
    substitute_variables(config)
    return config


def show_memory_usage(message):
    # Get the current process
    process = psutil.Process(os.getpid())

    # Get the memory usage in bytes
    memory_info = process.memory_info()

    # Convert bytes to megabytes
    memory_usage_mb = memory_info.rss / (1024 * 1024)

    # Print the current memory usage
    print(f"{message} memory usage: {memory_usage_mb:.2f} MB")

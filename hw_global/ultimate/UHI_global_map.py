import pandas as pd
import numpy as np
import xarray as xr
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def setup_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def save_plot(plt, filename, output_dir):
    """Save plot to specified output directory"""
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


THRESHOLD: int = 98
# #  Step 1: Load the data
summary_dir = (
    "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary"
)
merged_feather_path = os.path.join(
    summary_dir, f"updated_local_hour_adjusted_variables_HW{THRESHOLD}.feather"
)

local_hour_adjusted_df = pd.read_feather(merged_feather_path)
local_hour_adjusted_df.info()
local_hour_adjusted_df
location_ID_path = os.path.join(summary_dir, "location_IDs.nc")
location_ID_ds = xr.open_dataset(location_ID_path)
# #  Step 2: Validate the data
# ##  Step 2.0 validate hw_def data
summary_dir = (
    "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary"
)
hw_def_feather_path = os.path.join(summary_dir, f"hw_def_{THRESHOLD}.feather")
hw_def_95_df = pd.read_feather(hw_def_feather_path)
hw_def_95_df.info()
import duckdb
import pandas

results = duckdb.sql(
    f"""
        SELECT time, TREFMXAV_R, location_ID, HW{THRESHOLD}, exceeded
                     , threshold{THRESHOLD}, Nth_day
                     , event_ID, global_event_ID
        FROM hw_def_95_df
        where location_ID = 10889
        and HW{THRESHOLD} is true
        limit 30
                     """
).df()
results

results = duckdb.sql(
    f"""
        select avg(total_days) as average_days
                     from (
                     SELECT  global_event_ID, count(1) as total_days
        FROM hw_def_95_df
        where HW{THRESHOLD} is true
        group by global_event_ID
                     )
                     """
).df()
results
# ##  Step 2.1 Check the continuity of dates within each event
import pandas as pd

# ## Step 2.3 Manually inspect a few events
import duckdb
import pandas

local_df_results = duckdb.sql(
    f"""
        SELECT time,  location_ID, HW{THRESHOLD}
                     , event_ID, global_event_ID
        FROM local_hour_adjusted_df
        where location_ID = 10889
        --and HW{THRESHOLD} is true
        limit 30
                     """
).df()
local_df_results
# Inspect a few events manually
sample_events = (
    local_hour_adjusted_df.groupby(["location_ID", "event_ID"])
    .head(1)
    .sort_values("event_ID")
)
print(sample_events[["location_ID", "event_ID", "local_time"]])
sample_events[["location_ID", "event_ID", "local_time", "UHI_diff"]].query(
    "location_ID == 35793"
)
local_hour_adjusted_df[
    ["lon", "lat", "location_ID", "event_ID", "time", "local_time", "UHI_diff"]
].query("location_ID == 10889")
local_hour_adjusted_df.head()
# # Step 3: For each urban grid, identify HWs with positive and negative UHI-HW interactions and then calculate the mean UHI_diff value. Then compare the meteorological conditions (air temperature, humidity, wind, planet boundary layer depth, etc.) between the positive UHI-HW-interaction event and negative UHI-HW-interaction event.
# ##  Step 3.1: Calculate the average event frequency (number of even per year) and average event duration  per location
import pandas as pd

# Assuming your DataFrame is named local_hour_adjusted_df
# Ensure proper datatypes especially for 'time' column
local_hour_adjusted_df["time"] = pd.to_datetime(local_hour_adjusted_df["time"])
local_hour_adjusted_df["year"] = local_hour_adjusted_df["time"].dt.year

# Group by location_ID, year, and event_ID
grouped = local_hour_adjusted_df.groupby(["location_ID", "year", "event_ID"])

# Calculate event duration in days (count rows and divide by 24)
event_duration = grouped.size().div(24).reset_index(name="Duration in Days")

# Aggregate data
# Number of distinct years per location
years_per_location = local_hour_adjusted_df.groupby("location_ID")["year"].nunique()

# Number of distinct events per location
distinct_events = local_hour_adjusted_df.groupby("location_ID")["event_ID"].nunique()

# Average event duration per location
average_duration = event_duration.groupby("location_ID")["Duration in Days"].mean()

# Normalize event count by the number of years
events_per_year = distinct_events / years_per_location

# Combine all results into a single DataFrame
hw_freq_by_location = pd.DataFrame(
    {
        "location_ID": events_per_year.index,
        "Num_events": events_per_year.values,
        "Duration_avg": average_duration.values,
    }
)
hw_freq_by_location.head()


hw_freq_by_location
hw_freq_by_location.info()
# show % of rows that has Duration_avg < 3.0
hw_freq_by_location.query("Duration_avg < 3.0").count() / hw_freq_by_location.count()
percentage = (hw_freq_by_location["Duration_avg"] < 3.0).mean() * 100
percentage

# ##  Step 3.2: define day and night time
# Daytime: 08:00 to 16:00 local time. (Keer paper)
# Nighttime: 20:00 to 04:00 local time.
import pandas as pd

# Assuming 'local_hour_adjusted_df' is your DataFrame name

# Step 1: Define masks for daytime and nighttime
daytime_mask = local_hour_adjusted_df["local_hour"].between(8, 16)
nighttime_mask = local_hour_adjusted_df["local_hour"].between(
    20, 24
) | local_hour_adjusted_df["local_hour"].between(0, 4)


# ##  Step 3.3: Calculate the mean UHI_diff value for each event day and night
# Function to compute averages for UHI_diff based on given mask and aggregate by a specified key
def compute_uhi_diff_averages(df, mask, group_key):
    return df[mask].groupby(group_key)["UHI_diff"].mean()


# Calculate averages for UHI_diff for daytime, nighttime and whole event
daytime_uhi_diff_avg = compute_uhi_diff_averages(
    local_hour_adjusted_df, daytime_mask, "location_ID"
)
nighttime_uhi_diff_avg = compute_uhi_diff_averages(
    local_hour_adjusted_df, nighttime_mask, "location_ID"
)
uhi_diff_avg = local_hour_adjusted_df.groupby("location_ID")["UHI_diff"].mean()

# Combine the results into a single DataFrame
uhi_diff_summary = pd.DataFrame(
    {
        "Daytime_UHI_diff_avg": daytime_uhi_diff_avg,
        "Nighttime_UHI_diff_avg": nighttime_uhi_diff_avg,
        "Overall_UHI_diff_avg": uhi_diff_avg,
    }
)

# Reset index if needed to make location_ID a column
uhi_diff_summary.reset_index(inplace=True)
uhi_diff_summary
mean_value = uhi_diff_summary["Overall_UHI_diff_avg"].mean()
std_dev = uhi_diff_summary["Overall_UHI_diff_avg"].std()

# Define thresholds
threshold_low = mean_value - std_dev
threshold_high = mean_value + std_dev

mean_value, std_dev, threshold_low, threshold_high

# ##  Step 3.4: Based on the distribution, Decide the threshold for positive and negative UHI_diff
# Define thresholds
threshold_low = -0.2
threshold_high = 0.2
# Classify values
uhi_diff_summary["Daytime_UHI_Category"] = pd.cut(
    uhi_diff_summary["Daytime_UHI_diff_avg"],
    bins=[-float("inf"), threshold_low, threshold_high, float("inf")],
    labels=["Negative", "Insignificant", "Positive"],
)

# View the DataFrame with the new categorization
uhi_diff_summary

# Convert the location_ID_ds to DataFrame
location_ID_df = location_ID_ds.to_dataframe().reset_index()

# Merge the DataFrames to add lon and lat cols
uhi_diff_summary = pd.merge(
    uhi_diff_summary,
    location_ID_df[["location_ID", "lon", "lat"]],
    on="location_ID",
    how="left",
)

uhi_diff_summary.info()
uhi_diff_summary

# # Step 4: Data Analysis
# ##  Step 4.1: Create a global map showing the UHI_diff values for each grid cell, color-coded by the UHI_diff category
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# Function to normalize longitude values
def normalize_longitude(lon):
    return ((lon + 180) % 360) - 180


# Function to draw the map with categories
def draw_map_subplot(data, title, output_dir=None):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    if data.empty:
        print(f"No data available for {title}. Skipping plot.")
        ax.set_title(title + " - No Data Available")
        return

    m = Basemap(
        projection="cyl",
        lon_0=0,
        ax=ax,
        fix_aspect=False,
        llcrnrlat=-44.94133,
        urcrnrlat=65.12386,
    )
    m.drawcoastlines(color="0.15", linewidth=0.5, zorder=3)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color="white", lake_color="lightcyan")
    m.drawmapboundary(fill_color="lightcyan")
    m.drawparallels(np.arange(-90.0, 91.0, 30.0), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180.0, 181.0, 60.0), labels=[0, 0, 0, 1], fontsize=10)

    normalized_lons = normalize_longitude(data["lon"].values)
    x, y = m(normalized_lons, data["lat"].values)

    # Dynamically set color map limits based on the data
    vmin, vmax = data["UHI_diff"].min(), data["UHI_diff"].max()
    sc = m.scatter(
        x,
        y,
        c=data["UHI_diff"],
        cmap="coolwarm",
        marker="o",
        edgecolor="none",
        s=10,
        alpha=0.75,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
    ax.set_title(title)

    if output_dir:
        filename = f"{title.replace(' ', '_')}.png"
        save_plot(plt, filename, output_dir)
    else:
        plt.show()


# Main function to create plots
def plot_uhi_maps(df, output_dir=None):
    categories = ["Positive", "Insignificant", "Negative"]
    variables = [
        ("Daytime_UHI_diff_avg", "Daytime UHI"),
        ("Nighttime_UHI_diff_avg", "Nighttime UHI"),
        ("Overall_UHI_diff_avg", "Overall UHI"),
    ]

    for var in variables:
        for cat in categories:
            filtered_data = df[df["Daytime_UHI_Category"] == cat]
            filtered_data = filtered_data[["location_ID", "lon", "lat", var[0]]].rename(
                columns={var[0]: "UHI_diff"}
            )
            draw_map_subplot(filtered_data, f"{var[1]} - {cat}", output_dir)


def plot_uhi_distributions(uhi_diff_summary, output_dir):
    """Create and save distribution plots for UHI differences"""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    uhi_columns = [
        "Daytime_UHI_diff_avg",
        "Nighttime_UHI_diff_avg",
        "Overall_UHI_diff_avg",
    ]
    titles = ["Daytime UHI", "Nighttime UHI", "Overall UHI"]

    for i, column in enumerate(uhi_columns):
        ax1 = axes[i]
        sns.histplot(
            uhi_diff_summary[column],
            kde=False,
            ax=ax1,
            color="blue",
            label="Number of Events",
        )
        ax1.set_title(f"{titles[i]}")
        ax1.set_xlabel("UHI Difference")
        ax1.set_ylabel("Number of Events")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        data = uhi_diff_summary[column].dropna()
        counts, bin_edges = np.histogram(data, bins=100, density=False)
        cdf = np.cumsum(counts)
        ax2.plot(bin_edges[1:], cdf, color="green", label="Cumulative Total")
        ax2.set_ylabel("Cumulative Total")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    save_plot(plt, "uhi_distributions.png", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate UHI global maps")
    parser.add_argument(
        "--output_dir",
        default="/home/jguo/tmp/output",
        help="Directory for output files",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)

    # Load and process data
    THRESHOLD = 98
    summary_dir = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary"
    merged_feather_path = os.path.join(
        summary_dir, f"local_hour_adjusted_variables_HW{THRESHOLD}.feather"
    )

    # Load data and perform analysis
    local_hour_adjusted_df = pd.read_feather(merged_feather_path)
    location_ID_path = os.path.join(summary_dir, "location_IDs.nc")
    location_ID_ds = xr.open_dataset(location_ID_path)

    # Save distribution plots
    plot_uhi_distributions(uhi_diff_summary, output_dir)

    # Generate and save map plots
    plot_uhi_maps(uhi_diff_summary, output_dir)


if __name__ == "__main__":
    main()

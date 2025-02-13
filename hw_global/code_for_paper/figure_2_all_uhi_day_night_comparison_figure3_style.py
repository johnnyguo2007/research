import pandas as pd
import numpy as np
import xarray as xr
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm


def setup_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def save_plot(plt, filename, output_dir):
    """Save plot to specified output directory"""
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def create_map(ax, data, title, vmin=None, vmax=None, cmap="coolwarm"):
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

    sc = m.scatter(
        x,
        y,
        c=data["UHI_diff"],
        cmap=cmap,
        marker="o",
        edgecolor="none",
        s=10,
        alpha=0.75,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
    ax.set_title(title)
    return sc


def normalize_longitude(lon):
    return ((lon + 180) % 360) - 180


def create_map_figure3_style(ax, data, title, vmin, vmax, cmap):
    # Calculate grid dimensions based on data
    lat_resolution = 180/192  # Degrees per grid cell for latitude
    lon_resolution = 360/288  # Degrees per grid cell for longitude
    
    # Define latitude bounds from Figure3.py
    lat_south = -56.0733
    lat_north = 84.34555
    
    # Create latitude and longitude arrays
    lats = np.arange(-90, 90 + lat_resolution, lat_resolution)
    lons = np.arange(-180, 180 + lon_resolution, lon_resolution)
    
    # Filter latitudes to exclude poles
    lat_mask = (lats >= lat_south) & (lats <= lat_north)
    filtered_lats = lats[lat_mask]
    
    m = Basemap(
        projection="cyl",
        lon_0=0,
        ax=ax,
        fix_aspect=False,
        llcrnrlat=lat_south,
        urcrnrlat=lat_north,
        llcrnrlon=lons[0],
        urcrnrlon=lons[-1],
        rsphere=6371200.0,
        resolution="l",
        area_thresh=10000,
    )
    m.drawmapboundary(fill_color="lightcyan")
    m.fillcontinents(color="white", lake_color="lightcyan")
    m.drawcoastlines(linewidth=0.3, zorder=3)

    # Draw parallels and meridians
    parallels = np.arange(-90, 91, 30)  # -90 to 90 by 30 degrees
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, linewidth=0.3)
    meridians = np.arange(-180, 181, 60)  # -180 to 180 by 60 degrees
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, linewidth=0.3)

    # Normalize longitudes to [-180, 180]
    data["lon"] = np.where(data["lon"] > 180, data["lon"] - 360, data["lon"])

    # Create grid excluding poles
    lons, lats = np.meshgrid(lons[:-1], filtered_lats)  # Exclude last point to avoid duplicate
    x, y = m(lons, lats)

    # Create land mask for filtered grid
    land_mask = np.zeros_like(lons, dtype=bool)
    for lon, lat in zip(data["lon"], data["lat"]):
        if lat_south <= lat <= lat_north:
            lon_idx = np.argmin(np.abs(lons[0, :] - lon))
            lat_idx = np.argmin(np.abs(lats[:, 0] - lat))
            land_mask[lat_idx, lon_idx] = True

    # Create data array for filtered grid
    data_to_plot = np.full_like(lons, np.nan)
    for lon, lat, uhi in zip(data["lon"], data["lat"], data["UHI_diff"]):
        if lat_south <= lat <= lat_north:
            lon_idx = np.argmin(np.abs(lons[0, :] - lon))
            lat_idx = np.argmin(np.abs(lats[:, 0] - lat))
            data_to_plot[lat_idx, lon_idx] = uhi

    data_to_plot[~land_mask] = np.nan

    cs = m.pcolormesh(
        x,
        y,
        data_to_plot,
        cmap=cmap,
        norm=BoundaryNorm(
            np.linspace(vmin, vmax, 256), ncolors=plt.get_cmap(cmap).N, clip=True
        ),
        zorder=2,
        shading="auto",
    )
    cbar = plt.colorbar(
        cs, ax=ax, orientation="vertical", pad=0.02, extend="both", format="%.1f"
    )
    cbar.set_label('(°C)', rotation=0, labelpad= -1, y=0.05, ha='right')
    ax.set_title(title, fontsize=14, fontweight="bold", pad=-1)
    return cs


def plot_all_uhi_maps_figure3_style(df, output_dir=None):
    """
    Plot all UHI values for day and night with synchronized color scales
    using the style of Figure3.py
    """
    # Prepare data for day and night
    day_data = df[["location_ID", "lon", "lat", "Daytime_UHI_diff_avg"]].rename(
        columns={"Daytime_UHI_diff_avg": "UHI_diff"}
    )
    night_data = df[["location_ID", "lon", "lat", "Nighttime_UHI_diff_avg"]].rename(
        columns={"Nighttime_UHI_diff_avg": "UHI_diff"}
    )

    # Calculate the maximum absolute value for symmetric color scaling
    combined_data = pd.concat([day_data["UHI_diff"], night_data["UHI_diff"]], ignore_index=True)
    max_abs = np.nanmax(np.abs(combined_data))

    # Calculate the 5th and 95th percentiles for more robust min/max
    percentile_vmin = np.nanpercentile(pd.concat([day_data["UHI_diff"], night_data["UHI_diff"]], ignore_index=True), 2)
    percentile_vmax = np.nanpercentile(pd.concat([day_data["UHI_diff"], night_data["UHI_diff"]], ignore_index=True), 98)

    # Set vmin and vmax to be the minimum of percentile limits and symmetric max_abs
    vmin = min(percentile_vmin, -max_abs)
    vmax = max(percentile_vmax, max_abs)  # Updated to center colorbar at 0 while respecting percentile limits
    vmin = -1.0
    vmax = 1.0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), dpi=600)
    plt.rcParams.update({"font.sans-serif": "Arial"})

    create_map_figure3_style(ax1, day_data, "Day", vmin, vmax, "coolwarm")
    create_map_figure3_style(ax2, night_data, "Night", vmin, vmax, "coolwarm")
    plt.tight_layout()

    if output_dir:
        save_plot(plt, "Figure_2_ab_all_uhi_day_night_comparison_figure3_style.png", output_dir)
    else:
        plt.show()


def plot_all_uhi_maps(df, output_dir=None):
    """Plot all UHI values for day and night with synchronized color scales"""
    # Prepare data for day and night
    day_data = df[["location_ID", "lon", "lat", "Daytime_UHI_diff_avg"]].rename(
        columns={"Daytime_UHI_diff_avg": "UHI_diff"}
    )
    night_data = df[["location_ID", "lon", "lat", "Nighttime_UHI_diff_avg"]].rename(
        columns={"Nighttime_UHI_diff_avg": "UHI_diff"}
    )

    # Find global min and max for consistent color scaling
    vmin = min(day_data["UHI_diff"].min(), night_data["UHI_diff"].min())
    vmax = max(day_data["UHI_diff"].max(), night_data["UHI_diff"].max())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=600)

    # Create both maps using the top-level create_map
    create_map(ax1, day_data, "All Daytime UHI", vmin=vmin, vmax=vmax, cmap="coolwarm")
    create_map(
        ax2, night_data, "All Nighttime UHI", vmin=vmin, vmax=vmax, cmap="coolwarm"
    )

    plt.tight_layout()

    if output_dir:
        save_plot(plt, "all_uhi_day_night_comparison.png", output_dir)
    else:
        plt.show()


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


def draw_map_subplot(data, title, output_dir=None):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
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

def plot_missing_kgmajorclass(df, output_dir):
    """Plots locations where KGMajorClass is missing on a global map."""

    # Filter for rows where KGMajorClass is missing
    missing_data = df[df['KGMajorClass'].isna()]
    
    if missing_data.empty:
        print("No missing KGMajorClass data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    m = Basemap(projection='cyl', lon_0=0, ax=ax, fix_aspect=False,
                llcrnrlat=-44.94133, urcrnrlat=65.12386)
    m.drawcoastlines(color='0.15', linewidth=0.5, zorder=3)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='white', lake_color='lightcyan')
    m.drawmapboundary(fill_color='lightcyan')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0,0,0,1], fontsize=10)

    # Normalize longitudes and plot
    normalized_lons = normalize_longitude(missing_data['lon'].values)
    x, y = m(normalized_lons, missing_data['lat'].values)
    m.scatter(x, y, color='red', marker='o', s=10, alpha=0.75, zorder=4)

    ax.set_title('Locations with Missing KGMajorClass')
    
    if output_dir:
        save_plot(plt, "missing_kgmajorclass_locations.png", output_dir)
    else:
        plt.show()

def calculate_and_save_stats(df, output_dir):
    """Calculate and save day/night UHI statistics"""
    # Calculate day/night statistics
    daytime_stats = {
        'mean': df['Daytime_UHI_diff_avg'].mean(),
        'std': df['Daytime_UHI_diff_avg'].std()
    }
    
    nighttime_stats = {
        'mean': df['Nighttime_UHI_diff_avg'].mean(),
        'std': df['Nighttime_UHI_diff_avg'].std()
    }
    
    overall_stats = {
        'mean': df['Overall_UHI_diff_avg'].mean(),
        'std': df['Overall_UHI_diff_avg'].std()
    }
    
    # Format the statistics into a string
    stats_text = (
        "UHI Difference Statistics:\n"
        "========================\n\n"
        "Daytime:\n"
        f"  Mean: {daytime_stats['mean']:.3f}°C\n"
        f"  Standard Deviation: {daytime_stats['std']:.3f}°C\n\n"
        "Nighttime:\n"
        f"  Mean: {nighttime_stats['mean']:.3f}°C\n"
        f"  Standard Deviation: {nighttime_stats['std']:.3f}°C\n\n"
        "Overall:\n"
        f"  Mean: {overall_stats['mean']:.3f}°C\n"
        f"  Standard Deviation: {overall_stats['std']:.3f}°C\n"
    )
    
    # Save to file
    output_path = os.path.join(output_dir, 'uhi_diff_statistics.txt')
    with open(output_path, 'w') as f:
        f.write(stats_text)

def main():
    parser = argparse.ArgumentParser(description="Generate UHI global maps")
    parser.add_argument(
        "--output_dir",
        default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper",
        help="Directory for output files",
    )
    args = parser.parse_args()
    import matplotlib.font_manager

    matplotlib.font_manager.findfont("Arial", rebuild_if_missing=True)

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

    # Define masks for daytime and nighttime
    day_night_avg = True
    if day_night_avg:
        daytime_mask = local_hour_adjusted_df["local_hour"].between(8, 16)
        nighttime_mask = local_hour_adjusted_df["local_hour"].between(
            20, 24
        ) | local_hour_adjusted_df["local_hour"].between(0, 4)
    else: # use 16 and 4
        daytime_mask = local_hour_adjusted_df["local_hour"].between(16, 16)
        nighttime_mask = local_hour_adjusted_df["local_hour"].between(
            4, 4
        ) 

    # Calculate averages for UHI_diff
    daytime_uhi_diff_avg = local_hour_adjusted_df[daytime_mask].groupby("location_ID")["UHI_diff"].mean()
    nighttime_uhi_diff_avg = local_hour_adjusted_df[nighttime_mask].groupby("location_ID")["UHI_diff"].mean()
    uhi_diff_avg = local_hour_adjusted_df.groupby("location_ID")["UHI_diff"].mean()

    # Combine into summary DataFrame
    uhi_diff_summary = pd.DataFrame({
        "Daytime_UHI_diff_avg": daytime_uhi_diff_avg,
        "Nighttime_UHI_diff_avg": nighttime_uhi_diff_avg,
        "Overall_UHI_diff_avg": uhi_diff_avg,
    })
    uhi_diff_summary.reset_index(inplace=True)

    # Merge with location data
    location_ID_df = location_ID_ds.to_dataframe().reset_index()
    uhi_diff_summary = pd.merge(
        uhi_diff_summary,
        location_ID_df[["location_ID", "lon", "lat"]],
        on="location_ID",
        how="left",
    )

    # Classify values
    threshold_low = -0.2
    threshold_high = 0.2
    uhi_diff_summary["Daytime_UHI_Category"] = pd.cut(
        uhi_diff_summary["Daytime_UHI_diff_avg"],
        bins=[-float("inf"), threshold_low, threshold_high, float("inf")],
        labels=["Negative", "Insignificant", "Positive"],
    )
    
    # Merge the KGMajorClass from the local_hour_adjusted_df into uhi_diff_summary
    uhi_diff_summary = pd.merge(uhi_diff_summary,
                                 local_hour_adjusted_df[['location_ID', 'KGMajorClass']].drop_duplicates(),
                                 on='location_ID',
                                 how='left')

    # Generate plots
    plot_all_uhi_maps(uhi_diff_summary, output_dir)
    plot_all_uhi_maps_figure3_style(uhi_diff_summary, output_dir)
    # plot_missing_kgmajorclass(uhi_diff_summary, output_dir)

    # After creating uhi_diff_summary DataFrame, add:
    calculate_and_save_stats(uhi_diff_summary, output_dir)


if __name__ == "__main__":
    main()
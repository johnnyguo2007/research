# %%
import pandas as pd
import os
import sys

sys.path.append('/home/jguo/research/hw_global/ultimate/')
# Import get_latex_label from plot_util
from mlflow_tools.plot_util import replace_cold_with_continental

# Load your data
# Path to your Feather file
feather_path = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather"

if not os.path.exists(feather_path):
    print(f"Error: Feather file not found at {feather_path}")
    sys.exit(1)
    
# Read the Feather file into a DataFrame
df = pd.read_feather(feather_path)

# 1. Calculate Daily Average
df['date'] = df['time'].dt.date  # Extract date

# 2. Sort so day_in_event is computed in the correct chronological order
df = df.sort_values(['location_ID', 'event_ID', 'time'])

# 3. Compute "day_in_event"
df['day_in_event'] = df.groupby(['location_ID', 'event_ID'])['date']\
                    .transform(lambda x: (pd.to_datetime(x) - pd.to_datetime(x.min())).dt.days)

# 4. Calculate event duration for each heatwave event
event_durations = df.groupby(['location_ID', 'event_ID', 'KGMajorClass'])['day_in_event'].max() + 1
event_durations_df = event_durations.reset_index()
event_durations_df = event_durations_df.rename(columns={'day_in_event': 'event_duration'})

# 5. Calculate average event duration globally and by climate zone
global_avg_duration = event_durations_df['event_duration'].mean()
global_std_duration = event_durations_df['event_duration'].std()

print("--- Global Heatwave Statistics ---")
print(f"Average Duration: {global_avg_duration:.2f} ± {global_std_duration:.2f} days")
print(f"Total Events: {len(event_durations_df)}")
print(f"Median Duration: {event_durations_df['event_duration'].median():.2f} days")
print(f"Duration Range: {event_durations_df['event_duration'].min()} - {event_durations_df['event_duration'].max()} days")


# 6. Calculate by climate zone
zone_stats = event_durations_df.groupby('KGMajorClass')['event_duration'].agg(['mean', 'std', 'count']).round(2)
zone_stats = zone_stats.rename(columns={'mean': 'avg_duration', 'std': 'std_duration', 'count': 'event_count'})

# Remove Polar zone and replace Cold with Continental
zone_stats = zone_stats[zone_stats.index != 'Polar'].copy()
zone_stats.index = zone_stats.index.map(lambda x: replace_cold_with_continental(x))

print("\n--- Heatwave Statistics by Climate Zone ---")
print(zone_stats)

# Save statistics to CSV
output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper'
os.makedirs(output_dir, exist_ok=True)
stats_output_path = os.path.join(output_dir, 'avg_event_length_stats.csv')
zone_stats.to_csv(stats_output_path)
print(f"\nStatistics saved to {stats_output_path}") 
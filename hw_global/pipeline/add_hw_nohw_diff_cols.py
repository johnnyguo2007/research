import pandas as pd
import os
import argparse

THRESHOLD = "98"

# Parse arguments
parser = argparse.ArgumentParser(description="Add hw_nohw_diff cols.")
parser.add_argument("--summary_dir", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary",
                    help="Directory for saving summary files and artifacts.")
parser.add_argument("--merged_feather_file", type=str, default=f"local_hour_adjusted_variables_HW{THRESHOLD}.feather",
                    help="File name of the merged feather file containing the dataset.")
parser.add_argument("--nohw_file", type=str, default=f"no_HW{THRESHOLD}_group_by_location_ID_year_local_hour.feather",
                    help="File name of the merged feather file containing the dataset.")
args = parser.parse_args()

summary_dir = args.summary_dir

# Load data
merged_feather_path = os.path.join(summary_dir, args.merged_feather_file)
print(f"Loading data from {merged_feather_path}")
local_hour_adjusted_df = pd.read_feather(merged_feather_path)
print(f"Loaded dataframe with shape: {local_hour_adjusted_df.shape}")

# Load no_HW98_group_by_location_ID_year_local_hour.feather data
no_HW_year_hourly_path = os.path.join(summary_dir, args.nohw_file)
print(f"Loading no_HW_year_hourly data from {no_HW_year_hourly_path}")
no_HW_year_hourly = pd.read_feather(no_HW_year_hourly_path)

# Load df_daily_vars and get HW-NoHW diff variables
df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')
hw_nohw_diff_vars = df_daily_vars.loc[df_daily_vars['HW_NOHW_Diff'] == 'Y', 'Variable'].tolist()

# Subset no_HW_year_hourly to include only the necessary columns
no_HW_year_hourly_subset = no_HW_year_hourly[['location_ID', 'year', 'hour'] + hw_nohw_diff_vars]

# Merge local_hour_adjusted_df with no_HW_year_hourly_subset using left join
local_hour_adjusted_df = local_hour_adjusted_df.merge(
    no_HW_year_hourly_subset, 
    on=['location_ID', 'year', 'hour'], 
    how='left',
    suffixes=('', '_nohw')
)

# Calculate HW-NoHW diff variables and add them to local_hour_adjusted_df
for var in hw_nohw_diff_vars:
    diff_var = f"hw_nohw_diff_{var}"
    local_hour_adjusted_df[diff_var] = local_hour_adjusted_df[var] - local_hour_adjusted_df[f"{var}_nohw"]
    # Drop the temporary _nohw column
    local_hour_adjusted_df.drop(f"{var}_nohw", axis=1, inplace=True)

# Save the updated dataframe
output_file = os.path.join(summary_dir, 'updated_' + args.merged_feather_file)
local_hour_adjusted_df.to_feather(output_file)
print(f"Updated dataframe saved to {output_file}")

print("Script completed successfully.")


# # ... (rest of the code remains the same) ...

# # Load feature list
# print("Loading feature list...")
# daily_vars = df_daily_vars.loc[df_daily_vars[args.feature_column] == 'Y', 'Variable']
# daily_var_lst = daily_vars.tolist()

# # Add HW-NoHW diff variables to daily_var_lst
# daily_var_lst.extend([f"hw_nohw_diff_{var}" for var in hw_nohw_diff_vars])

# ... (rest of the code remains the same) ...
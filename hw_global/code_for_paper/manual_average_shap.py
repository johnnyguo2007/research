import pandas as pd
import pyarrow.feather as feather
import os

# Define the input and output paths
input_file = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts/284640530508641411/810c2219de0646da91abca00b6e1a6d7/artifacts/shap_values_with_additional_columns.feather"
output_dir = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/hourly_stacked_bar"

# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# Read the feather file into a pandas DataFrame
df = pd.read_feather(input_file)

# Filter out rows where 'KGMajorClass' is 'Polar'
df = df[df['KGMajorClass'] != 'Polar']

# Identify columns ending with '_shap'
shap_cols = [col for col in df.columns if col.endswith('_shap')]

# Group by 'KGMajorClass' and 'local_hour' and calculate the mean of shap columns
grouped_df = df.groupby(['KGMajorClass', 'local_hour'])[shap_cols].mean().reset_index()

# Calculate mean base_value grouped by 'KGMajorClass' and 'local_hour'
base_value_by_hour = df.groupby(['KGMajorClass', 'local_hour'])['base_value'].mean().reset_index()

# Calculate mean base_value grouped by 'KGMajorClass'
base_value_by_kg = df.groupby('KGMajorClass')['base_value'].mean().reset_index()

# Merge the base_value_by_hour with grouped_df
merged_df = pd.merge(grouped_df, base_value_by_hour, on=['KGMajorClass', 'local_hour'], suffixes=('', '_mean_base_value'))

# Subtract KG level base_value from KG, local_hour level base_value
base_value_diff = pd.merge(base_value_by_hour, base_value_by_kg, on='KGMajorClass', suffixes=('_hour', '_kg'))
base_value_diff['base_value_diff'] = base_value_diff['base_value_hour'] - base_value_diff['base_value_kg']

# Distribute the difference proportionally into all shap values
for col in shap_cols:
    base_value_diff[col] = base_value_diff['base_value_diff'] * (grouped_df[col] / grouped_df[shap_cols].sum(axis=1))

# Define output file paths
output_feather = os.path.join(output_dir, "shap_values_by_kg_hour.feather")
output_csv = os.path.join(output_dir, "shap_values_by_kg_hour.csv")

# Write the grouped DataFrame to a feather file
feather.write_feather(grouped_df, output_feather)

# Write the merged DataFrame to a CSV file
merged_df.to_csv(output_csv, index=False)

# Write the base_value_by_kg DataFrame to a CSV file
base_value_by_kg.to_csv(os.path.join(output_dir, "base_value_by_kg.csv"), index=False)

# Keep only 'local_hour', shap columns, and 'base_value_kg' renamed to 'base_value'
adjusted_cols = ['local_hour'] + shap_cols + ['base_value']
base_value_diff['base_value'] = base_value_diff['base_value_kg']
adjusted_df = base_value_diff[adjusted_cols]

# Write one file for each KGMajorClass
for kg_class in base_value_diff['KGMajorClass'].unique():
    kg_df = adjusted_df[base_value_diff['KGMajorClass'] == kg_class]
    kg_output_csv = os.path.join(output_dir, f"adjusted_shap_values_{kg_class}.csv")
    kg_output_feather = os.path.join(output_dir, f"adjusted_shap_values_{kg_class}.feather")
    kg_df.to_csv(kg_output_csv, index=False)
    feather.write_feather(kg_df, kg_output_feather)

print(f"Successfully wrote adjusted shap values for each KGMajorClass to separate files.")
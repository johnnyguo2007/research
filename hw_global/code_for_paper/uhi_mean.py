import pandas as pd
import pyarrow.feather as feather

# Define the input file path (same as before)
input_file = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts/284640530508641411/810c2219de0646da91abca00b6e1a6d7/artifacts/shap_values_with_additional_columns.feather"

# Read the feather file into a pandas DataFrame
df = pd.read_feather(input_file)

# Group by 'KGMajorClass' and calculate the mean of 'UHI_diff'
uhi_diff_means = df.groupby('KGMajorClass')['UHI_diff'].mean()

# Print the results
print("Mean UHI_diff by KGMajorClass:")
print(uhi_diff_means)
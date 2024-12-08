import pandas as pd

def combine_feather_files():
    # Define file paths
    day_file = (
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/"
        "research_results/summary/mlflow/mlartifacts/143560831933597437/"
        "a557890b9462417da6a603eca02dafe8/artifacts/day_shap_values_with_additional_columns.feather"
    )
    
    night_file = (
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/"
        "research_results/summary/mlflow/mlartifacts/143560831933597437/"
        "a557890b9462417da6a603eca02dafe8/artifacts/night_shap_values_with_additional_columns.feather"
    )
    
    output_file = (
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/"
        "research_results/summary/mlflow/mlartifacts/143560831933597437/"
        "a557890b9462417da6a603eca02dafe8/artifacts/shap_values_with_additional_columns.feather"
    )
    
    # Read the day and night Feather files
    try:
        df_day = pd.read_feather(day_file)
        print(f"Successfully read {day_file}")
    except Exception as e:
        print(f"Error reading {day_file}: {e}")
        return

    try:
        df_night = pd.read_feather(night_file)
        print(f"Successfully read {night_file}")
    except Exception as e:
        print(f"Error reading {night_file}: {e}")
        return

    # Concatenate the DataFrames
    combined_df = pd.concat([df_day, df_night], ignore_index=True)
    print("Successfully concatenated day and night DataFrames.")

    # Write the combined DataFrame to the output Feather file
    try:
        combined_df.to_feather(output_file)
        print(f"Successfully wrote combined DataFrame to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    combine_feather_files()

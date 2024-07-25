import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_long_name(var_name, df_daily_vars):
    if df_daily_vars is None:
        logging.warning("df_daily_vars is None. Returning original var_name.")
        return var_name
    
    if var_name.startswith('delta_'):
        original_var_name = var_name.replace('delta_', '')
        original_long_name = df_daily_vars.loc[df_daily_vars['Variable'] == original_var_name, 'Long Name'].values
        if original_long_name.size > 0:
            return f"Difference of {original_long_name[0]}"
        else:
            return f"Difference of {original_var_name} (No long name found)"
    else:
        long_name = df_daily_vars.loc[df_daily_vars['Variable'] == var_name, 'Long Name'].values
        if long_name.size > 0:
            return long_name[0]
        else:
            return f"{var_name} (No long name found)"

def add_long_name(input_df, join_column='Feature', df_daily_vars=None):
    input_df['Long Name'] = input_df[join_column].apply(lambda x: get_long_name(x, df_daily_vars))
    return input_df

def get_shap_feature_importance(shap_values, feature_names, df_daily_vars):
    shap_feature_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_feature_importance})
    shap_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    shap_importance_df = add_long_name(shap_importance_df, join_column='Feature', df_daily_vars=df_daily_vars)
    return shap_importance_df

def main(experiment_name):
    # Set up MLflow
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")

    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Get the run with the latest timestamp within the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
    if len(runs) == 0:
        logging.error(f"No runs found in experiment '{experiment_name}'. Please check the experiment name and make sure it contains runs.")
        return

    run = runs.iloc[0]
    run_id = run.run_id

    # Retrieve the artifact directory from the MLflow server
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    artifact_uri = artifact_uri.replace("mlflow-artifacts:", "/home/jguo/research/hw_global/mlartifacts")

    # Load df_daily_vars
    df_daily_vars_path = os.path.join(artifact_uri, "hourlyDataSchema.xlsx")
    if os.path.exists(df_daily_vars_path):
        df_daily_vars = pd.read_excel(df_daily_vars_path)
        logging.info(f"Loaded df_daily_vars from {df_daily_vars_path}")
    else:
        logging.warning(f"df_daily_vars file not found at {df_daily_vars_path}. Continuing without it.")
        df_daily_vars = None

    # Determine whether it's daytime or nighttime based on the experiment name
    if 'day' in experiment_name.lower():
        time_period = 'day'
    elif 'night' in experiment_name.lower():
        time_period = 'night'
    else:
        raise ValueError("Experiment name should contain either 'day' or 'night' to determine the time period.")

    # Load data from MLflow
    shap_values_path = os.path.join(artifact_uri, f"shap_values.npy")
    if os.path.exists(shap_values_path):
        shap_values = np.load(shap_values_path)
        logging.info(f"Loaded shap_values from {shap_values_path}")
    else:
        logging.error(f"shap_values file not found at {shap_values_path}")
        return

    feature_names_path = os.path.join(artifact_uri, f"feature_names.txt")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f]
        logging.info(f"Loaded feature_names from {feature_names_path}")
    else:
        logging.error(f"feature_names file not found at {feature_names_path}")
        return

    X_path = os.path.join(artifact_uri, f"X_data.feather")
    if os.path.exists(X_path):
        X = pd.read_feather(X_path)
        logging.info(f"Loaded X data from {X_path}")
    else:
        logging.warning(f"X_data file not found at {X_path}. Continuing without it.")
        X = None

    shap_feature_importance = get_shap_feature_importance(shap_values, feature_names, df_daily_vars)

    # SHAP feature importance waterfall plot
    logging.info(f"Creating SHAP feature importance waterfall plot for {time_period}time...")

    # Normalize the SHAP feature importances
    shap_feature_importance['Normalized Importance'] = shap_feature_importance['Importance'] / shap_feature_importance['Importance'].sum()

    # Format the percentages for display  
    shap_feature_importance['Percentage'] = shap_feature_importance['Normalized Importance'].apply(lambda x: f'{x:.2%}')

    # Sort the features by importance in descending order
    shap_feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

    # Create the waterfall plot with all features
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_feature_importance['Importance'],
            base_values=0,
            feature_names=shap_feature_importance['Long Name'].tolist() + ['Percentage contribution']  
        ),
        show=False,
        max_display=len(shap_feature_importance)  # Display all features
    )

    # Customize the plot
    plt.gcf().set_size_inches(15, 0.5 * len(shap_feature_importance))  # Adjust figure height based on number of features 
    plt.gcf().subplots_adjust(left=0.4)  # Increase left margin to make room for y-axis labels

    # Add percentage contribution text next to each bar
    for i, p in enumerate(plt.gca().patches):
        plt.text(p.get_width() * 1.01, p.get_y() + p.get_height() / 2,
                 shap_feature_importance['Percentage'][i],
                 ha='left', va='center') 

    output_path = os.path.join(artifact_uri, f"post_process_{time_period}_shap_feature_importance_waterfall_plot.png")
    plt.savefig(output_path)
    logging.info(f"SHAP feature importance waterfall plot for {time_period}time saved to {output_path}")
    plt.clf()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate and save SHAP feature importance waterfall plot based on experiment name.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the MLflow experiment.")

    args = parser.parse_args()

    main(args.experiment_name)
    
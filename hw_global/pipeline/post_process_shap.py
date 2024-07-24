import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
from scipy.stats import linregress
import argparse

def feature_linear_slope(df, feature_name, label, confidence_level=0.95):
    try:
        slope, _, _, p_value, stderr = linregress(df[feature_name], df[label])
        t_value = np.abs(np.percentile(np.random.standard_t(df[feature_name].shape[0] - 2, 100000),
                                       [(1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2]))[1]
        margin_of_error = t_value * stderr
        return slope, margin_of_error, p_value
    except Exception as e:
        print(f"Error calculating slope for {feature_name}: {e}")
        return None, None, None

def get_long_name(var_name, df_daily_vars):
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

# Parse arguments
parser = argparse.ArgumentParser(description="Generate SHAP feature importance waterfall plots for latest run in specified experiment.")
parser.add_argument("--experiment_name", type=str, required=True, help="Name of the MLflow experiment.")

args = parser.parse_args()

# Set up MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Get the experiment by name
experiment = mlflow.get_experiment_by_name(args.experiment_name)
experiment_id = experiment.experiment_id

# Get the run with the latest timestamp within the experiment
run = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)[0]
run_id = run.info.run_id

# Load df_daily_vars
df_daily_vars_path = f"mlruns/{experiment_id}/{run_id}/artifacts/hourlyDataSchema.xlsx"
df_daily_vars = pd.read_excel(df_daily_vars_path)

# Get feature importance using SHAP values
def get_shap_feature_importance(shap_values, feature_names):
    shap_feature_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_feature_importance})
    shap_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    shap_importance_df = add_long_name(shap_importance_df, join_column='Feature')
    return shap_importance_df

for time_period in ['day', 'night']:
    # Load data from MLflow
    shap_values_path = f"mlruns/{experiment_id}/{run_id}/artifacts/{time_period}time_shap_values.npy"
    shap_values = np.load(shap_values_path)

    feature_names_path = f"mlruns/{experiment_id}/{run_id}/artifacts/{time_period}time_feature_names.txt"
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]

    X_path = f"mlruns/{experiment_id}/{run_id}/artifacts/{time_period}time_X_data.feather"
    X = pd.read_feather(X_path)

    shap_feature_importance = get_shap_feature_importance(shap_values, feature_names)

    # SHAP feature importance waterfall plot
    print(f"Creating SHAP feature importance waterfall plot for {time_period}time...")

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

    output_path = f"post_process_{time_period}_shap_feature_importance_waterfall_plot.png"
    plt.savefig(output_path)
    print(f"SHAP feature importance waterfall plot for {time_period}time saved to {output_path}")
    plt.clf()
    
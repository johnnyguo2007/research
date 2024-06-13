import pandas as pd
import numpy as np
import xarray as xr
import os
import netCDF4
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.catboost
import mlflow.shap
from scipy.stats import linregress
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Run UHI model for day or night data.")
parser.add_argument("--time_period", choices=["day", "night"], required=True, help="Specify whether to run for day or night data.")
args = parser.parse_args()

# Set summary directory and experiment name
summary_dir = '/Trex/test_case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
experiment_name = f'Production_UHI_{args.time_period.capitalize()}_add_delta_FSA'

# Create the MLflow experiment
mlflow.set_experiment(experiment_name)
mlflow.start_run()

# Create directory for saving figures and artifacts
figure_dir = os.path.join(summary_dir, 'mlflow', experiment_name)
os.makedirs(figure_dir, exist_ok=True)

# Load data
merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables_with_location_ID_event_ID_and_sur.feather')
local_hour_adjusted_df = pd.read_feather(merged_feather_path)

# # Filter data to have year 1985 only
# local_hour_adjusted_df = local_hour_adjusted_df[local_hour_adjusted_df['year'] == 1985]


# Load location ID dataset
location_ID_path = os.path.join(summary_dir, 'location_IDs.nc')
location_ID_ds = xr.open_dataset(location_ID_path, engine='netcdf4')

# Load feature list
df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')
daily_vars = df_daily_vars.loc[df_daily_vars['X_vars2'] == 'Y', 'Variable']
daily_var_lst = daily_vars.tolist()

# Load delta feature list
delta_vars = df_daily_vars.loc[df_daily_vars['X_vars_delta'] == 'Y', 'Variable']
delta_var_lst = delta_vars.tolist()

# Calculate delta variables and add to dataframe
for var in delta_var_lst:
    var_U = f"{var}_U"
    var_R = f"{var}_R"
    delta_var = f"delta_{var}"
    if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
        local_hour_adjusted_df[delta_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
        daily_var_lst.append(delta_var)  # Add delta variable to daily_var_lst
    else:
        print(f"Warning: {var_U} or {var_R} not found in dataframe columns.")

print("daily_var_lst", daily_var_lst)

# Save df_daily_vars to Excel file and log as artifact
df_daily_vars_path = os.path.join(figure_dir, 'df_daily_vars.xlsx')
df_daily_vars.to_excel(df_daily_vars_path, index=False)
mlflow.log_artifact(df_daily_vars_path)

# Save daily_var_lst to text file and log as artifact
daily_var_lst_path = os.path.join(figure_dir, 'daily_var_lst.txt')
with open(daily_var_lst_path, 'w') as f:
    for var in daily_var_lst:
        f.write(f"{var}\n")
mlflow.log_artifact(daily_var_lst_path)

# Define helper functions
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

def add_long_name(input_df, join_column='Feature', df_daily_vars=df_daily_vars):
    input_df['Long Name'] = input_df[join_column].apply(lambda x: get_long_name(x, df_daily_vars))
    return input_df

# Define day and night masks
daytime_mask = local_hour_adjusted_df['local_hour'].between(8, 16)
nighttime_mask = (local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 4))

# Separate daytime and nighttime data
if args.time_period == "day":
    uhi_diff = local_hour_adjusted_df[daytime_mask]
else:
    uhi_diff = local_hour_adjusted_df[nighttime_mask]

X = uhi_diff[daily_var_lst]
y = uhi_diff['UHI_diff']

# Define linear slope function
def feature_linear_slope(df, feature_name, label, confidence_level=0.95):
    slope, _, _, p_value, stderr = linregress(df[feature_name], df[label])
    t_value = np.abs(np.percentile(np.random.standard_t(df[feature_name].shape[0] - 2, 100000), [(1-confidence_level)/2, 1-(1-confidence_level)/2]))[1]
    margin_of_error = t_value * stderr
    return slope, margin_of_error, p_value

def combine_slopes(daytime_df, nighttime_df, features, labels=['UHI', 'UHI_diff'], confidence_level=0.95):
    data = {}
    for feature in features:
        slopes_with_intervals = []
        for df, time in [(daytime_df, 'Day'), (nighttime_df, 'Night')]:
            for label in labels:
                slope, margin_of_error, p_value = feature_linear_slope(df, feature, label, confidence_level)
                slope_with_interval = f"{slope:.6f} (± {margin_of_error:.6f}, P: {p_value:.6f})"
                slopes_with_intervals.append(slope_with_interval)
        data[feature] = slopes_with_intervals
    columns = [f'{time}_{label}_slope' for time in ['Day', 'Night'] for label in labels]
    results_df = pd.DataFrame(data, index=columns).transpose()
    return results_df

feature_names = daily_var_lst
results_df = combine_slopes(local_hour_adjusted_df[daytime_mask], local_hour_adjusted_df[nighttime_mask], feature_names)
sorted_results_df = results_df.sort_values('Day_UHI_slope', ascending=False)

# Save and log the sorted results DataFrame
sorted_results_path = os.path.join(figure_dir, 'sorted_results_df.csv')
sorted_results_df.to_csv(sorted_results_path)
mlflow.log_artifact(sorted_results_path)

# Train and evaluate models
def train_and_evaluate(time_uhi_diff, daily_var_lst, model_name):
    X = time_uhi_diff[daily_var_lst]
    y = time_uhi_diff['UHI_diff']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    train_pool = Pool(X_train, y_train)
    validation_pool = Pool(X_val, y_val)
    model = CatBoostRegressor(
        iterations=100000,
        learning_rate=0.03,
        depth=10,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        task_type='GPU',
        early_stopping_rounds=100,
        verbose=False
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50, verbose=False)
    
    # Log model parameters
    mlflow.log_param(f"{model_name}_iterations", model.get_param('iterations'))
    mlflow.log_param(f"{model_name}_learning_rate", model.get_param('learning_rate'))
    mlflow.log_param(f"{model_name}_depth", model.get_param('depth'))
    
    # Calculate and log metrics
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    
    mlflow.log_metric(f"{model_name}_train_rmse", train_rmse)
    mlflow.log_metric(f"{model_name}_val_rmse", val_rmse)
    mlflow.log_metric(f"{model_name}_train_mae", train_mae)
    mlflow.log_metric(f"{model_name}_val_mae", val_mae)
    mlflow.log_metric(f"{model_name}_train_r2", train_r2)
    mlflow.log_metric(f"{model_name}_val_r2", val_r2)
    
    return model

model = train_and_evaluate(uhi_diff, daily_var_lst=daily_var_lst, model_name=f"{args.time_period}_model")

# Log model
mlflow.catboost.log_model(model, f"{args.time_period}_model")


# Get feature importance
def get_ordered_feature_importance(model: CatBoostRegressor, pool, type='FeatureImportance'):
    if type == 'FeatureImportance':
        feature_importances = model.get_feature_importance()
    else:
        feature_importances = model.get_feature_importance(pool, type=type)
    
    feature_names = pool.get_feature_names()
    print(f"Length of feature_importances: {len(feature_importances)}")
    print(f"Length of feature_names: {len(feature_names)}")
    
    # Ensure the lengths match
    if len(feature_importances) != len(feature_names):
        raise ValueError("Feature importances and feature names lengths do not match")
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance_df = add_long_name(feature_importance_df, join_column='Feature')
    return feature_importance_df

full_pool = Pool(X, y)

# Feature importance plots
feature_importance = get_ordered_feature_importance(model, full_pool)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title(f'{args.time_period.capitalize()}time Feature Importance')
mlflow.log_figure(plt.gcf(), f'{args.time_period}time_feature_importance.png')
plt.clf()

# SHAP summary plots
shap_values = model.get_feature_importance(full_pool, type='ShapValues')[:,:-1]
shap.summary_plot(shap_values, X, show=False)
plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_summary_plot.png')
plt.clf()

# SHAP waterfall plots
feature_importances = model.get_feature_importance()
expected_value = shap_values[0, -1]
long_names = [get_long_name(f, df_daily_vars) for f in full_pool.get_feature_names()]
shap.waterfall_plot(shap.Explanation(feature_importances, base_values=expected_value, feature_names=long_names), show=False)
plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
plt.gcf().subplots_adjust(left=0.3)  # Increase left margin to make room for y-axis labels
mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_waterfall_plot.png')
plt.clf()

# SHAP dependence plots
def plot_dependence_grid(shap_values, X, feature_names, time_period, target_feature='U10', plots_per_row=2):
    feature_names = [f for f in feature_names if f != target_feature]
    num_features = len(feature_names)
    num_rows = (num_features + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(30, 10 * num_rows))
    axes = axes.flatten()

    for i, feature_name in enumerate(feature_names):
        shap.dependence_plot(ind=target_feature, shap_values=shap_values, features=X, interaction_index=feature_name, ax=axes[i],  show=False)
        axes[i].set_title(f"{time_period.capitalize()} time {target_feature} vs {feature_name}")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.set_size_inches(30, 10 * num_rows)
    mlflow.log_figure(plt.gcf(), f'{time_period}_dependence_plot_{target_feature}.png')
    plt.clf()

top_features = feature_importance['Feature'].head(3).tolist()

# Dependence plots
for feature in top_features:
    plot_dependence_grid(shap_values, X, feature_names=full_pool.get_feature_names(), time_period=args.time_period, target_feature=feature, plots_per_row=2)


mlflow.end_run()

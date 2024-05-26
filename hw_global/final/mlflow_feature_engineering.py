import pandas as pd
import numpy as np
import xarray as xr
import os
import netCDF4
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.catboost
import mlflow.shap
from scipy.stats import linregress

# Set summary directory and experiment name
summary_dir = '/Trex/test_case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
experiment_name = 'UHI_Day_Night_1y_1985_catboost'

# Create the MLflow experiment
mlflow.set_experiment(experiment_name)
mlflow.start_run()

# Create directory for saving figures and artifacts
figure_dir = os.path.join(summary_dir, 'mlflow', experiment_name)
os.makedirs(figure_dir, exist_ok=True)

# Load data
merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables_with_location_ID_event_ID.feather')
local_hour_adjusted_df = pd.read_feather(merged_feather_path)

# Filter data to have year 1985 only
local_hour_adjusted_df = local_hour_adjusted_df[local_hour_adjusted_df['year'] == 1985]

# Load location ID dataset
location_ID_path = os.path.join(summary_dir, 'location_IDs.nc')
location_ID_ds = xr.open_dataset(location_ID_path, engine='netcdf4')

# Load feature list
df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')
daily_vars = df_daily_vars.loc[df_daily_vars['X_vars2'] == 'Y', 'Variable']
daily_var_lst = daily_vars.tolist()

# Define helper functions
def get_long_names(variables, df):
    formatted_names = []
    for var in variables:
        long_name = df.loc[df['Variable'] == var, 'Long Name'].values
        if long_name.size > 0:
            formatted_names.append(f"{var} ({long_name[0]})")
        else:
            formatted_names.append(f"{var} (No long name found)")
    return formatted_names

def add_long_name(input_df, join_column='Feature', df_daily_vars=df_daily_vars):
    merged_df = pd.merge(input_df, df_daily_vars[['Variable', 'Long Name']], left_on=join_column, right_on='Variable', how='left')
    merged_df.drop(columns=['Variable'], inplace=True)
    return merged_df

# Define day and night masks
daytime_mask = local_hour_adjusted_df['local_hour'].between(8, 16)
nighttime_mask = (local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 4))

# Separate daytime and nighttime data
daytime_uhi_diff = local_hour_adjusted_df[daytime_mask]
X_day = daytime_uhi_diff[daily_var_lst]
y_day = daytime_uhi_diff['UHI_diff']

nighttime_uhi_diff = local_hour_adjusted_df[nighttime_mask]
X_night = nighttime_uhi_diff[daily_var_lst]
y_night = nighttime_uhi_diff['UHI_diff']

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
results_df = combine_slopes(daytime_uhi_diff, nighttime_uhi_diff, feature_names)
sorted_results_df = results_df.sort_values('Day_UHI_slope', ascending=False)

# Save and log the sorted results DataFrame
sorted_results_path = os.path.join(figure_dir, 'sorted_results_df.csv')
sorted_results_df.to_csv(sorted_results_path)
mlflow.log_artifact(sorted_results_path)

# Train and evaluate models
def train_and_evaluate(time_uhi_diff, df_daily_vars, model_name):
    daily_vars = df_daily_vars.loc[df_daily_vars['X_vars2'] == 'Y', 'Variable']
    daily_var_lst = daily_vars.tolist()
    X = time_uhi_diff[daily_var_lst]
    y = time_uhi_diff['UHI_diff']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_pool = Pool(X_train, y_train)
    validation_pool = Pool(X_val, y_val)
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
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
    mlflow.log_metric(f"{model_name}_train_rmse", train_rmse)
    mlflow.log_metric(f"{model_name}_val_rmse", val_rmse)
    
    return model

night_model = train_and_evaluate(nighttime_uhi_diff, df_daily_vars=df_daily_vars, model_name="night_model")
day_model = train_and_evaluate(daytime_uhi_diff, df_daily_vars=df_daily_vars, model_name="day_model")

# Log models
mlflow.catboost.log_model(day_model, "day_model")
mlflow.catboost.log_model(night_model, "night_model")

# Get feature importance
def get_ordered_feature_importance(model: CatBoostRegressor, pool, type='FeatureImportance'):
    if type == 'FeatureImportance':
        feature_importances = model.get_feature_importance()
    else:
        feature_importances = model.get_feature_importance(pool, type=type)
    feature_importance_df = pd.DataFrame({'Feature': pool.get_feature_names(), 'Importance': feature_importances})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance_df = add_long_name(feature_importance_df, join_column='Feature')
    return feature_importance_df

day_full_pool = Pool(X_day, y_day)
night_full_pool = Pool(X_night, y_night)

# Feature importance plots
day_feature_importance = get_ordered_feature_importance(day_model, day_full_pool)
night_feature_importance = get_ordered_feature_importance(night_model, night_full_pool)

plt.figure(figsize=(10, 6))
plt.barh(day_feature_importance['Feature'], day_feature_importance['Importance'])
plt.title('Daytime Feature Importance')
mlflow.log_figure(plt.gcf(), 'daytime_feature_importance.png')
plt.clf()

plt.figure(figsize=(10, 6))
plt.barh(night_feature_importance['Feature'], night_feature_importance['Importance'])
plt.title('Nighttime Feature Importance')
mlflow.log_figure(plt.gcf(), 'nighttime_feature_importance.png')
plt.clf()

# SHAP summary plots
day_shap_values = day_model.get_feature_importance(day_full_pool, type='ShapValues')[:,:-1]
shap.summary_plot(day_shap_values, X_day, show=False)
mlflow.log_figure(plt.gcf(), 'day_shap_summary_plot.png')
plt.clf()

night_shap_values = night_model.get_feature_importance(night_full_pool, type='ShapValues')[:,:-1]
shap.summary_plot(night_shap_values, X_night, show=False)
mlflow.log_figure(plt.gcf(), 'night_shap_summary_plot.png')
plt.clf()

# SHAP waterfall plots
day_feature_importances = day_model.get_feature_importance()
expected_value = day_shap_values[0, -1]
long_names = get_long_names(day_full_pool.get_feature_names(), df_daily_vars)
shap.waterfall_plot(shap.Explanation(day_feature_importances, base_values=expected_value, feature_names=long_names), show=False)
mlflow.log_figure(plt.gcf(), 'day_shap_waterfall_plot.png')
plt.clf()

night_feature_importances = night_model.get_feature_importance()
expected_value_night = night_shap_values[0, -1]
long_names_night = get_long_names(night_full_pool.get_feature_names(), df_daily_vars)
shap.waterfall_plot(shap.Explanation(night_feature_importances, base_values=expected_value_night, feature_names=long_names_night), show=False)
mlflow.log_figure(plt.gcf(), 'night_shap_waterfall_plot.png')
plt.clf()

# SHAP dependence plots
def plot_dependence_grid(shap_values, X, feature_names, plots_per_row=3):
    num_features = len(feature_names)
    num_rows = (num_features + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(30, 10 * num_rows))
    axes = axes.flatten()
    for i, feature_name in enumerate(feature_names):
        shap.dependence_plot(ind=feature_name, shap_values=shap_values, features=X, ax=axes[i], show=False)
        axes[i].set_title(f"Dependence plot for {feature_name}")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    return fig

top_day_features = day_feature_importance['Feature'].head(3).tolist()
top_night_features = night_feature_importance['Feature'].head(3).tolist()

# Daytime dependence plots
for feature in top_day_features:
    fig = plot_dependence_grid(day_shap_values, X_day, [feature])
    mlflow.log_figure(fig, f'day_dependence_plot_{feature}.png')
    plt.clf()

# Nighttime dependence plots
for feature in top_night_features:
    fig = plot_dependence_grid(night_shap_values, X_night, [feature])
    mlflow.log_figure(fig, f'night_dependence_plot_{feature}.png')
    plt.clf()

mlflow.end_run()

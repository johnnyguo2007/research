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
import sys

"""
This script runs a UHI (Urban Heat Island) model for day or night data using CatBoost.
It includes data loading, preprocessing, model training, evaluation, and SHAP analysis.
The script uses MLflow for experiment tracking and logging.

Usage:
python script_name.py --time_period [day/night] --summary_dir [path] --merged_feather_file [STR] 
                      [--iterations INT] [--learning_rate FLOAT] [--depth INT] 
                      [--filters STR] [--run_type STR] [--exp_name_extra STR] 
                      [--shap_calculation] [--feature_column STR] [--include_delta]
"""

# Define filter functions
def filter_by_year(df, year):
    """Filter dataframe by specific year."""
    return df[df['year'] == int(year)]

def filter_by_temperature_above_300(df, temperature):
    """Filter dataframe by temperature above a threshold."""
    return df[df['temperature'] > float(temperature)]

def filter_by_hw_count(df, threshold):
    """Filter dataframe by heatwave count below a threshold."""
    threshold = int(threshold)
    hw_counts = df[['lat', 'lon', 'year']].groupby(['lat', 'lon', 'year']).size().reset_index(name='count')
    locations_to_include = hw_counts[hw_counts['count'] <= threshold][['lat', 'lon']].drop_duplicates()
    df = df.merge(locations_to_include, on=['lat', 'lon'], how='left', indicator=True)
    return df[df['_merge'] == 'left_only'].drop(columns=['_merge'])

def filter_by_uhi_diff_category(df, threshold, category):
    """Filter dataframe by UHI difference category."""
    if category == 'Positive':
        return df[df['UHI_diff'] > threshold]
    elif category == 'Insignificant':
        return df[(df['UHI_diff'] >= -threshold) & (df['UHI_diff'] <= threshold)]
    elif category == 'Negative':
        return df[df['UHI_diff'] < -threshold]
    else:
        raise ValueError("Invalid category. Choose 'Positive', 'Insignificant', or 'Negative'.")

# Parse arguments
parser = argparse.ArgumentParser(description="Run UHI model for day or night data.")
parser.add_argument("--time_period", choices=["day", "night"], required=True, help="Specify whether to run for day or night data.")
parser.add_argument("--summary_dir", type=str, required=True, help="Directory for saving summary files and artifacts.")
parser.add_argument("--merged_feather_file", type=str, required=True, help="File name of the merged feather file containing the dataset.")
parser.add_argument("--iterations", type=int, default=100000, help="Number of iterations for the CatBoost model.")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the CatBoost model.")
parser.add_argument("--depth", type=int, default=10, help="Depth of the trees for the CatBoost model.")
parser.add_argument("--filters", type=str, default="", help="Comma-separated list of filter function names and parameters to apply to the dataframe.")
parser.add_argument("--run_type", type=str, default="test", help="Beginning part of experiment name")
parser.add_argument("--exp_name_extra", type=str, default="", help="Extra info that goes to the end of experiment name")
parser.add_argument("--shap_calculation", action="store_true", help="If set, SHAP-related calculations and graphs will be performed.")
parser.add_argument("--feature_column", type=str, default="X_vars2", help="Column name in df_daily_vars to select features")
parser.add_argument("--include_delta", action="store_true", help="Include delta variables in the feature list")

args = parser.parse_args()

print("Starting UHI model script...")

# Set summary directory and experiment name
summary_dir = args.summary_dir
experiment_name = f'{args.run_type}_{args.time_period.capitalize()}_{args.exp_name_extra}'

print(f"Setting up MLflow experiment: {experiment_name}")
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment(experiment_name)
mlflow.start_run()

# Log command line arguments to MLflow
mlflow.log_param("summary_dir", args.summary_dir)
mlflow.log_param("merged_feather_file", args.merged_feather_file)
mlflow.log_param("time_period", args.time_period)
mlflow.log_param("iterations", args.iterations)
mlflow.log_param("learning_rate", args.learning_rate)
mlflow.log_param("depth", args.depth)
mlflow.log_param("feature_selection_column", args.feature_column)
mlflow.log_param("include_delta_variables", args.include_delta)

# Log the full command line
command_line = f"python {' '.join(sys.argv)}"
mlflow.log_param("command_line", command_line)

# Create directory for saving figures and artifacts
figure_dir = os.path.join(summary_dir, 'mlflow', experiment_name)
os.makedirs(figure_dir, exist_ok=True)
print(f"Created figure directory: {figure_dir}")

# Load data
merged_feather_path = os.path.join(summary_dir, args.merged_feather_file)
print(f"Loading data from {merged_feather_path}")
local_hour_adjusted_df = pd.read_feather(merged_feather_path)
print(f"Loaded dataframe with shape: {local_hour_adjusted_df.shape}")

# Apply filters if any
if args.filters:
    print("Applying filters...")
    filter_function_pairs = args.filters.split(';')
    applied_filters = []
    for filter_function_pair in filter_function_pairs:
        filter_parts = filter_function_pair.split(',')
        filter_function_name = filter_parts[0]
        filter_params = filter_parts[1:]
        if filter_function_name in globals():
            print(f"Applying filter: {filter_function_name} with parameters {filter_params}")
            local_hour_adjusted_df = globals()[filter_function_name](local_hour_adjusted_df, *filter_params)
            applied_filters.append(f"{filter_function_name}({','.join(filter_params)})")
        else:
            print(f"Warning: Filter function {filter_function_name} not found. Skipping.")
    
    # Log applied filters as a parameter
    mlflow.log_param("applied_filters", "; ".join(applied_filters))
    print(f"Dataframe shape after applying filters: {local_hour_adjusted_df.shape}")
else:
    mlflow.log_param("applied_filters", "None")
    print("No filters applied")

# Load location ID dataset
location_ID_path = os.path.join(summary_dir, 'location_IDs.nc')
print(f"Loading location ID dataset from {location_ID_path}")
location_ID_ds = xr.open_dataset(location_ID_path, engine='netcdf4')

# Load feature list
print("Loading feature list...")
df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')
daily_vars = df_daily_vars.loc[df_daily_vars[args.feature_column] == 'Y', 'Variable']
daily_var_lst = daily_vars.tolist()

# Load delta feature list
delta_vars = df_daily_vars.loc[df_daily_vars['X_vars_delta'] == 'Y', 'Variable']
delta_var_lst = delta_vars.tolist()

# Calculate delta variables and add to dataframe if include_delta is True
if args.include_delta:
    print("Calculating delta variables...")
    for var in delta_var_lst:
        var_U = f"{var}_U"
        var_R = f"{var}_R"
        delta_var = f"delta_{var}"
        if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
            local_hour_adjusted_df[delta_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
            daily_var_lst.append(delta_var)  # Add delta variable to daily_var_lst
        else:
            print(f"Warning: {var_U} or {var_R} not found in dataframe columns.")
else:
    print("Skipping delta variable calculation.")

print(f"Final feature list: {daily_var_lst}")

# Save df_daily_vars to Excel file and log as artifact
df_daily_vars_path = os.path.join(figure_dir, 'hourlyDataSchema.xlsx')
df_daily_vars.to_excel(df_daily_vars_path, index=False)
mlflow.log_artifact(df_daily_vars_path)
print(f"Saved df_daily_vars to {df_daily_vars_path}")

# Save daily_var_lst to text file and log as artifact
daily_var_lst_path = os.path.join(figure_dir, 'daily_var_lst.txt')
with open(daily_var_lst_path, 'w') as f:
    for var in daily_var_lst:
        f.write(f"{var}\n")
mlflow.log_artifact(daily_var_lst_path)
print(f"Saved daily_var_lst to {daily_var_lst_path}")

# Define helper functions
def get_long_name(var_name, df_daily_vars):
    """Get long name for a variable."""
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
    """Add long names to a dataframe."""
    input_df['Long Name'] = input_df[join_column].apply(lambda x: get_long_name(x, df_daily_vars))
    return input_df

# Define day and night masks
print("Defining day and night masks...")
daytime_mask = local_hour_adjusted_df['local_hour'].between(8, 16)
nighttime_mask = (local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 4))

# Separate daytime and nighttime data
print(f"Separating {args.time_period} data...")
if args.time_period == "day":
    uhi_diff = local_hour_adjusted_df[daytime_mask]
else:
    uhi_diff = local_hour_adjusted_df[nighttime_mask]

X = uhi_diff[daily_var_lst]
y = uhi_diff['UHI_diff']
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Define linear slope function with error handling
def feature_linear_slope(df, feature_name, label, confidence_level=0.95):
    """Calculate linear slope with error handling."""
    try:
        slope, _, _, p_value, stderr = linregress(df[feature_name], df[label])
        t_value = np.abs(np.percentile(np.random.standard_t(df[feature_name].shape[0] - 2, 100000), [(1-confidence_level)/2, 1-(1-confidence_level)/2]))[1]
        margin_of_error = t_value * stderr
        return slope, margin_of_error, p_value
    except Exception as e:
        print(f"Error calculating slope for {feature_name}: {e}")
        return None, None, None

def combine_slopes(daytime_df, nighttime_df, features, labels=['UHI', 'UHI_diff'], confidence_level=0.95):
    """Combine slopes for day and night data."""
    data = {}
    for feature in features:
        slopes_with_intervals = []
        for df, time in [(daytime_df, 'Day'), (nighttime_df, 'Night')]:
            for label in labels:
                print(f"Calculating slope for {time}time {label} vs {feature}")
                slope, margin_of_error, p_value = feature_linear_slope(df, feature, label, confidence_level)
                if slope is not None:
                    slope_with_interval = f"{slope:.6f} (± {margin_of_error:.6f}, P: {p_value:.6f})"
                else:
                    slope_with_interval = "Error in calculation"
                slopes_with_intervals.append(slope_with_interval)
        data[feature] = slopes_with_intervals
    columns = [f'{time}_{label}_slope' for time in ['Day', 'Night'] for label in labels]
    results_df = pd.DataFrame(data, index=columns).transpose()
    return results_df

print("Calculating feature slopes...")
feature_names = daily_var_lst
results_df = combine_slopes(local_hour_adjusted_df[daytime_mask], local_hour_adjusted_df[nighttime_mask], feature_names)
sorted_results_df = results_df.sort_values('Day_UHI_slope', ascending=False)

# Save and log the sorted results DataFrame
sorted_results_path = os.path.join(figure_dir, 'sorted_results_df.csv')
sorted_results_df.to_csv(sorted_results_path)
mlflow.log_artifact(sorted_results_path)
print(f"Saved sorted results to {sorted_results_path}")

# Train and evaluate models
def train_and_evaluate(time_uhi_diff, daily_var_lst, model_name, iterations, learning_rate, depth):
    """Train and evaluate CatBoost model."""
    print(f"Training and evaluating {model_name}...")
    X = time_uhi_diff[daily_var_lst]
    y = time_uhi_diff['UHI_diff']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    train_pool = Pool(X_train, y_train)
    validation_pool = Pool(X_val, y_val)
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
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
    
    print(f"Model {model_name} metrics:")
    print(f"Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}")
    print(f"Train R2: {train_r2:.4f}, Validation R2: {val_r2:.4f}")
    
    return model

print("Training the model...")
model = train_and_evaluate(uhi_diff, daily_var_lst=daily_var_lst, model_name=f"{args.time_period}_model", 
                           iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth)

# Log model
print("Logging the trained model...")
mlflow.catboost.log_model(model, f"{args.time_period}_model")

# SHAP-related calculations and plotting
if args.shap_calculation:
    print("Starting SHAP calculations...")
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
    print("Creating feature importance plots...")
    feature_importance = get_ordered_feature_importance(model, full_pool)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title(f'{args.time_period.capitalize()}time Feature Importance')
    mlflow.log_figure(plt.gcf(), f'{args.time_period}time_feature_importance.png')
    plt.clf()

    # SHAP summary plots
    print("Creating SHAP summary plots...")
    shap_values = model.get_feature_importance(full_pool, type='ShapValues')[:,:-1]

    # Save SHAP values and feature names
    shap_values_path = os.path.join(figure_dir, 'shap_values.npy')
    np.save(shap_values_path, shap_values)
    mlflow.log_artifact(shap_values_path)
    print(f"Saved SHAP values to {shap_values_path}")

    feature_names_path = os.path.join(figure_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for feature in full_pool.get_feature_names():
            f.write(f"{feature}\n")
    mlflow.log_artifact(feature_names_path)
    print(f"Saved feature names to {feature_names_path}")

    # Log X data
    X_path = os.path.join(figure_dir, 'X_data.feather')
    X.to_feather(X_path)
    mlflow.log_artifact(X_path)
    print(f"Saved X data to {X_path}")

    shap.summary_plot(shap_values, X, show=False)
    plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_summary_plot.png')
    plt.clf()

    # SHAP waterfall plots
    print("Creating SHAP waterfall plots...")
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
    print("Creating SHAP dependence plots...")
    for feature in top_features:
        print(f"Creating dependence plot for {feature}")
        plot_dependence_grid(shap_values, X, feature_names=full_pool.get_feature_names(), time_period=args.time_period, target_feature=feature, plots_per_row=2)

print("Script execution completed.")
mlflow.end_run()

# Sample code for using saved SHAP data later
'''
import mlflow
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Set up MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_name = "your_experiment_name"
mlflow.set_experiment(experiment_name)

# Load the run you want to analyze
run_id = "your_run_id"
run = mlflow.get_run(run_id)

# Load SHAP values
shap_values_path = mlflow.artifacts.download_artifacts(run_id, "shap_values.npy")
shap_values = np.load(shap_values_path)

# Load feature names
feature_names_path = mlflow.artifacts.download_artifacts(run_id, "feature_names.txt")
with open(feature_names_path, 'r') as f:
    feature_names = [line.strip() for line in f]

# Load X data
X_path = mlflow.artifacts.download_artifacts(run_id, "X_data.csv")
X = pd.read_feather(X_path)

# Now you can create SHAP plots
shap.summary_plot(shap_values, X, feature_names=feature_names)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.show()

# Create a SHAP dependence plot
shap.dependence_plot("feature_name", shap_values, X, feature_names=feature_names)
plt.title("SHAP Dependence Plot")
plt.tight_layout()
plt.show()

# Create a SHAP force plot for a single prediction
shap.force_plot(shap.expected_value[0], shap_values[0], X.iloc[0], feature_names=feature_names)
plt.title("SHAP Force Plot")
plt.tight_layout()
plt.show()
'''
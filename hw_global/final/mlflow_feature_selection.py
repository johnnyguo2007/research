import pandas as pd
import numpy as np
import xarray as xr
import os
import netCDF4
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.catboost
import mlflow.shap
from scipy.stats import linregress
import argparse
import sys
import torch
from catboost import CatBoostRegressor, EFeaturesSelectionAlgorithm, EShapCalcType


# Define filter functions
def filter_by_year(df, year):
    return df[df['year'] == int(year)]


def filter_by_temperature_above_300(df, temperature):
    return df[df['temperature'] > float(temperature)]

def filter_by_KGMajorClass(df, major_class):
    return df[df['KGMajorClass'] == major_class]

def filter_by_hw_count(df, threshold):
    threshold = int(threshold)
    hw_counts = df[['lat', 'lon', 'year']].groupby(['lat', 'lon', 'year']).size().reset_index(name='count')
    locations_to_include = hw_counts[hw_counts['count'] <= threshold][['lat', 'lon']].drop_duplicates()
    df = df.merge(locations_to_include, on=['lat', 'lon'], how='left', indicator=True)
    return df[df['_merge'] == 'left_only'].drop(columns=['_merge'])


def filter_by_uhi_diff_category(df, threshold, category):
    if category == 'Positive':
        return df[df['UHI_diff'] > threshold]
    elif category == 'Insignificant':
        return df[(df['UHI_diff'] >= -threshold) & (df['UHI_diff'] <= threshold)]
    elif category == 'Negative':
        return df[df['UHI_diff'] < -threshold]
    else:
        raise ValueError("Invalid category. Choose 'Positive', 'Insignificant', or 'Negative'.")


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CustomRFECV(RFECV):
    def _fit(self, X, y, step_score=None):
        clear_gpu_memory()
        return super()._fit(X, y, step_score)


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


def combine_slopes(daytime_df, nighttime_df, features, labels=['UHI', 'UHI_diff'], confidence_level=0.95):
    """Combine slopes for day and night data."""
    data = {}
    for feature in features:
        slopes_with_intervals = []
        for df, time in [(daytime_df, 'Day'), (nighttime_df, 'Night')]:
            for label in labels:
                # print(f"Calculating slope for {time}time {label} vs {feature}")
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


def get_feature_group(feature_name, df_daily_vars):
    if feature_name.startswith('delta_'):
        return feature_name.replace('delta_', '')
    elif feature_name.startswith('hw_nohw_diff_'):
        return feature_name.replace('hw_nohw_diff_', '')
    elif feature_name.startswith('Double_Differencing_'):
        return feature_name.replace('Double_Differencing_', '')
    else:
        return feature_name


def train_and_evaluate(time_uhi_diff, daily_var_lst, feature_groups, model_name, iterations, learning_rate, depth,
                       feature_selection=False, num_features=None):
    print(f"Training and evaluating {model_name}...")
    X = time_uhi_diff[daily_var_lst]
    y = time_uhi_diff['UHI_diff']

    mean_uhi_diff = y.mean()
    mlflow.log_metric(f"{args.time_period}_mean_uhi_diff", mean_uhi_diff)
    print(f"Logged mean value of UHI_diff for {args.time_period}time: {mean_uhi_diff:.4f}")

    clear_gpu_memory()

    if num_features is not None:
        print(f"Using CatBoost's select_features to select {num_features} features...")
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            task_type='GPU',
            devices='0',
            verbose=False
        )

        # Create a dictionary to map features to their corresponding groups
        feature_to_group = dict(zip(daily_var_lst, feature_groups))

        # Create a new array to store the feature group for each feature
        feature_group_array = np.array([feature_to_group[feature] for feature in daily_var_lst])

        # Create a path for the feature selection plot
        feature_selection_plot_path = os.path.join(figure_dir, f'{model_name}_feature_selection_plot.png')

        summary = model.select_features(
            X, y,
            features_for_select=daily_var_lst,
            num_features_to_select=num_features,
            steps=8,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=True,
            logging_level='Silent',
            plot=True,
            plot_file=feature_selection_plot_path,  # Enable plot generation and specify the file path
            eval_feature_groups=feature_group_array  # Specify the feature groups
        )

        # Log the feature selection plot as an artifact
        mlflow.log_artifact(feature_selection_plot_path)
        print(f"Feature selection plot saved to {feature_selection_plot_path}")

        selected_features = summary['selected_features']
        eliminated_features = summary['eliminated_features']

        print(f"Selected features: {selected_features}")
        print(f"Eliminated features: {eliminated_features}")

        X_selected = X.iloc[:, selected_features]
        optimal_num_features = len(selected_features)
    elif feature_selection:
        base_model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            task_type='GPU',
            devices='0',
            verbose=False
        )

        rfecv = CustomRFECV(
            estimator=base_model,
            step=1,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=1
        )

        print("Starting RFECV...")
        rfecv.fit(X, y)
        print("RFECV completed.")

        optimal_num_features = rfecv.n_features_
        selected_features = X.columns[rfecv.support_].tolist()

        print(f"Optimal number of features: {optimal_num_features}")
        print(f"Selected features: {selected_features}")

        X_selected = X[selected_features]

        # Plot number of features VS. cross-validation scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross-validation score (neg_mean_squared_error)")
        plt.title("Optimal number of features")
        mlflow.log_figure(plt.gcf(), f"{model_name}_feature_selection_plot.png")
        plt.close()
    else:
        X_selected = X
        optimal_num_features = len(daily_var_lst)
        selected_features = daily_var_lst

    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.1, random_state=42)

    clear_gpu_memory()

    final_model = model if num_features is not None else CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=False
    )

    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50,
                    verbose=False)

    # Log model parameters
    mlflow.log_param(f"{model_name}_iterations", final_model.get_param('iterations'))
    mlflow.log_param(f"{model_name}_learning_rate", final_model.get_param('learning_rate')) 
    mlflow.log_param(f"{model_name}_depth", final_model.get_param('depth'))
    mlflow.log_param(f"{model_name}_optimal_num_features", optimal_num_features)

    if isinstance(selected_features[0], str):
        mlflow.log_param(f"{model_name}_selected_features", ", ".join(selected_features))
    else:
        mlflow.log_param(f"{model_name}_selected_features", ", ".join([daily_var_lst[idx] for idx in selected_features]))

    # Calculate and log metrics
    y_pred_train = final_model.predict(X_train)
    y_pred_val = final_model.predict(X_val)
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

    return final_model, selected_features, X_selected

# Parse arguments
parser = argparse.ArgumentParser(description="Run UHI model for day or night data.")
parser.add_argument("--time_period", choices=["day", "night"], required=True,
                    help="Specify whether to run for day or night data.")
parser.add_argument("--summary_dir", type=str, required=True, help="Directory for saving summary files and artifacts.")
parser.add_argument("--merged_feather_file", type=str, required=True,
                    help="File name of the merged feather file containing the dataset.")
parser.add_argument("--iterations", type=int, default=100000, help="Number of iterations for the CatBoost model.")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the CatBoost model.")
parser.add_argument("--depth", type=int, default=10, help="Depth of the trees for the CatBoost model.")
parser.add_argument("--filters", type=str, default="",
                    help="Comma-separated list of filter function names and parameters to apply to the dataframe.")
parser.add_argument("--run_type", type=str, default="test", help="Beginning part of experiment name")
parser.add_argument("--exp_name_extra", type=str, default="", help="Extra info that goes to the end of experiment name")
parser.add_argument("--shap_calculation", action="store_true",
                    help="If set, SHAP-related calculations and graphs will be performed.")
parser.add_argument("--feature_column", type=str, default="X_vars2",
                    help="Column name in df_daily_vars to select features")
parser.add_argument("--delta_column", type=str, default="X_vars_delta",
                    help="Column name in df_daily_vars to select delta features")
parser.add_argument("--hw_nohw_diff_column", type=str, default="HW_NOHW_Diff",
                    help="Column name in df_daily_vars to select HW-NoHW diff features")
parser.add_argument("--double_diff_column", type=str, default="Double_Diff",
                    help="Column name in df_daily_vars to select Double Differencing features")
parser.add_argument("--delta_mode", choices=["none", "include", "only"], default="include",
                    help="'none': don't use delta variables, 'include': use both original and delta variables, 'only': use only delta variables")
parser.add_argument("--feature_selection", action="store_true", help="If set, perform feature selection using RFECV")
parser.add_argument("--num_features", type=int, default=None, help="Number of features to select using CatBoost's feature selection")
parser.add_argument("--daily_freq", action="store_true",
                    help="If set, data will be averaged by day before training the model.")

args = parser.parse_args()

print("Starting UHI model script...")

# Set summary directory and experiment name
summary_dir = args.summary_dir
experiment_name = f'{args.run_type}_{args.time_period.capitalize()}_{args.exp_name_extra}'

print(f"Setting up MLflow experiment: {experiment_name}")
mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")
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
mlflow.log_param("delta_selection_column", args.delta_column)
mlflow.log_param("hw_nohw_diff_column", args.hw_nohw_diff_column)
mlflow.log_param("double_diff_column", args.double_diff_column)
mlflow.log_param("include_delta_variables", args.delta_mode)
mlflow.log_param("perform_feature_selection", args.feature_selection)

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
    mlflow.log_param(f"data_shape_after_filers", local_hour_adjusted_df.shape)
else:
    mlflow.log_param("applied_filters", "None")
    print("No filters applied")

# Load feature list
print("Loading feature list...")
df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')

daily_vars = df_daily_vars.loc[df_daily_vars[args.feature_column] == 'Y', 'Variable']
daily_var_lst = daily_vars.tolist()

delta_vars = df_daily_vars.loc[df_daily_vars[args.delta_column] == 'Y', 'Variable']
delta_var_lst = delta_vars.tolist()

hw_nohw_diff_vars = df_daily_vars.loc[df_daily_vars[args.hw_nohw_diff_column] == 'Y', 'Variable']
# Add HW-NoHW diff variables to daily_var_lst
daily_var_lst.extend([f"hw_nohw_diff_{var}" for var in hw_nohw_diff_vars])

# Add Double Differencing variables
double_diff_vars = df_daily_vars.loc[df_daily_vars[args.double_diff_column] == 'Y', 'Variable']
daily_var_lst.extend([f"Double_Differencing_{var}" for var in double_diff_vars])

# Log the feature selection column and delta mode
mlflow.log_param("feature_selection_column", args.feature_column)
mlflow.log_param("delta_selection_column", args.delta_column)
mlflow.log_param("hw_nohw_diff_column", args.hw_nohw_diff_column)
mlflow.log_param("double_diff_column", args.double_diff_column)
mlflow.log_param("delta_mode", args.delta_mode)

# Calculate delta variables and modify daily_var_lst based on delta_mode
if args.delta_mode in ["include", "only"]:
    print("Calculating delta variables...")
    for var in delta_var_lst:
        var_U = f"{var}_U"
        var_R = f"{var}_R"
        delta_var = f"delta_{var}"
        if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
            local_hour_adjusted_df[delta_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
        else:
            print(f"Warning: {var_U} or {var_R} not found in dataframe columns.")

    if args.delta_mode == "only":
        print("Using only delta variables...")
        daily_var_lst = [f"delta_{var}" for var in delta_var_lst]
    else:  # "include"
        print("Including delta variables...")
        daily_var_lst += [f"delta_{var}" for var in delta_var_lst]
else:
    print("Using original variables only...")

# Calculate Double Differencing variables
print("Calculating Double Differencing variables...")
for var in double_diff_vars:
    var_U = f"hw_nohw_diff_{var}_U"
    var_R = f"hw_nohw_diff_{var}_R"
    double_diff_var = f"Double_Differencing_{var}"
    if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
        local_hour_adjusted_df[double_diff_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
    else:
        print(f"Warning: {var_U} or {var_R} not found in dataframe columns.")

print(f"Final feature list: {daily_var_lst}")

# Save df_daily_vars to Excel file and log as artifact
df_daily_vars_path = os.path.join(figure_dir, 'hourlyDataSchema.xlsx')
df_daily_vars.to_excel(df_daily_vars_path, index=False)
mlflow.log_artifact(df_daily_vars_path)
print(f"Saved df_daily_vars to {df_daily_vars_path}")


def get_long_name(var_name, df_daily_vars):
    if var_name.startswith('delta_'):
        original_var_name = var_name.replace('delta_', '')
        original_long_name = df_daily_vars.loc[df_daily_vars['Variable'] == original_var_name, 'Long Name'].values
        if original_long_name.size > 0:
            return f"U and R Difference of {original_long_name[0]}"
        else:
            return f"U and R Difference of {original_var_name} (No long name found)"
    elif var_name.startswith('hw_nohw_diff_'):
        original_var_name = var_name.replace('hw_nohw_diff_', '')
        original_long_name = df_daily_vars.loc[df_daily_vars['Variable'] == original_var_name, 'Long Name'].values
        if original_long_name.size > 0:
            return f"HW Non-HW Difference of {original_long_name[0]}"
        else:
            return f"HW Non-HW Difference of {original_var_name} (No long name found)"
    elif var_name.startswith('Double_Differencing_'):
        original_var_name = var_name.replace('Double_Differencing_', '')  
        original_long_name = df_daily_vars.loc[df_daily_vars['Variable'] == original_var_name, 'Long Name'].values
        if original_long_name.size > 0:
            return f"Double Difference of {original_long_name[0]}"
        else:
            return f"Double Difference of {original_var_name} (No long name found)"
    else:
        long_name = df_daily_vars.loc[df_daily_vars['Variable'] == var_name, 'Long Name'].values
        if long_name.size > 0:
            return long_name[0]
        else:
            return f"{var_name} (No long name found)"


def add_long_name(input_df, join_column='Feature', df_daily_vars=df_daily_vars):
    input_df['Long Name'] = input_df[join_column].apply(lambda x: get_long_name(x, df_daily_vars))
    return input_df


# Save daily_var_lst to text file and log as artifact
daily_var_lst_path = os.path.join(figure_dir, 'daily_var_lst.txt')
with open(daily_var_lst_path, 'w') as f:
    for var in daily_var_lst:
        f.write(f"{var}\n")
mlflow.log_artifact(daily_var_lst_path)
print(f"Saved daily_var_lst to {daily_var_lst_path}")

# Define day and night masks
print("Defining day and night masks...")
daytime_mask = local_hour_adjusted_df['local_hour'].between(7, 16)
nighttime_mask = (
        local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 6))

local_hour_adjusted_df['date'] = pd.to_datetime(local_hour_adjusted_df['local_time']).dt.date
# Separate daytime and nighttime data
print(f"Separating {args.time_period} data...")
if args.time_period == "day":
    uhi_diff = local_hour_adjusted_df[daytime_mask]
else:
    uhi_diff = local_hour_adjusted_df[nighttime_mask]

if args.daily_freq:
    print("Calculating daily average...")
    # Subset the columns before grouping
    uhi_diff = uhi_diff[daily_var_lst + ['UHI_diff', 'lat', 'lon', 'date']]

    # Now perform the grouping and aggregation
    uhi_diff = uhi_diff.groupby(['lat', 'lon', 'date']).mean().reset_index()

X = uhi_diff[daily_var_lst]
y = uhi_diff['UHI_diff']
print(f"X shape: {X.shape}, y shape: {y.shape}")

print("Calculating feature slopes...")
feature_names = daily_var_lst
results_df = combine_slopes(local_hour_adjusted_df[daytime_mask], local_hour_adjusted_df[nighttime_mask], feature_names)
sorted_results_df = results_df.sort_values('Day_UHI_slope', ascending=False)

# Save and log the sorted results DataFrame
sorted_results_path = os.path.join(figure_dir, 'sorted_results_df.csv')
sorted_results_df.to_csv(sorted_results_path)
mlflow.log_artifact(sorted_results_path)
print(f"Saved sorted results to {sorted_results_path}")

feature_groups = [get_feature_group(feature, df_daily_vars) for feature in daily_var_lst]

print("Training the model...")
model, selected_features, X_selected = train_and_evaluate(uhi_diff, daily_var_lst=daily_var_lst, feature_groups=feature_groups,
                                                          model_name=f"{args.time_period}_model",
                                                          iterations=args.iterations, learning_rate=args.learning_rate,
                                                          depth=args.depth, feature_selection=args.feature_selection,
                                                          num_features=args.num_features)

# Log model
print("Logging the trained model...")
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


full_pool = Pool(X_selected, y)

# Feature importance plots
print("Creating feature importance plots...")
feature_importance = get_ordered_feature_importance(model, full_pool)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title(f'{args.time_period.capitalize()}time Feature Importance')
mlflow.log_figure(plt.gcf(), f'{args.time_period}time_feature_importance.png')
plt.clf()

# Log feature_importance data
feature_importance_path = os.path.join(figure_dir, 'feature_importance.feather')
feature_importance.to_feather(feature_importance_path)
mlflow.log_artifact(feature_importance_path)
print(f"Saved feature_importance data to {feature_importance_path}")

# SHAP-related calculations and plotting
if args.shap_calculation:
    print("Starting SHAP calculations...")
    # SHAP summary plots
    print("Creating SHAP summary plots...")
    shap_values = model.get_feature_importance(full_pool, type='ShapValues')[:, :-1]

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
    X_selected.to_feather(X_path)
    mlflow.log_artifact(X_path)
    print(f"Saved X data to {X_path}")

    shap.summary_plot(shap_values, X_selected, show=False)
    plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_summary_plot.png')
    plt.clf()

    # SHAP waterfall plots
    print("Creating SHAP waterfall plots...")
    feature_importances = model.get_feature_importance()
    expected_value = shap_values[0, -1]
    long_names = [get_long_name(f, df_daily_vars) for f in full_pool.get_feature_names()]
    shap.waterfall_plot(shap.Explanation(feature_importances, base_values=expected_value, feature_names=long_names),
                        show=False)
    plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
    plt.gcf().subplots_adjust(left=0.3)  # Increase left margin to make room for y-axis labels
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_waterfall_plot.png')
    plt.clf()

    # Get feature importance using SHAP values
    def get_shap_feature_importance(shap_values, feature_names, df_daily_vars):
        shap_feature_importance = np.abs(shap_values).mean(axis=0)
        total_importance = np.sum(shap_feature_importance)
        shap_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': shap_feature_importance,
            'Percentage': (shap_feature_importance / total_importance) * 100
        })
        shap_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        shap_importance_df = add_long_name(shap_importance_df, join_column='Feature', df_daily_vars=df_daily_vars)
        return shap_importance_df

    shap_feature_importance = get_shap_feature_importance(shap_values, full_pool.get_feature_names(), df_daily_vars)
    shap_feature_importance['Feature Group'] = shap_feature_importance['Feature'].apply(lambda x: get_feature_group(x, df_daily_vars))

    shap_feature_importance_by_group = shap_feature_importance.groupby('Feature Group')['Importance'].sum().reset_index()

    # Calculate the total importance
    total_importance = shap_feature_importance_by_group['Importance'].sum()

    # Add the "Percentage" column
    shap_feature_importance_by_group['Percentage'] = (shap_feature_importance_by_group['Importance'] / total_importance) * 100

    # Sort by "Importance" in descending order
    shap_feature_importance_by_group.sort_values(by='Importance', ascending=False, inplace=True)

    shap_feature_importance_by_group_path = os.path.join(figure_dir, f'{args.time_period}_shap_feature_importance_by_group.csv')
    shap_feature_importance_by_group.to_csv(shap_feature_importance_by_group_path, index=False)
    mlflow.log_artifact(shap_feature_importance_by_group_path)
    print(f"Saved SHAP feature importance by group data to {shap_feature_importance_by_group_path}")

    # Create a dictionary to map features to their corresponding groups
    feature_to_group = dict(zip(shap_feature_importance['Feature'], shap_feature_importance['Feature Group']))

    # Create a new array to store the feature group for each feature
    feature_groups = np.array([feature_to_group[feature] for feature in full_pool.get_feature_names()])

    # Create a color mapping for feature groups
    unique_feature_groups = np.unique(feature_groups)
    color_map = plt.cm.get_cmap('tab20', len(unique_feature_groups))
    group_colors = color_map(np.arange(len(unique_feature_groups)))
    group_color_dict = dict(zip(unique_feature_groups, group_colors))

    # Map feature groups to colors
    feature_colors = [group_color_dict[group] for group in feature_groups]

    # Create the SHAP summary plot by feature groups
    shap.summary_plot(shap_values, X_selected, plot_type='bar', feature_names=full_pool.get_feature_names(), 
                    max_display=len(shap_feature_importance), show=False, color=feature_colors)
    plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_summary_plot_by_group.png')
    plt.clf()

    # SHAP feature importance waterfall plot
    print("Creating SHAP feature importance waterfall plot...")
    plt.figure(figsize=(12, 0.5 * len(shap_feature_importance)))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_feature_importance['Importance'].values,
            base_values=shap_values[:, -1].mean(),
            data=X_selected.iloc[0],  # Use the first row of X_selected
            feature_names=shap_feature_importance['Long Name'].tolist()
        ),
        show=False,
        max_display=len(shap_feature_importance)
    )
    plt.title(f'SHAP Waterfall Plot for {args.time_period.capitalize()}time')
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_feature_importance_waterfall_plot.png')
    plt.clf()

    # Percentage contribution horizontal bar plot
    print("Creating percentage contribution plot...")
    plt.figure(figsize=(12, 0.5 * len(shap_feature_importance)))
    plt.barh(shap_feature_importance['Long Name'], shap_feature_importance['Percentage'], 
             align='center', color='#ff0051')
    plt.title(f'Feature Importance (%) for {args.time_period.capitalize()}time')
    plt.xlabel('Percentage Contribution')
    plt.gca().invert_yaxis()
    
    for i, percentage in enumerate(shap_feature_importance['Percentage']):
        plt.text(percentage, i, f' {percentage:.1f}%', va='center')

    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_percentage_contribution_plot.png')
    plt.clf()

    # Log SHAP feature importance data
    shap_importance_path = os.path.join(figure_dir, f'{args.time_period}_shap_feature_importance.csv')
    shap_feature_importance.to_csv(shap_importance_path, index=False)
    mlflow.log_artifact(shap_importance_path)
    print(f"Saved SHAP feature importance data to {shap_importance_path}")

    # SHAP dependence plots
    def plot_dependence_grid(shap_values, X, feature_names, time_period, target_feature='U10'):
        feature_names = [f for f in feature_names if f != target_feature]

        # Create the nested folder structure
        nested_path = os.path.join('x_dependence_plots', target_feature)
        full_nested_path = os.path.join(figure_dir, nested_path)
        os.makedirs(full_nested_path, exist_ok=True)

        for i, feature_name in enumerate(feature_names):
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(
                ind=target_feature, 
                shap_values=shap_values, 
                features=X,
                interaction_index=feature_name, 
                ax=ax, 
                show=False
            )
            ax.set_title(f"{time_period.capitalize()} time {target_feature} vs {feature_name}")

            # Save the figure
            plot_filename = f'{time_period}_{target_feature}_vs_{feature_name}.png'
            plot_path = os.path.join(full_nested_path, plot_filename)
            fig.savefig(plot_path)
            plt.close(fig)

            # Log the artifact with the nested path
            mlflow.log_artifact(plot_path, nested_path)

        print(f"Dependence plots for {target_feature} have been created and logged to MLflow.")

top_features = feature_importance['Feature'].tolist()

# Dependence plots
print("Creating SHAP dependence plots...")
for feature in top_features:
    print(f"Creating dependence plots for {feature}")
    plot_dependence_grid(shap_values, X_selected, feature_names=full_pool.get_feature_names(),
                         time_period=args.time_period, target_feature=feature)

print("Script execution completed.")
mlflow.end_run()
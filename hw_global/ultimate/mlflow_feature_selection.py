# Import necessary libraries
import pandas as pd
import numpy as np
import xarray as xr  # Not used in the current script
import os
import netCDF4  # Not used in the current script
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV
from catboost import CatBoostRegressor, Pool, EFeaturesSelectionAlgorithm, EShapCalcType
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.catboost
import mlflow.shap
from scipy.stats import linregress
import argparse
import sys
import torch


# Define functions for filtering the dataframe
def filter_by_year(df, year):
    """Filters the dataframe by year."""
    return df[df['year'] == int(year)]


def filter_by_temperature_above_300(df, temperature):
    """Filters the dataframe by temperature above a given threshold."""
    return df[df['temperature'] > float(temperature)]


def filter_by_KGMajorClass(df, major_class):
    """Filters the dataframe by KGMajorClass."""
    return df[df['KGMajorClass'] == major_class]


def filter_by_hw_count(df, threshold):
    """Filters the dataframe by the number of heatwave events per location."""
    threshold = int(threshold)
    hw_counts = df[['lat', 'lon', 'year']].groupby(['lat', 'lon', 'year']).size().reset_index(name='count')
    locations_to_include = hw_counts[hw_counts['count'] <= threshold][['lat', 'lon']].drop_duplicates()
    df = df.merge(locations_to_include, on=['lat', 'lon'], how='left', indicator=True)
    return df[df['_merge'] == 'left_only'].drop(columns=['_merge'])


def filter_by_uhi_diff_category(df, threshold, category):
    """Filters the dataframe by UHI_diff category (Positive, Insignificant, or Negative)."""
    if category == 'Positive':
        return df[df['UHI_diff'] > threshold]
    elif category == 'Insignificant':
        return df[(df['UHI_diff'] >= -threshold) & (df['UHI_diff'] <= threshold)]
    elif category == 'Negative':
        return df[df['UHI_diff'] < -threshold]
    else:
        raise ValueError("Invalid category. Choose 'Positive', 'Insignificant', or 'Negative'.")


def clear_gpu_memory():
    """Clears GPU memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Custom RFECV class to clear GPU memory before each fit
class CustomRFECV(RFECV):
    def _fit(self, X, y, step_score=None):
        clear_gpu_memory()
        return super()._fit(X, y, step_score)


def feature_linear_slope(df, feature_name, label, confidence_level=0.95):
    """Calculates the linear slope between a feature and a label with margin of error and p-value."""
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
    """Combines slopes for day and night data for given features and labels."""
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


def get_feature_group(feature_name):
    """Extracts the feature group from the feature name."""
    prefixes = ['delta_', 'hw_nohw_diff_', 'Double_Differencing_']
    for prefix in prefixes:
        if feature_name.startswith(prefix):
            return feature_name.replace(prefix, '')
    return feature_name


def train_and_evaluate(time_uhi_diff, daily_var_lst, feature_groups, model_name, iterations, learning_rate, depth,
                       feature_selection=False, num_features=None):
    """Trains and evaluates a CatBoostRegressor model."""

    print(f"Training and evaluating {model_name}...")
    X = time_uhi_diff[daily_var_lst]
    y = time_uhi_diff['UHI_diff']

    # Log mean UHI_diff
    mean_uhi_diff = y.mean()
    mlflow.log_metric(f"{args.time_period}_mean_uhi_diff", mean_uhi_diff)
    print(f"Logged mean value of UHI_diff for {args.time_period}time: {mean_uhi_diff:.4f}")

    clear_gpu_memory()

    # Feature selection using CatBoost's built-in method
    if num_features is not None:
        print(f"Using CatBoost's select_features to select {num_features} features...")

        # Initialize CatBoost model
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            task_type='GPU',  # Use GPU for training
            devices='0',  # Specify GPU device index
            verbose=False  # Suppress verbose output
        )

        # Map features to their groups
        feature_to_group = dict(zip(daily_var_lst, feature_groups))
        feature_group_array = np.array([feature_to_group[feature] for feature in daily_var_lst])

        # Path for the feature selection plot
        feature_selection_plot_path = os.path.join(figure_dir, f'{model_name}_feature_selection_plot.png')

        # Perform feature selection
        summary = model.select_features(
            X, y,
            features_for_select=daily_var_lst,
            num_features_to_select=num_features,
            steps=8,  # Number of selection steps
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,  # Use SHAP values for selection
            shap_calc_type=EShapCalcType.Regular,  # Type of SHAP calculation
            train_final_model=True,  # Train the final model on selected features
            logging_level='Silent',  # Suppress output
            plot=True,  # Generate a plot
            plot_file=feature_selection_plot_path,  # Save the plot to a file
            eval_feature_groups=feature_group_array  # Use feature groups
        )

        # Log the feature selection plot
        mlflow.log_artifact(feature_selection_plot_path)
        print(f"Feature selection plot saved to {feature_selection_plot_path}")

        selected_features = summary['selected_features']
        eliminated_features = summary['eliminated_features']

        print(f"Selected features: {selected_features}")
        print(f"Eliminated features: {eliminated_features}")

        X_selected = X.iloc[:, selected_features]
        optimal_num_features = len(selected_features)

    # Feature selection using RFECV
    elif feature_selection:
        print("Starting feature selection using RFECV...")

        # Initialize base model for RFECV
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

        # Initialize RFECV
        rfecv = CustomRFECV(
            estimator=base_model,
            step=1,  # Number of features to eliminate at each step
            cv=KFold(5, shuffle=True, random_state=42),  # Cross-validation strategy
            scoring='neg_mean_squared_error',  # Scoring metric
            n_jobs=1,  # Number of parallel jobs
            verbose=1  # Verbose output
        )

        # Fit RFECV
        rfecv.fit(X, y)
        print("RFECV completed.")

        optimal_num_features = rfecv.n_features_
        selected_features = X.columns[rfecv.support_].tolist()

        print(f"Optimal number of features: {optimal_num_features}")
        print(f"Selected features: {selected_features}")

        X_selected = X[selected_features]

        # Plot feature selection results
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross-validation score (neg_mean_squared_error)")
        plt.title("Optimal number of features")
        mlflow.log_figure(plt.gcf(), f"{model_name}_feature_selection_plot.png")
        plt.close()

    # No feature selection
    else:
        X_selected = X
        optimal_num_features = len(daily_var_lst)
        selected_features = daily_var_lst

        # 10-fold Cross-validation
        print("Performing 10-fold cross-validation...")
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_model = CatBoostRegressor(
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

        cv_scores = cross_val_score(cv_model, X_selected, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
        rmse_scores = np.sqrt(-cv_scores)  # Convert negative MSE to RMSE

        avg_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        min_rmse = np.min(rmse_scores)
        max_rmse = np.max(rmse_scores)

        mlflow.log_metric(f"{model_name}_avg_cv_rmse", avg_rmse)
        mlflow.log_metric(f"{model_name}_std_cv_rmse", std_rmse)
        mlflow.log_metric(f"{model_name}_min_cv_rmse", min_rmse)
        mlflow.log_metric(f"{model_name}_max_cv_rmse", max_rmse)

        print(f"Average CV RMSE: {avg_rmse:.4f}")
        print(f"Standard Deviation of CV RMSE: {std_rmse:.4f}")
        print(f"Min CV RMSE: {min_rmse:.4f}")
        print(f"Max CV RMSE: {max_rmse:.4f}")

    # Train on the full dataset after cross-validation
    print("Training the final model on the full dataset...")
    clear_gpu_memory()
    final_model = CatBoostRegressor(
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
    final_model.fit(X_selected, y, verbose=False)
    mlflow.log_param(f"{model_name}_iterations", final_model.get_param('iterations'))
    mlflow.log_param(f"{model_name}_learning_rate", final_model.get_param('learning_rate'))
    mlflow.log_param(f"{model_name}_depth", final_model.get_param('depth'))
    mlflow.log_param(f"{model_name}_optimal_num_features", optimal_num_features)

    if isinstance(selected_features[0], str):
        mlflow.log_param(f"{model_name}_selected_features", ", ".join(selected_features))
    else:
        mlflow.log_param(f"{model_name}_selected_features", ", ".join([daily_var_lst[idx] for idx in selected_features]))


    # Evaluate the model and log metrics
    y_pred = final_model.predict(X_selected)

    train_rmse = mean_squared_error(y, y_pred, squared=False)
    train_r2 = r2_score(y, y_pred)


    mlflow.log_metric(f"{model_name}_whole_rmse", train_rmse)
    mlflow.log_metric(f"{model_name}_whole_r2", train_r2)

    print(f"Model {model_name} metrics:")
    print(f"Whole RMSE: {train_rmse:.4f}")
    print(f"Whole R2: {train_r2:.4f}")
    
    if False:
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.1, random_state=42)

        clear_gpu_memory()

        # Initialize and train the final model
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

        # Log model parameters and selected features
        mlflow.log_param(f"{model_name}_iterations", final_model.get_param('iterations'))
        mlflow.log_param(f"{model_name}_learning_rate", final_model.get_param('learning_rate'))
        mlflow.log_param(f"{model_name}_depth", final_model.get_param('depth'))
        mlflow.log_param(f"{model_name}_optimal_num_features", optimal_num_features)

        if isinstance(selected_features[0], str):
            mlflow.log_param(f"{model_name}_selected_features", ", ".join(selected_features))
        else:
            mlflow.log_param(f"{model_name}_selected_features", ", ".join([daily_var_lst[idx] for idx in selected_features]))


        # Evaluate the model and log metrics
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


# Argument parsing
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
                    help="Comma-separated list of filter function names and parameters to apply to the dataframe." +
                         "Multiple filters can be chained using semicolons. " +
                         "Example: --filters filter_by_year,2020;filter_by_temperature_above_300,305")
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
parser.add_argument("--num_features", type=int, default=None,
                    help="Number of features to select using CatBoost's feature selection. If provided, --feature_selection will be ignored.")
parser.add_argument("--daily_freq", action="store_true",
                    help="If set, data will be averaged by day before training the model.")

args = parser.parse_args()

print("Starting UHI model script...")

# Set up MLflow experiment
summary_dir = args.summary_dir
experiment_name = f'{args.run_type}_{args.time_period.capitalize()}_{args.exp_name_extra}'

print(f"Setting up MLflow experiment: {experiment_name}")
mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")  # Set your MLflow tracking URI
mlflow.set_experiment(experiment_name)
mlflow.start_run()

# Log command line arguments and the full command line
for arg, value in vars(args).items():
    mlflow.log_param(arg, value)

command_line = f"python {' '.join(sys.argv)}"
mlflow.log_param("command_line", command_line)

# Create directory for figures and artifacts
figure_dir = os.path.join(summary_dir, 'mlflow', experiment_name)
os.makedirs(figure_dir, exist_ok=True)
print(f"Created figure directory: {figure_dir}")

# Load data
merged_feather_path = os.path.join(summary_dir, args.merged_feather_file)
print(f"Loading data from {merged_feather_path}")
local_hour_adjusted_df = pd.read_feather(merged_feather_path)
print(f"Loaded dataframe with shape: {local_hour_adjusted_df.shape}")

# Apply filters
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

    # Log applied filters
    mlflow.log_param("applied_filters", "; ".join(applied_filters))
    print(f"Dataframe shape after applying filters: {local_hour_adjusted_df.shape}")
    mlflow.log_param(f"data_shape_after_filers", local_hour_adjusted_df.shape)
else:
    mlflow.log_param("applied_filters", "None")
    print("No filters applied")


# Load feature list from Excel file
print("Loading feature list...")
df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')  # Replace with your file path

# Select features based on columns specified in arguments
daily_vars = df_daily_vars.loc[df_daily_vars[args.feature_column] == 'Y', 'Variable']
daily_var_lst = daily_vars.tolist()

delta_vars = df_daily_vars.loc[df_daily_vars[args.delta_column] == 'Y', 'Variable']
delta_var_lst = delta_vars.tolist()

hw_nohw_diff_vars = df_daily_vars.loc[df_daily_vars[args.hw_nohw_diff_column] == 'Y', 'Variable']
daily_var_lst.extend([f"hw_nohw_diff_{var}" for var in hw_nohw_diff_vars])  # Add HW-NoHW diff features

double_diff_vars = df_daily_vars.loc[df_daily_vars[args.double_diff_column] == 'Y', 'Variable']
daily_var_lst.extend([f"Double_Differencing_{var}" for var in double_diff_vars])  # Add Double Differencing features


# Calculate delta variables based on delta_mode argument
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

# Save and log df_daily_vars and daily_var_lst
df_daily_vars_path = os.path.join(figure_dir, 'hourlyDataSchema.xlsx')
df_daily_vars.to_excel(df_daily_vars_path, index=False)
mlflow.log_artifact(df_daily_vars_path)
print(f"Saved df_daily_vars to {df_daily_vars_path}")

daily_var_lst_path = os.path.join(figure_dir, 'daily_var_lst.txt')
with open(daily_var_lst_path, 'w') as f:
    for var in daily_var_lst:
        f.write(f"{var}\n")
mlflow.log_artifact(daily_var_lst_path)
print(f"Saved daily_var_lst to {daily_var_lst_path}")


# Function to get long names of variables
def get_long_name(var_name, df_daily_vars):
    """Retrieves the long name of a variable from df_daily_vars."""
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
    """Adds a 'Long Name' column to the input dataframe by mapping variable names to their long names."""
    input_df['Long Name'] = input_df[join_column].apply(lambda x: get_long_name(x, df_daily_vars))
    return input_df


# Define day and night masks and separate data based on time_period argument
print("Defining day and night masks...")
daytime_mask = local_hour_adjusted_df['local_hour'].between(7, 16)
nighttime_mask = (
        local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 6))

local_hour_adjusted_df['date'] = pd.to_datetime(local_hour_adjusted_df['local_time']).dt.date

print(f"Separating {args.time_period} data...")
if args.time_period == "day":
    uhi_diff = local_hour_adjusted_df[daytime_mask]
else:
    uhi_diff = local_hour_adjusted_df[nighttime_mask]

# Calculate daily average if daily_freq argument is set
if args.daily_freq:
    print("Calculating daily average...")
    # Subset the columns before grouping
    uhi_diff = uhi_diff[daily_var_lst + ['UHI_diff', 'lat', 'lon', 'date']]
    # Now perform the grouping and aggregation
    uhi_diff = uhi_diff.groupby(['lat', 'lon', 'date']).mean().reset_index()

X = uhi_diff[daily_var_lst]
y = uhi_diff['UHI_diff']
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Calculate and log feature slopes
print("Calculating feature slopes...")
feature_names = daily_var_lst
results_df = combine_slopes(local_hour_adjusted_df[daytime_mask], local_hour_adjusted_df[nighttime_mask], feature_names)
sorted_results_df = results_df.sort_values('Day_UHI_slope', ascending=False)

sorted_results_path = os.path.join(figure_dir, 'sorted_results_df.csv')
sorted_results_df.to_csv(sorted_results_path)
mlflow.log_artifact(sorted_results_path)
print(f"Saved sorted results to {sorted_results_path}")

# Get feature groups
feature_groups = [get_feature_group(feature) for feature in daily_var_lst]

# Train and evaluate the model
print("Training the model...")
model, selected_features, X_selected = train_and_evaluate(uhi_diff, daily_var_lst=daily_var_lst,
                                                          feature_groups=feature_groups,
                                                          model_name=f"{args.time_period}_model",
                                                          iterations=args.iterations, learning_rate=args.learning_rate,
                                                          depth=args.depth, feature_selection=args.feature_selection,
                                                          num_features=args.num_features)

# Log the trained model
print("Logging the trained model...")
mlflow.catboost.log_model(model, f"{args.time_period}_model")


# Function to get ordered feature importance
def get_ordered_feature_importance(model: CatBoostRegressor, pool, type='FeatureImportance'):
    """Calculates and returns ordered feature importance."""
    if type == 'FeatureImportance':
        feature_importances = model.get_feature_importance()
    else:
        feature_importances = model.get_feature_importance(pool, type=type)

    feature_names = pool.get_feature_names()

    # Ensure lengths match to avoid errors
    if len(feature_importances) != len(feature_names):
        raise ValueError("Feature importances and feature names lengths do not match")

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance_df = add_long_name(feature_importance_df, join_column='Feature')
    return feature_importance_df


# Create a pool for feature importance calculations
full_pool = Pool(X_selected, y)

# Calculate and plot feature importance
print("Creating feature importance plots...")
feature_importance = get_ordered_feature_importance(model, full_pool)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title(f'{args.time_period.capitalize()}time Feature Importance')
mlflow.log_figure(plt.gcf(), f'{args.time_period}time_feature_importance.png')
plt.clf()  # Clear the current figure

# Save and log feature importance data
feature_importance_path = os.path.join(figure_dir, 'feature_importance.feather')
feature_importance.to_feather(feature_importance_path)
mlflow.log_artifact(feature_importance_path)
print(f"Saved feature_importance data to {feature_importance_path}")


# SHAP calculations and plots
if args.shap_calculation:
    print("Starting SHAP calculations...")

    # Calculate SHAP values
    shap_values = model.get_feature_importance(full_pool, type='ShapValues')[:, :-1]

    # Save and log SHAP values, feature names, and X data
    shap_values_path = os.path.join(figure_dir, 'shap_values.npy')
    np.save(shap_values_path, shap_values)
    mlflow.log_artifact(shap_values_path)
    print(f"Saved SHAP values to {shap_values_path}")

    # Ensure that the length of shap_values matches the number of rows in uhi_diff
    if shap_values.shape[0] != uhi_diff.shape[0]:
        raise ValueError("The number of SHAP values does not match the number of rows in uhi_diff.")

    # Create a DataFrame from shap_values
    shap_values_df = pd.DataFrame(shap_values, columns=full_pool.get_feature_names())

    # Select the desired columns from uhi_diff
    additional_columns = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'KGMajorClass', 'UHI_diff']
    if not all(col in uhi_diff.columns for col in additional_columns):
        missing_cols = list(set(additional_columns) - set(uhi_diff.columns))
        raise ValueError(f"The following columns are missing in uhi_diff: {missing_cols}")

    uhi_diff_selected = uhi_diff[additional_columns].reset_index(drop=True)

    # Calculate estimation error
    y_pred = model.predict(X_selected)
    estimation_error = y_pred - uhi_diff_selected['UHI_diff']

    # Add estimation error to the DataFrame
    uhi_diff_selected['Estimation_Error'] = estimation_error

    # Concatenate the SHAP values with the additional columns
    combined_df = pd.concat([shap_values_df, uhi_diff_selected], axis=1)

    # Define the path to save the combined DataFrame
    combined_feather_path = os.path.join(figure_dir, 'shap_values_with_additional_columns.feather')

    # Save the combined DataFrame as a Feather file
    combined_df.reset_index(drop=True).to_feather(combined_feather_path)
    mlflow.log_artifact(combined_feather_path)
    print(f"Saved combined SHAP values and additional columns to {combined_feather_path}")

    feature_names_path = os.path.join(figure_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for feature in full_pool.get_feature_names():
            f.write(f"{feature}\n")
    mlflow.log_artifact(feature_names_path)
    print(f"Saved feature names to {feature_names_path}")

    X_path = os.path.join(figure_dir, 'X_data.feather')
    X_selected.to_feather(X_path)
    mlflow.log_artifact(X_path)
    print(f"Saved X data to {X_path}")

    # Create and log SHAP summary plot
    print("Creating SHAP summary plots...")
    shap.summary_plot(shap_values, X_selected, show=False)  # show=False to prevent displaying the plot
    plt.gcf().set_size_inches(15, 10)  # Adjust figure size
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_summary_plot.png')
    plt.clf()  # Clear the current figure


    # Create and log SHAP waterfall plot
    print("Creating SHAP waterfall plots...")
    feature_importances = model.get_feature_importance()
    expected_value = shap_values[0, -1]
    long_names = [get_long_name(f, df_daily_vars) for f in full_pool.get_feature_names()]
    shap.waterfall_plot(shap.Explanation(feature_importances, base_values=expected_value, feature_names=long_names),
                        show=False)
    plt.gcf().set_size_inches(15, 10)  # Adjust figure size
    plt.gcf().subplots_adjust(left=0.3)  # Increase left margin for y-axis labels
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_waterfall_plot.png')
    plt.clf()  # Clear the current figure

    # Function to calculate SHAP feature importance
    def get_shap_feature_importance(shap_values, feature_names, df_daily_vars):
        """Calculates SHAP feature importance."""
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

    # Calculate and save SHAP feature importance
    shap_feature_importance = get_shap_feature_importance(shap_values, full_pool.get_feature_names(), df_daily_vars)
    shap_feature_importance['Feature Group'] = shap_feature_importance['Feature'].apply(lambda x: get_feature_group(x))

    # Calculate SHAP feature importance by group
    shap_feature_importance_by_group = shap_feature_importance.groupby('Feature Group')['Importance'].sum().reset_index()
    total_importance = shap_feature_importance_by_group['Importance'].sum()
    shap_feature_importance_by_group['Percentage'] = (
                shap_feature_importance_by_group['Importance'] / total_importance) * 100
    shap_feature_importance_by_group.sort_values(by='Importance', ascending=False, inplace=True)

    shap_feature_importance_by_group_path = os.path.join(figure_dir,
                                                        f'{args.time_period}_shap_feature_importance_by_group.csv')
    shap_feature_importance_by_group.to_csv(shap_feature_importance_by_group_path, index=False)
    mlflow.log_artifact(shap_feature_importance_by_group_path)
    print(f"Saved SHAP feature importance by group data to {shap_feature_importance_by_group_path}")

    # Create and log SHAP summary plot by feature group
    feature_to_group = dict(zip(shap_feature_importance['Feature'], shap_feature_importance['Feature Group']))
    feature_groups = np.array([feature_to_group[feature] for feature in full_pool.get_feature_names()])
    unique_feature_groups = np.unique(feature_groups)
    color_map = plt.cm.get_cmap('tab20', len(unique_feature_groups))
    group_colors = color_map(np.arange(len(unique_feature_groups)))
    group_color_dict = dict(zip(unique_feature_groups, group_colors))
    feature_colors = [group_color_dict[group] for group in feature_groups]

    shap.summary_plot(shap_values, X_selected, plot_type='bar', feature_names=full_pool.get_feature_names(),
                      max_display=len(shap_feature_importance), show=False, color=feature_colors)
    plt.gcf().set_size_inches(15, 10)
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_summary_plot_by_group.png')
    plt.clf()

    # Create and log SHAP feature importance waterfall plot
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

    # Create and log percentage contribution horizontal bar plot
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

    # Save and log SHAP feature importance data
    shap_importance_path = os.path.join(figure_dir, f'{args.time_period}_shap_feature_importance.csv')
    shap_feature_importance.to_csv(shap_importance_path, index=False)
    mlflow.log_artifact(shap_importance_path)
    print(f"Saved SHAP feature importance data to {shap_importance_path}")

    # Function to create and log SHAP dependence plots
    def plot_dependence_grid(shap_values, X, feature_names, time_period, target_feature='U10'):
        """Creates and logs SHAP dependence plots."""
        feature_names = [f for f in feature_names if f != target_feature]

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

            plot_filename = f'{time_period}_{target_feature}_vs_{feature_name}.png'
            plot_path = os.path.join(full_nested_path, plot_filename)
            fig.savefig(plot_path)
            plt.close(fig)

            mlflow.log_artifact(plot_path, nested_path)

        print(f"Dependence plots for {target_feature} have been created and logged to MLflow.")

    # Create SHAP dependence plots for top features
    top_features = feature_importance['Feature'].tolist()
    print("Creating SHAP dependence plots...")
    for feature in top_features:
        print(f"Creating dependence plots for {feature}")
        plot_dependence_grid(shap_values, X_selected, feature_names=full_pool.get_feature_names(),
                             time_period=args.time_period, target_feature=feature)

# End MLflow run
print("Script execution completed.")
mlflow.end_run()
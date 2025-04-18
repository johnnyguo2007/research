import pandas as pd
import numpy as np
import xarray as xr
import os
import netCDF4
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.tensorflow
import mlflow.shap
from scipy.stats import linregress
import argparse
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define filter functions
def filter_by_year(df, year):
    return df[df['year'] == int(year)]

def filter_by_temperature_above_300(df, temperature):
    return df[df['temperature'] > float(temperature)]

def filter_by_hw_count(df, threshold):
    threshold = int(threshold)
    hw_counts = df[['lat', 'lon', 'year']].groupby(['lat', 'lon', 'year']).size().reset_index(name='count')
    locations_to_include = hw_counts[hw_counts['count'] <= threshold][['lat', 'lon']].drop_duplicates()
    df = df.merge(locations_to_include, on=['lat', 'lon'], how='left', indicator=True)
    return df[df['_merge'] == 'left_only'].drop(columns=['_merge'])

# Parse arguments
parser = argparse.ArgumentParser(description="Run UHI model for day or night data.")
parser.add_argument("--time_period", choices=["day", "night"], required=True, help="Specify whether to run for day or night data.")
parser.add_argument("--iterations", type=int, default=100, help="Number of epochs for the neural network model.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the neural network model.")
parser.add_argument("--filters", type=str, default="", help="Comma-separated list of filter function names and parameters to apply to the dataframe.")
parser.add_argument("--run_type", type=str, default="test", help="Beginning part of experiment name")
parser.add_argument("--exp_name_extra", type=str, default="", help="extra info that goes to the end of experiment name ")
parser.add_argument("--shap_calculation", action="store_true", help="If set, SHAP-related calculations and graphs will be performed.")

args = parser.parse_args()

# Set summary directory and experiment name
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
experiment_name = f'{args.run_type}_{args.time_period.capitalize()}_{args.exp_name_extra}'

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create the MLflow experiment
mlflow.set_experiment(experiment_name)
mlflow.start_run()

# Log command line arguments to MLflow
mlflow.log_param("time_period", args.time_period)
mlflow.log_param("iterations", args.iterations)
mlflow.log_param("learning_rate", args.learning_rate)

# Log the full command line
command_line = f"python {' '.join(sys.argv)}"
mlflow.log_param("command_line", command_line)

# Create directory for saving figures and artifacts
figure_dir = os.path.join(summary_dir, 'mlflow', experiment_name)
os.makedirs(figure_dir, exist_ok=True)

# Load data
merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables_with_location_ID_event_ID_and_sur.feather')
local_hour_adjusted_df = pd.read_feather(merged_feather_path)

# Apply filters if any
if args.filters:
    filter_function_pairs = args.filters.split(',')
    for filter_function_pair in filter_function_pairs:
        filter_function_name, filter_param = filter_function_pair.split('=')
        if filter_function_name in globals():
            local_hour_adjusted_df = globals()[filter_function_name](local_hour_adjusted_df, filter_param)
        else:
            print(f"Warning: Filter function {filter_function_name} not found. Skipping.")

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
    if (var_name.startswith('delta_')):
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

# Define linear slope function with error handling
def feature_linear_slope(df, feature_name, label, confidence_level=0.95):
    try:
        slope, _, _, p_value, stderr = linregress(df[feature_name], df[label])
        t_value = np.abs(np.percentile(np.random.standard_t(df[feature_name].shape[0] - 2, 100000), [(1-confidence_level)/2, 1-(1-confidence_level)/2]))[1]
        margin_of_error = t_value * stderr
        return slope, margin_of_error, p_value
    except Exception as e:
        print(f"Error calculating slope for {feature_name}: {e}")
        return None, None, None

def combine_slopes(daytime_df, nighttime_df, features, labels=['UHI', 'UHI_diff'], confidence_level=0.95):
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

feature_names = daily_var_lst
results_df = combine_slopes(local_hour_adjusted_df[daytime_mask], local_hour_adjusted_df[nighttime_mask], feature_names)
sorted_results_df = results_df.sort_values('Day_UHI_slope', ascending=False)

# Save and log the sorted results DataFrame
sorted_results_path = os.path.join(figure_dir, 'sorted_results_df.csv')
sorted_results_df.to_csv(sorted_results_path)
mlflow.log_artifact(sorted_results_path)

# Neural Network Model Definition
def create_model(input_shape, learning_rate):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

# Train and evaluate models
def train_and_evaluate(time_uhi_diff, daily_var_lst, model_name, epochs, learning_rate):
    X = time_uhi_diff[daily_var_lst]
    y = time_uhi_diff['UHI_diff']
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    model = create_model(X_train.shape[1], learning_rate)
    
    # Fit model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)
    
    # Log model parameters
    mlflow.log_param(f"{model_name}_epochs", epochs)
    mlflow.log_param(f"{model_name}_learning_rate", learning_rate)
    
    # Calculate and log metrics
    y_pred_train = model.predict(X_train).flatten()
    y_pred_val = model.predict(X_val).flatten()
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
    
    return model, scaler, history

model, scaler, history = train_and_evaluate(uhi_diff, daily_var_lst=daily_var_lst, model_name=f"{args.time_period}_model", epochs=args.iterations, learning_rate=args.learning_rate)

# Save the model and scaler
model_path = os.path.join(figure_dir, f"{args.time_period}_model")
scaler_path = os.path.join(figure_dir, f"{args.time_period}_scaler.pkl")
model.save(model_path)
mlflow.tensorflow.log_model(tf_saved_model_dir=model_path, tf_meta_graph_tags=None, tf_signature_def_key=None)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
mlflow.log_artifact(scaler_path)

# Plot training history
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Training and Validation Loss ({args.time_period.capitalize()}time)')
mlflow.log_figure(plt.gcf(), f'{args.time_period}_training_history.png')
plt.clf()

# SHAP-related calculations and plotting
if args.shap_calculation:
    import shap
    explainer = shap.KernelExplainer(model.predict, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    
    # Save SHAP values and feature names
    shap_values_path = os.path.join(figure_dir, 'shap_values.npy')
    np.save(shap_values_path, shap_values)
    mlflow.log_artifact(shap_values_path)

    feature_names_path = os.path.join(figure_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for feature in daily_var_lst:
            f.write(f"{feature}\n")
    mlflow.log_artifact(feature_names_path)

    # Log X data
    X_path = os.path.join(figure_dir, 'X_data.feather')
    X.to_feather(X_path)
    mlflow.log_artifact(X_path)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_scaled, feature_names=daily_var_lst, show=False)
    plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
    mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_summary_plot.png')
    plt.clf()

    # SHAP dependence plots
    top_features = feature_names.head(3).tolist()

    for feature in top_features:
        shap.dependence_plot(feature, shap_values, X_scaled, feature_names=daily_var_lst, show=False)
        plt.gcf().set_size_inches(15, 10)  # Adjust the figure size
        mlflow.log_figure(plt.gcf(), f'{args.time_period}_shap_dependence_plot_{feature}.png')
        plt.clf()

print("Done")
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

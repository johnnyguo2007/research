#!/bin/bash

# Set the threshold for heat wave count
HW_COUNT_THRESHOLD=60

# Set other parameters
ITERATIONS=100000
LEARNING_RATE=0.01
DEPTH=10
BASE_RUN_TYPE="Hourly_kg_model"


# Function to run the experiment
run_experiment() {
    local time_period=$1
    local hw_percentile=$2
    local merged_file=$3

    # Construct the column names using the time_period variable
    feature_column="${time_period}_selected"
    delta_column="${time_period}_delta_selected"
    hw_nohw_diff_column="${time_period}_Hw_no_hw_selected"
    double_diff_column="${time_period}_DD_selected"

    # Construct the run type and experiment name
    run_type="${BASE_RUN_TYPE}"
    exp_name_extra="HW${hw_percentile}_no_filter"

# --filters "filter_by_year,1985" \

    # python /home/jguo/research/hw_global/ultimate/mlflow_feature_selection.py \
    python /home/jguo/research/hw_global/ultimate/hourly_ke_model.py \
        --summary_dir /Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary \
        --merged_feather_file $merged_file \
        --time_period "${time_period}" \
        --iterations $ITERATIONS \
        --learning_rate $LEARNING_RATE \
        --depth $DEPTH \
        --run_type "${run_type}" \
        --exp_name_extra "${exp_name_extra}" \
        --shap_calculation \
        --feature_column "${feature_column}" \
        --delta_column "${delta_column}" \
        --hw_nohw_diff_column "${hw_nohw_diff_column}" \
        --double_diff_column "${double_diff_column}" \
        --delta_mode "include" \
        --exclude_features "local_hour"
}

# Run experiments for HW95 and HW90, for day and night
for hw_percentile in 98; do
    merged_file="updated_local_hour_adjusted_variables_HW${hw_percentile}.feather"    
    # for time_period in "day" "night"; do
    for time_period in "hourly"; do
        echo "Running experiment for ${time_period}, HW${hw_percentile}"
        run_experiment "${time_period}" $hw_percentile $merged_file
    done
done

# run_experiment "day" 95 "local_hour_adjusted_variables_HW95.feather"
echo "All experiments completed."

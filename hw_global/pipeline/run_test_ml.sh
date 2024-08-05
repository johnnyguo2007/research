#!/bin/bash

# Set the threshold for heat wave count
HW_COUNT_THRESHOLD=60

# Set other parameters
ITERATIONS=1000
LEARNING_RATE=0.01
DEPTH=6
BASE_RUN_TYPE="TEST_DD"  # Base part of the run type

# Function to run the experiment
run_experiment() {
    local time_period=$1
    local hw_percentile=$2
    local merged_file=$3

    # Construct the run type and experiment name
    run_type="${BASE_RUN_TYPE}"
    exp_name_extra="HW${hw_percentile}_no_filter"

        # python /home/jguo/research/hw_global/final/mlflow_feature_selection.py \
        # --summary_dir /Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary \
        # --merged_feather_file $merged_file \
        # --time_period $time_period \
        # --iterations $ITERATIONS \
        # --learning_rate $LEARNING_RATE \
        # --depth $DEPTH \
        # --run_type "${run_type}" \
        # --exp_name_extra "${exp_name_extra}" \
        # --shap_calculation \
        # --filters "filter_by_KGMajorClass,${kg_major_class}" \
        # --feature_column "X_ML_Selected" \
        # --delta_column "X_ML_Delta_Selected" \
        # --delta_mode "include"
    
    python /home/jguo/research/hw_global/final/mlflow_feature_selection.py \
        --summary_dir /Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary \
        --merged_feather_file $merged_file \
        --time_period $time_period \
        --iterations $ITERATIONS \
        --learning_rate $LEARNING_RATE \
        --depth $DEPTH \
        --run_type "${run_type}" \
        --exp_name_extra "${exp_name_extra}" \
        --filters "filter_by_year,1985" \
        --feature_column "X_ML_Selected" \
        --delta_column "X_ML_Delta_Selected" \
        --delta_mode "include"
}

# Run experiments for different KGMajorClass categories
# kg_major_classes=("Arid" "Cold" "Temperate" "Tropical")
kg_major_classes=( "Temperate")

# Run experiments for HW95 and HW90, for day and night
# for hw_percentile in 98 99; do
for hw_percentile in 98; do
    merged_file="updated_local_hour_adjusted_variables_HW${hw_percentile}.feather"
    
    # for time_period in "day" "night"; do
    for time_period in "day" ; do

        echo "Running experiment for $time_period, HW${hw_percentile}"
        run_experiment $time_period $hw_percentile $merged_file

    done
done

echo "All experiments completed."
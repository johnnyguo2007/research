#!/bin/bash

# Set the threshold for heat wave count
HW_COUNT_THRESHOLD=60

# Set other parameters
ITERATIONS=100000
LEARNING_RATE=0.01
DEPTH=10
BASE_RUN_TYPE="SOIL_KG"  # Base part of the run type

# Function to run the experiment
run_experiment() {
    local time_period=$1
    local hw_percentile=$2
    local merged_file=$3
    local kg_major_class=$4

    # Construct the run type and experiment name
    run_type="${BASE_RUN_TYPE}_${kg_major_class}"
    exp_name_extra="Qstor_Delta_HW${hw_percentile}_filter"

    python /home/jguo/research/hw_global/final/mlflow_feature_selection.py \
        --summary_dir /Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary \
        --merged_feather_file $merged_file \
        --time_period $time_period \
        --iterations $ITERATIONS \
        --learning_rate $LEARNING_RATE \
        --depth $DEPTH \
        --run_type "${run_type}" \
        --exp_name_extra "${exp_name_extra}" \
        --shap_calculation \
        --filters "filter_by_KGMajorClass,${kg_major_class}" \
        --feature_column "X_ML_Selected" \
        --delta_column "X_ML_Delta_Selected" \
        --delta_mode "include"
}

# Run experiments for different KGMajorClass categories
kg_major_classes=("Arid" "Cold" "Temperate" "Tropical")

# Run experiments for HW95 and HW90, for day and night
for hw_percentile in 98 99; do
    merged_file="local_hour_adjusted_variables_HW${hw_percentile}.feather"
    
    for time_period in "day" "night"; do
        for kg_major_class in "${kg_major_classes[@]}"; do
            echo "Running experiment for $time_period, HW${hw_percentile}, KGMajorClass: ${kg_major_class}"
            run_experiment $time_period $hw_percentile $merged_file "$kg_major_class"
        done
    done
done

echo "All experiments completed."
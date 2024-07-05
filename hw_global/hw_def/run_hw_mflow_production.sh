#!/bin/bash

# Set the threshold for heat wave count
HW_COUNT_THRESHOLD=60

# Set other parameters
ITERATIONS=100000
LEARNING_RATE=0.01
DEPTH=10
RUN_TYPE="production"

# Function to run the experiment
run_experiment() {
    local time_period=$1
    local hw_percentile=$2
    local merged_file=$3

    python /home/jguo/research/hw_global/final/mlflow_production_run.py \
        --summary_dir /Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/hw95_summary \
        --merged_feather_file $merged_file \
        --time_period $time_period \
        --iterations $ITERATIONS \
        --learning_rate $LEARNING_RATE \
        --depth $DEPTH \
        --run_type $RUN_TYPE \
        --exp_name_extra "HW${hw_percentile}_no_filter" \
        --shap_calculation
}

# Run experiments for HW95 and HW90, for day and night
for hw_percentile in 95 90; do
    merged_file="local_hour_adjusted_variables_HW${hw_percentile}.feather"
    
    for time_period in "day" "night"; do
        echo "Running experiment for $time_period, HW${hw_percentile}"
        run_experiment $time_period $hw_percentile $merged_file
    done
done

echo "All experiments completed."
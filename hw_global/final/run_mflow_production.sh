#!/bin/bash

# # Run the first command
# python mlflow_production_run.py --time_period day --iterations 100000 --learning_rate 0.01 --depth 8 --filters filter_by_hw_count=60 --run_type Production_layer8 --exp_name_extra hw_60

# # Run the second command
# python mlflow_production_run.py --time_period day --iterations 100000 --learning_rate 0.01 --depth 9 --filters filter_by_hw_count=60 --run_type Production_layer9 --exp_name_extra hw_60


# Set the threshold for UHI_diff categories
THRESHOLD=0.2

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
    local category=$2
    
    python mlflow_production_run.py \
        --time_period $time_period \
        --iterations $ITERATIONS \
        --learning_rate $LEARNING_RATE \
        --depth $DEPTH \
        --run_type $RUN_TYPE \
        --exp_name_extra "${category}_UHI_HW${HW_COUNT_THRESHOLD}" \
        --filters "filter_by_hw_count=${HW_COUNT_THRESHOLD};filter_by_uhi_diff_category=${THRESHOLD},${category}" 
        
}

# Run experiments for day and night, and for each UHI_diff category
for time_period in "day" "night"; do
    for category in "Positive" "Negative"; do
        echo "Running experiment for $time_period, $category UHI, HW count <= ${HW_COUNT_THRESHOLD}"
        run_experiment $time_period $category
    done
done

echo "All experiments completed."
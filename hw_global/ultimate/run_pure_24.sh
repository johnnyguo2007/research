#!/bin/bash

# Set parameters
ITERATIONS=100000
LEARNING_RATE=0.01
DEPTH=10
BASE_RUN_TYPE="pure_soil_24"
HW_PERCENTILE=98
MERGED_FILE="updated_local_hour_adjusted_variables_HW${HW_PERCENTILE}.feather"
FEATURE_COLUMN="hourly_selected"
DELTA_COLUMN="hourly_delta_selected"
HW_NOHW_DIFF_COLUMN="hourly_Hw_no_hw_selected"
DOUBLE_DIFF_COLUMN="hourly_DD_selected"

# Function to run the experiment
run_experiment() {
    local local_hour=$1

    # Construct the run type and experiment name
    run_type="${BASE_RUN_TYPE}"
    exp_name_extra="HW${HW_PERCENTILE}_Hour${local_hour}"

    # Run the Python script
    python /home/jguo/research/hw_global/ultimate/hourly_kg_model.py \
        --summary_dir /Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary \
        --merged_feather_file $MERGED_FILE \
        --time_period "hourly" \
        --iterations $ITERATIONS \
        --learning_rate $LEARNING_RATE \
        --depth $DEPTH \
        --run_type "${run_type}" \
        --exp_name_extra "${exp_name_extra}" \
        --shap_calculation \
        --feature_column "${FEATURE_COLUMN}" \
        --delta_column "${DELTA_COLUMN}" \
        --hw_nohw_diff_column "${HW_NOHW_DIFF_COLUMN}" \
        --double_diff_column "${DOUBLE_DIFF_COLUMN}" \
        --delta_mode "include" \
        --exclude_features "local_hour" \
        --filters "filter_by_NO_KGMajorClass,Polar;filter_by_local_hour,${local_hour}"
}

# Run experiments for hours 6, 9, and 18
for local_hour in 6 9 18; do
    echo "Running experiment for Local Hour: ${local_hour}"
    run_experiment "$local_hour"
done

# Run experiments for the remaining hours
for local_hour in $(seq 0 23); do
    if [[ ! " 6 9 18 " =~ " ${local_hour} " ]]; then
        echo "Running experiment for Local Hour: ${local_hour}"
        run_experiment "$local_hour"
    fi
done

echo "All experiments completed." 
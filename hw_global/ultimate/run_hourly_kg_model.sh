#!/bin/bash

# Set parameters
ITERATIONS=100000
LEARNING_RATE=0.01
DEPTH=10
BASE_RUN_TYPE="Single_Hour_NO_FSA"
HW_PERCENTILE=98
MERGED_FILE="updated_local_hour_adjusted_variables_HW${HW_PERCENTILE}.feather"
FEATURE_COLUMN="hourly_selected"
DELTA_COLUMN="hourly_delta_selected"
HW_NOHW_DIFF_COLUMN="hourly_Hw_no_hw_selected"
DOUBLE_DIFF_COLUMN="hourly_DD_selected"

# Function to run the experiment
run_experiment() {
    local kg_major_class=$1
    local local_hour=$2

    # Construct the run type and experiment name
    run_type="${BASE_RUN_TYPE}_${kg_major_class}"
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
        --filters "filter_by_KGMajorClass,${kg_major_class};filter_by_local_hour,${local_hour}" \
        --post_process
}

# Define KGMajorClass categories
# kg_major_classes=("Arid" "Cold" "Temperate" "Tropical")
kg_major_classes=("Arid" "Cold")

# Loop through KGMajorClass categories and local hours
for kg_major_class in "${kg_major_classes[@]}"; do
    for local_hour in $(seq 8 10); do
        echo "Running experiment for KGMajorClass: ${kg_major_class}, Local Hour: ${local_hour}"
        run_experiment "$kg_major_class" "$local_hour"
    done
done

echo "All experiments completed." 
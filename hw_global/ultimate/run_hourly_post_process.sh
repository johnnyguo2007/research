#!/bin/bash

# Define the MLflow tracking URI
TRACKING_URI="http://192.168.4.85:8080"

# Define the base output path for generated plots
OUTPUT_PATH="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary"

# Define an array of patterns and their corresponding generate types
declare -A patterns
# patterns["^Single_Hour_NO_FSA_(Arid|Cold)_HW98_Hour(8|9|10)$"]="all"
# patterns["^Single_Hour_NO_FSA_(Temperate)_HW98_Hour(8|9|10)$"]="summary group_summary"
# patterns["^Single_Hour_NO_FSA_(Tropical)_HW98_Hour(8|9|10)$"]="dependency"

# patterns["^Single_Hour_NO_FSA_(Arid|Cold)_HW98_Hour(8|9|10)$"]="marginal_effects"
# patterns["^Single_Hour_NO_FSA_Arid_Hourly_HW98_Hour8$"]="dependency"

# # Add a pattern to generate summary, waterfall, and dependency plots for pure_24 experiments with hours 0-10
# patterns["^pure_24_Hourly_HW98_Hour(10|[0-9])$"]="summary waterfall dependency"

# Add a pattern to generate summary, waterfall, and dependency plots for pure_24 experiments with hours 11-23
# patterns["^pure_24_Hourly_HW98_Hour(1[1-9]|2[0-3])$"]="summary waterfall dependency"

# patterns["^pure_soil_24_Hourly_HW98_Hour([0-9]|1[0-9]|2[0-3])$"]="combine_shap"
# patterns["^pure_soil_24_Hourly_HW98_Hour([0-9]|1[0-9]|2[0-3])$"]="summary waterfall dependency"

# patterns["^pure2_soil_ddq2m_24_Hourly_HW98_Hour9$"]="summary waterfall dependency"
# patterns["^Final_noFSA_Hourly_HW98_Hour[0-9]{1,2}$"]="summary waterfall dependency"
# patterns["^Final_noFSA_Hourly_HW98_Hour[0-9]+$"]="combine_shap"

# patterns["^mixed_FSA_Hourly_HW98_Hour[0-9]{1,2}$"]="summary"
# patterns["^Final3_NO_LE_Hourly_HW98_Hour(10|6|12)$"]="summary waterfall dependency"
# patterns["^Final3_NO_LE_Hourly_HW98_Hour[0-9]+$"]="summary waterfall dependency"
# patterns["^Final3_NO_LE_Hourly_HW98_Hour[0-9]+$"]="combine_shap"

# patterns["^Selection_Final_Day_HW98_no_filter$"]="summary waterfall dependency"
# patterns["^Selection_Final_Night_HW98_no_filter$"]="summary waterfall dependency"
patterns["^Selection_Final_(Day|Night)_HW98_no_filter$"]="combine_shap"

# Loop through the patterns and generate types
for pattern in "${!patterns[@]}"; do
    generate_types="${patterns[$pattern]}"
    echo "Processing experiments matching pattern: $pattern"
    echo "Generating plots of types: $generate_types"

    # Run the Python script with the current pattern, generate types, and model subduration
    python /home/jguo/research/hw_global/ultimate/hourly_post_process.py \
        --pattern "$pattern" \
        --generate_types "$generate_types" \
        --tracking_uri "$TRACKING_URI" \
        --output_path "$OUTPUT_PATH" \
        --model_subdur "night_model"
done

echo "All experiments processed." 
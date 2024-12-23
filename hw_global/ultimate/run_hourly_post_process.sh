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
patterns["^Single_Hour_NO_FSA_Arid_HW98_Hour8$"]="marginal_effects"

# Loop through the patterns and generate types
for pattern in "${!patterns[@]}"; do
    generate_types="${patterns[$pattern]}"
    echo "Processing experiments matching pattern: $pattern"
    echo "Generating plots of types: $generate_types"

    # Run the Python script with the current pattern and generate types
    python /home/jguo/research/hw_global/ultimate/hourly_post_process.py \
        --pattern "$pattern" \
        --generate_types "$generate_types" \
        --tracking_uri "$TRACKING_URI" \
        --output_path "$OUTPUT_PATH"
done

echo "All experiments processed." 
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_data(feather_path):
    """Loads data from a Feather file."""
    logging.info(f"Loading data from {feather_path}")
    df = pd.read_feather(feather_path)
    logging.info(f"Loaded DataFrame with shape {df.shape}")
    return df

def add_feature_group(df):
    """Adds a 'Feature Group' column to the dataframe."""
    logging.info("Adding Feature Group column to DataFrame")
    df['Feature Group'] = df['Feature'].apply(get_feature_group)
    logging.info(f"Found {len(df['Feature Group'].unique())} unique feature groups")
    return df

def calculate_percentage_contribution(df):
    """Calculates the percentage contribution of each feature group per hour."""
    logging.info("Calculating percentage contributions by feature group and hour")
    grouped = df.groupby(['hour', 'Feature Group']).size().unstack(fill_value=0)
    logging.info(f"Created grouped DataFrame with shape {grouped.shape}")
    
    percentage = grouped.divide(grouped.sum(axis=1), axis=0) * 100
    logging.info("Calculated percentage contributions")
    return percentage

def plot_stacked_bar(plot_df, title, output_path=None, total_series=None):
    """Plots a stacked bar chart of contributions per hour with an optional total value curve."""
    logging.info(f"Creating stacked bar plot: {title}")
    
    ax = plot_df.plot(
        kind='bar', 
        stacked=True, 
        figsize=(15, 8), 
        colormap='tab20',
        width=0.8,
        ax=None
    )
    
    plt.xlabel('Hour of Day')
    ylabel = 'Percentage Contribution' if (plot_df.sum(axis=1).round(2) == 100.00).all() else 'Absolute Contribution'
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='Feature Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if total_series is not None:
        logging.info("Adding total value curve to plot")
        ax2 = ax.twinx()
        ax2.plot(plot_df.index, total_series, color='black', marker='o', linestyle='-', label='Total Value')
        ax2.set_ylabel('Total Value')
        ax2.legend(loc='upper right')
    else:
        logging.info("Adding 100% reference line to plot")
        ax2 = ax.twinx()
        ax2.plot(plot_df.index, [100] * len(plot_df.index), color='black', linestyle='--', label='Total 100%')
        ax2.set_ylabel('Total Percentage')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        logging.info(f"Saving plot to {output_path}")
        plt.savefig(output_path)
        plt.close()
        logging.info("Plot saved and figure closed")
    else:
        logging.info("Displaying plot")
        plt.show()

def main():
    logging.info("Starting main execution")
    
    import argparse

    parser = argparse.ArgumentParser(
        description="Report and plot SHAP value contributions by feature group and hour."
    )
    
    # Add command-line arguments
    parser.add_argument(
        '--local-hour-adjusted-df-path',
        type=str,
        default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather',
        help='Path to the local_hour_adjusted_df file.'
    )
    parser.add_argument(
        '--shap-values-feather-path',
        type=str,
        default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/Hourly_kg_model_Hourly_HW98_no_filter/shap_values_with_additional_columns.feather',
        help='Path to the shap_values_with_additional_columns.feather file.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./',
        help='Directory to save the output files.'
    )
    
    args = parser.parse_args()
    logging.info("Parsed command line arguments")
    
    # Extract arguments
    shap_values_feather_path = args.shap_values_feather_path
    local_hour_adjusted_df_path = args.local_hour_adjusted_df_path
    output_dir = args.output_dir
    output_feature_group = os.path.join(output_dir, 'shap_feature_group.feather') 
    output_pivot = os.path.join(output_dir, 'shap_pivot.feather')

    logging.info(f"Using output directory: {output_dir}")
    
    # Report SHAP contribution
    logging.info("Starting SHAP contribution analysis")
    df_feature_group = report_shap_contribution_from_feather(
        local_hour_adjusted_df_path, 
        shap_values_feather_path, 
        output_dir, 
        output_feature_group, 
        output_pivot
    )
    logging.info("SHAP contribution analysis completed")

    # Define the types of plots to generate
    plot_types = ['percentage', 'absolute']
    logging.info(f"Generating plots for types: {plot_types}")

    # Generate plots for each KGMajorClass and each plot type
    kg_major_classes = df_feature_group['KGMajorClass'].unique()
    logging.info(f"Found {len(kg_major_classes)} unique KGMajorClasses: {kg_major_classes}")

    for kg_class in kg_major_classes:
        logging.info(f"Processing KGMajorClass: {kg_class}")
        df_subset = df_feature_group[df_feature_group['KGMajorClass'] == kg_class]
        
        # Group by local_hour and Feature Group
        grouped = df_subset.groupby(['local_hour', 'Feature Group'])['Value'].sum().reset_index()
        logging.info(f"Created grouped DataFrame for {kg_class} with shape {grouped.shape}")
        
        # Pivot to have Feature Groups as columns
        pivot_df = grouped.pivot(index='local_hour', columns='Feature Group', values='Value').fillna(0)
        logging.info(f"Created pivot table with shape {pivot_df.shape}")
        
        for plot_type in plot_types:
            logging.info(f"Generating {plot_type} plot for {kg_class}")
            if plot_type == 'percentage':
                plot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
                total_series = None
            else:
                plot_df = pivot_df
                total_series = plot_df.sum(axis=1)
    
            title = f'ΔUHI Contribution by Hour - {kg_class}' + (f' ({plot_type.capitalize()})' if plot_type == 'percentage' else f' ({plot_type.capitalize()})')
            output_filename = f'feature_group_contribution_by_hour_{kg_class}_{plot_type}.png'
            output_path = os.path.join(output_dir, output_filename)
    
            plot_stacked_bar(plot_df, title, output_path=output_path, total_series=total_series)
            logging.info(f"Completed {plot_type} plot for {kg_class}")

    # Generate total plots
    logging.info("Generating total plots across all KGMajorClasses")
    total_grouped = df_feature_group.groupby(['local_hour', 'Feature Group'])['Value'].sum().reset_index()
    total_pivot_df = total_grouped.pivot(index='local_hour', columns='Feature Group', values='Value').fillna(0)
    
    for plot_type in plot_types:
        logging.info(f"Generating total {plot_type} plot")
        if plot_type == 'percentage':
            total_plot_df = total_pivot_df.div(total_pivot_df.sum(axis=1), axis=0) * 100
            total_series = None
        else:
            total_plot_df = total_pivot_df
            total_series = total_plot_df.sum(axis=1)
        
        total_title = f'ΔUHI Contribution by Hour - Global' + (f' ({plot_type.capitalize()})' if plot_type == 'percentage' else f' ({plot_type.capitalize()})')
        total_output_filename = f'feature_group_contribution_by_hour_total_{plot_type}.png'
        total_output_path = os.path.join(output_dir, total_output_filename)

        plot_stacked_bar(total_plot_df, total_title, output_path=total_output_path, total_series=total_series)
        logging.info(f"Completed total {plot_type} plot")

    logging.info("Script execution completed successfully")

if __name__ == "__main__":
    main() 
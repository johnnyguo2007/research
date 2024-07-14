import os
import pandas as pd
import argparse

def print_dataframe_info(df, filename):
    print(f"\nInformation for {filename}:")
    print(f"Shape: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())


def print_group_summary(df, group_column):
    total_rows = len(df)
    group_summary = df.groupby(group_column).size().reset_index(name='count')
    group_summary['percentage'] = group_summary['count'] / total_rows * 100
    
    print(f"\nSummary grouped by {group_column}:")
    print(group_summary.to_string(index=False))
    print(f"\nTotal rows: {total_rows}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Merge two feather files based on location_ID.')
    parser.add_argument('--file1', default='local_hour_adjusted_variables_HW99.feather',
                        help='Path to the first feather file (default: %(default)s)')
    parser.add_argument('--file2', default='kg_category_location_ID.feather',
                        help='Path to the second feather file (default: %(default)s)')
    args = parser.parse_args()

    # Set the directory path
    summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'

    # Define file paths
    file1_path = os.path.join(summary_dir, args.file1)
    file2_path = os.path.join(summary_dir, args.file2)

    # Read the feather files
    df1 = pd.read_feather(file1_path)
    df2 = pd.read_feather(file2_path)

    # Merge the dataframes
    merged_df = df1.merge(df2[['location_ID', 'KGClass', 'KGMajorClass']], on='location_ID', how='left')

    # Backup file1
    backup_path = file1_path + '.bk'
    os.rename(file1_path, backup_path)

    # Save the merged dataframe to file1
    merged_df.to_feather(file1_path)

    print(f"Files have been successfully merged and saved. Original file1 backed up as {backup_path}")

    # Print information about the new file1
    print_dataframe_info(merged_df, args.file1)

    # Print group summaries
    print_group_summary(merged_df, 'KGClass')
    print_group_summary(merged_df, 'KGMajorClass')

if __name__ == "__main__":
    main()
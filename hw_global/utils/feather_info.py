import sys
import pyarrow as pa
import pyarrow.feather as feather
from tabulate import tabulate
import json
import argparse
import math
import numpy as np
import pandas as pd

def format_number(num):
    return f"{num:,}"

def format_float(x):
    if isinstance(x, (float, np.float64, np.float32)):
        return f"{x:.4f}"
    return x

def dump_feather_metadata(file_name, show_data=False):
    try:
        # Read the Feather file metadata
        with pa.memory_map(file_name, 'r') as source:
            # Read the table metadata
            table = feather.read_table(source, memory_map=True)

            # Print basic file info
            print(f"Feather File Info for {file_name}:")
            print("=" * 40)
            print(tabulate([
                ["Number of rows", format_number(table.num_rows)],
                ["Number of columns", format_number(table.num_columns)]
            ], tablefmt="plain"))

            # Prepare column information
            column_info = []
            for i, column in enumerate(table.columns):
                column_info.append([
                    i,
                    table.field(i).name,
                    str(column.type),
                    str(table.field(i).nullable),
                ])

            # Print column information
            print("\nColumn Information:")
            print("=" * 40)
            print(tabulate(column_info, headers=["Index", "Name", "Type", "Nullable"], tablefmt="plain"))

            # Show data preview if requested
            if show_data:
                print("\nData Preview:")
                print("=" * 40)

                num_rows = table.num_rows
                num_cols = table.num_columns

                # Define the number of rows to preview
                preview_rows = 20

                # Function to display data in groups of 12 columns
                def display_data_groups(df_preview, row_label):
                    num_cols_preview = df_preview.shape[1]
                    num_groups = math.ceil(num_cols_preview / 12)

                    for i in range(num_groups):
                        start_col = i * 12
                        end_col = min((i + 1) * 12, num_cols_preview)

                        print(f"\n{row_label} (Columns {start_col+1}-{end_col}):")
                        print(tabulate(df_preview.iloc[:, start_col:end_col],
                                       headers='keys',
                                       tablefmt='pretty',
                                       showindex=False))

                # Display top rows
                if num_rows > 0:
                    head_rows = min(preview_rows, num_rows)
                    head_table = table.slice(0, head_rows)
                    df_head = head_table.to_pandas()
                    df_head = df_head.applymap(format_float)
                    display_data_groups(df_head, f"Top {head_rows} rows")

                # Display bottom rows
                if num_rows > 0:
                    tail_rows = min(preview_rows, num_rows)
                    tail_start_index = max(0, num_rows - tail_rows)
                    tail_table = table.slice(tail_start_index, tail_rows)
                    df_tail = tail_table.to_pandas()
                    df_tail = df_tail.applymap(format_float)
                    display_data_groups(df_tail, f"Bottom {tail_rows} rows")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump Feather file metadata and optionally show data preview.")
    parser.add_argument("file_name", help="Path to the Feather file")
    parser.add_argument("-l", "--show-data", action="store_true", help="Show top 20 and bottom 20 rows of data")
    args = parser.parse_args()

    dump_feather_metadata(args.file_name, args.show_data)
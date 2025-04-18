import pandas as pd
import openpyxl # Ensure openpyxl is imported

# Define the path to the Excel file
file_path = r"C:\Users\Dell0920\OneDrive\Johnny\research\paper_for_publication\Data\combined_selection_final_feature_group_contribution_by_hour_global_group_data.xlsx"

# Read the specified sheet ('DayNight'), first 11 columns, and first 25 data rows, using the openpyxl engine
df = pd.read_excel(file_path, sheet_name='DayNight', usecols=range(11), nrows=25, engine='openpyxl')

# Print the first 5 rows of the read data (optional, for verification)
print("Data read from 'DayNight' sheet (first 5 rows):")
print(df.head())

# Write the DataFrame to a new sheet named 'mixed' in the same Excel file
# Use ExcelWriter in append mode with if_sheet_exists='overlay' to write data
# starting at cell A1 (startrow=0, startcol=0) and only overwrite that region.
try:
    # mode='a' requires openpyxl
    # if_sheet_exists='overlay' writes starting at startrow, startcol
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name='mixed', index=False, header=True, startrow=0, startcol=0)
    print(f"Successfully wrote data to 'mixed' sheet in {file_path}")
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Example of further processing (optional)
# print(df.describe())

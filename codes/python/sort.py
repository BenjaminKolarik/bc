import pandas as pd
from openpyxl import load_workbook
from zipfile import BadZipFile
import os

# Function to sort sheets alphabetically
def sort_sheets_alphabetically(file_path, output_dir):
    try:
        # Validate the file format using openpyxl
        load_workbook(file_path)  # This will raise an error if the file is not a valid Excel file

        # Load the Excel file
        excel = pd.ExcelFile(file_path, engine="openpyxl")  # Specify the engine explicitly

        # Sort sheet names alphabetically
        sorted_sheets = sorted(excel.sheet_names)

        # Create output file path
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)

        # Create a new Excel writer
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name in sorted_sheets:
                # Read the sheet and write it back in sorted order
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"Sheets in {file_path} sorted alphabetically and saved to {output_path}.")
    except BadZipFile:
        print(f"Error: {file_path} is not a valid Excel file.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# List of file paths for the Excel files
file_paths = [
    '../../output/execution_times/final/Python/execution_times_python_extra_large.xlsx',
    '../../output/execution_times/final/Python/execution_times_python_large.xlsx',
    '../../output/execution_times/final/Python/execution_times_python_medium.xlsx',
    '../../output/execution_times/final/Python/execution_times_python_small.xlsx',
    '../../output/execution_times/final/R/execution_times_R_extra_large.xlsx',
    '../../output/execution_times/final/R/execution_times_R_large.xlsx',
    '../../output/execution_times/final/R/execution_times_R_medium.xlsx',
    '../../output/execution_times/final/R/execution_times_R_small.xlsx',
]

# Output directory for sorted files
output_directory = '../../output/execution_times/sorted'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each file
for file_path in file_paths:
    sort_sheets_alphabetically(file_path, output_directory)
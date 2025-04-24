import pandas as pd
from openpyxl import load_workbook
from zipfile import BadZipFile
import os

def sort_sheets_alphabetically(file_path, output_dir):
    try:
        load_workbook(file_path)

        excel = pd.ExcelFile(file_path, engine="openpyxl")  # Specify the engine explicitly

        sorted_sheets = sorted(excel.sheet_names)

        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name in sorted_sheets:
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"Sheets in {file_path} sorted alphabetically and saved to {output_path}.")
    except BadZipFile:
        print(f"Error: {file_path} is not a valid Excel file.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

file_paths = [
    '../../output/execution_times/sorted/Python/execution_times_python_extra_large.xlsx',
    '../../output/execution_times/sorted/Python/execution_times_python_large.xlsx',
    '../../output/execution_times/sorted/Python/execution_times_python_medium.xlsx',
    '../../output/execution_times/sorted/Python/execution_times_python_small.xlsx',
    '../../output/execution_times/sorted/R/execution_times_R_extra_large.xlsx',
    '../../output/execution_times/sorted/R/execution_times_R_large.xlsx',
    '../../output/execution_times/sorted/R/execution_times_R_medium.xlsx',
    '../../output/execution_times/sorted/R/execution_times_R_small.xlsx',
]

output_directory = '../../output/execution_times/sorted'

os.makedirs(output_directory, exist_ok=True)

for file_path in file_paths:
    sort_sheets_alphabetically(file_path, output_directory)

import pandas as pd

def merge_excels():
    # List of file paths for the Excel files
    file_paths = [
        '../../output/execution_times/h/chiara/execution_times_Python_extra_large_chiara.xlsx',
        '../../output/execution_times/h/martin/execution_times_Python_extra_large_martin.xlsx',
        '../../output/execution_times/h/lenovo/execution_times_Python_extra_large_lenovo.xlsx',
        '../../output/execution_times/h/hp/execution_times_Python_extra_large_hp.xlsx'
    ]

    # Output file path
    output_file = '../../output/execution_times/merged/Python/execution_times_Python_extra_large.xlsx'
    # Load all Excel files
    excels = [pd.ExcelFile(file) for file in file_paths]

    # Get the common sheet names across all files
    common_sheets = set(excels[0].sheet_names)
    for excel in excels[1:]:
        common_sheets.intersection_update(excel.sheet_names)

    # Dictionary to store merged data for all sheets
    merged_sheets = {}

    for sheet_name in common_sheets:
        # Read and concatenate the sheet from all files
        dfs = [pd.read_excel(file, sheet_name=sheet_name) for file in file_paths]
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates()

        # Store the merged DataFrame
        merged_sheets[sheet_name] = merged

    # Save all merged sheets to a new Excel file
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, data in merged_sheets.items():
            data.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"Merged sheets saved to {output_file}")



merge_excels()

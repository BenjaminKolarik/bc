import pandas as pd

def merge_excels():
    file_paths = [
        '../../output/execution_times/h/chiara/execution_times_Python_extra_large_chiara.xlsx',
        '../../output/execution_times/h/martin/execution_times_Python_extra_large_martin.xlsx',
        '../../output/execution_times/h/lenovo/execution_times_Python_extra_large_lenovo.xlsx',
        '../../output/execution_times/h/hp/execution_times_Python_extra_large_hp.xlsx'
    ]

    output_file = '../../output/execution_times/merged/Python/execution_times_Python_extra_large.xlsx'
    # Load all Excel files
    excels = [pd.ExcelFile(file) for file in file_paths]

    common_sheets = set(excels[0].sheet_names)
    for excel in excels[1:]:
        common_sheets.intersection_update(excel.sheet_names)

    merged_sheets = {}

    for sheet_name in common_sheets:
        dfs = [pd.read_excel(file, sheet_name=sheet_name) for file in file_paths]
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates()

        merged_sheets[sheet_name] = merged

    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, data in merged_sheets.items():
            data.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"Merged sheets saved to {output_file}")



merge_excels()

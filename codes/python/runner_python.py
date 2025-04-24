import fileinput
import subprocess


# List of scripts to run
#script_paths = ['LR_refactor_2.py', 'LR_scipy.py', 'LR_statsmodels.py', 'LR_scikit.py'] #OOP missing
#script_paths = ['ANOVA_refactor.py','ANOVA_scipy.py', 'ANOVA_statsmodels.py'] #OOP missing
#script_paths = ['LR_refactor_2.py']  # OOP missing
script_paths = ['ANOVA_refactor.py']  # OOP missing
# Base file path and range of values
#base_path = "../../input/LR/LR_"
base_path = "../../input/ANOVA/ANOVA_"
#values = [100, 1000, 10000, 100000]
values = ['small', 'medium', 'large', 'extra_large']
excel_file_sizes = ['small', 'medium', 'large', 'extra_large']
# Number of repetitions for each value
repetitions = 10

# Lines to replace
#original_lines = [
#    "data = data_load('../../input/LR/LR_100.xlsx')",
#    'excel_file="../../output/execution_times/execution_times_python_small.xlsx"'
#]
#new_lines_template = [
#    "data = data_load('{base_path}{value}.xlsx')",
#   'excel_file="../../output/execution_times/execution_times_python_{size}.xlsx"'
#]

original_lines = [
    "data = data_load('../../input/ANOVA/ANOVA_small.xlsx')",
    'excel_file="../../output/execution_times/execution_times_python_small.xlsx"'
]
new_lines_template = [
    "data = data_load('{base_path}{value}.xlsx')",
    'excel_file="../../output/execution_times/execution_times_python_{size}.xlsx"'
]

for value, size in zip(values, excel_file_sizes):
    for script_path in script_paths:
        for _ in range(repetitions):
            new_lines = [
                new_lines_template[0].format(base_path=base_path, value=value),
                new_lines_template[1].format(size=size)
            ]

            with fileinput.FileInput(script_path, inplace=True, backup='.bak', encoding='utf-8') as file:
                for line in file:
                    for original, new in zip(original_lines, new_lines):
                        line = line.replace(original, new)
                    print(line, end='')

            print(f"Running script: {script_path} with file: {base_path}{value}.xlsx and size: {size}")
            result = subprocess.run(['python', script_path], capture_output=True, text=True)

            print("Output:", result.stdout)

            with fileinput.FileInput(script_path, inplace=True, encoding='utf-8') as file:
                for line in file:
                    for new, original in zip(new_lines, original_lines):
                        line = line.replace(new, original)
                    print(line, end='')
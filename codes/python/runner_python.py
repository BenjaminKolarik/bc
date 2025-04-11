import os
import subprocess
script_path = 'LR_statsmodels.py'

num_runs = 10
for i in range(num_runs):
    print(f"Running {script_path} - Iteration {i+1}")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)
import os
import subprocess

# Path to the R script
script_path = '../R/ANOVA.R'

# Number of times to run the script
num_runs = 10

for i in range(num_runs):
    print(f"Running {script_path} - Iteration {i+1}")
    result = subprocess.run(['Rscript', script_path], capture_output=True, text=True)

    # Print the output of the script
    print(result.stdout)
    print(result.stderr)
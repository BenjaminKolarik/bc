import subprocess
from codes.python.execution_timer import measure_execution_time

def main():
    r_path = "../R/LR.R"
    subprocess.run(["Rscript", r_path])

if __name__ == '__main__':
    result = measure_execution_time(main)

    if isinstance(result, tuple):
        wait_time, execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")
        print(f"Waiting time: {wait_time:.6f} seconds")
        print(f"Active execution time: {execution_time - wait_time:.6f} seconds")
    else:
        execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")
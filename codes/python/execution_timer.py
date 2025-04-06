import time
import pandas as pd
import os
import datetime
from pathlib import Path

def measure_execution_time(func, *args, **kwargs):

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    if result is not None:
        return result, end_time - start_time
    else:
        return execution_time

def timed_input(prompt=""):

    pause_time = time.time()
    user_input = input(prompt)
    resume_time = time.time()
    return user_input, resume_time - pause_time


def append_execution_time(execution_time, method, computer_name, excel_file="../../output/execution_times/execution_times_python_medium.xlsx"):
    Path(os.path.dirname(excel_file)).mkdir(parents=True, exist_ok=True)

    new_data = pd.DataFrame({
        "Method": [method],
        "Computer": [computer_name],
        "Execution_time": [float(execution_time)],
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Language": ["Python"]
    })

    try:
        if os.path.exists(excel_file):
            try:
                with pd.ExcelFile(excel_file) as xls:
                    sheet_dict = {sheet: pd.read_excel(excel_file, sheet_name=sheet)
                                  for sheet in xls.sheet_names}

                if method in sheet_dict:
                    sheet_dict[method] = pd.concat([sheet_dict[method], new_data], ignore_index=True)
                else:
                    sheet_dict[method] = new_data

                with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                    for sheet_name, df in sheet_dict.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

            except Exception as e:
                print(f"Error reading existing Excel file: {str(e)}")
                print("Creating new Excel file instead...")
                with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                    new_data.to_excel(writer, sheet_name=method, index=False)
        else:
            # Create new file
            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                new_data.to_excel(writer, sheet_name=method, index=False)

        print(f"Execution time {execution_time:.6f} seconds added to {excel_file} in sheet '{method}'")

    except Exception as e:
        print(f"Error appending execution time: {str(e)}")
        print(f"Execution time was {execution_time:.6f} seconds for method {method}")


class ExecutionTimer:

    def __init__(self):
        self.total_time = 0
        self.wait_time = 0
        self.start_time = None
        self.is_running = False

    def start(self):
        """Start the timer"""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True

    def stop(self):
        """Stop the timer and return the elapsed time"""
        if self.is_running:
            self.total_time += time.time() - self.start_time
            self.is_running = False
            return self.total_time
        return None

    def add_wait_time(self, wait_time):
        """Add user input waiting time to be excluded from total execution time"""
        self.wait_time += wait_time

    def get_active_time(self):
        """Return the active execution time (total time - wait time)"""
        return self.total_time - self.wait_time

    def reset(self):
        """Reset the timer"""
        self.total_time = 0
        self.wait_time = 0
        self.start_time = None
        self.is_running = False
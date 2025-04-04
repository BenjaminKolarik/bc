from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

from codes.python.execution_timer import measure_execution_time
def load_data():
    data = pd.read_excel("../../input/develop_test/LR.xlsx")
    lr_data = data[['y', 'x']]
    lr_array = lr_data.to_numpy()
    sum_values = lr_array[-1, :]
    lr_array = lr_array[:-1, :]
    return lr_array, sum_values

def main():
    array, sum_array = load_data()
    x = array[:, 1].reshape(-1, 1)
    y = array[:, 0]

    model = LinearRegression()
    model.fit(x, y)
    print(f"Slope: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

if __name__ == "__main__":
    execution_time = measure_execution_time(main)
    print(f"Total execution time: {execution_time:.6f} seconds")

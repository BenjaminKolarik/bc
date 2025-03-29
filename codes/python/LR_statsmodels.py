import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from codes.python.execution_timer import measure_execution_time

def data_load():
    data = pd.read_excel("../../input/develop_test/LR.xlsx")
    LR_data = data[['y', 'x']]
    LR_array = LR_data.to_numpy()
    sum_values = LR_array[-1, :]
    LR_array = LR_array[:-1, :]

    return LR_array, sum_values

def graph(array_y, array_x, model):
    array_x = sm.add_constant(array_x)  # Add constant term
    y_pred = model.predict(array_x)
    plt.scatter(array_x[:, 1], array_y, label="Data Points", color='blue')
    plt.plot(array_x[:, 1], y_pred, label="Regression line", color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression using Statsmodels")
    plt.legend()
    plt.show()


def main():
    array, sum_array = data_load()

    x = array[:, 1]
    y = array[:, 0]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    y_pred = model.predict(x)
    print(y_pred)
    print(x)
    print(model.summary())
    #sns.residplot(x=array[:, 1], y=y, lowess=True, color="g")
    graph(y, array[:, 1], model)
    plt.show()

if __name__ == "__main__":
    result = measure_execution_time(main)

    if isinstance(result, tuple):
        wait_time, execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")
        print(f"Waiting time: {wait_time:.6f} seconds")
        print(f"Active execution time: {execution_time - wait_time:.6f} seconds")
    else:
        execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")
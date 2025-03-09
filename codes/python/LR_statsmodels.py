import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

def data_load():
    data = pd.read_excel("../input/mtcars/LR.xlsx")
    LR_data = data[['y', 'x']]
    LR_array = LR_data.to_numpy()
    sum_values = LR_array[-1, :]
    LR_array = LR_array[:-1, :]

    return LR_array, sum_values

def graph(array_y, array_x):
    array_x = sm.add_constant(array_x)  # Add constant term
    y_pred = model.predict(array_x)
    plt.scatter(array_x[:, 1], array_y, label="Data Points", color='blue')
    plt.plot(array_x[:, 1], y_pred, label="Regression line", color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression using Statsmodels")
    plt.legend()
    plt.show()



array, sum_array = data_load()

x = array[:, 1]
y = array[:, 0]
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
y_pred = model.predict(x)
print(y_pred)
print(x)
print(model.summary())
graph(y, array[:, 1])
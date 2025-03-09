from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_excel("../../input/mtcars/LR.xlsx")
    lr_data = data[['y', 'x']]
    lr_array = lr_data.to_numpy()
    sum_values = lr_array[-1, :]
    lr_array = lr_array[:-1, :]
    return lr_array, sum_values


array, sum_array = load_data()
x = array[:, 1].reshape(-1, 1)
y = array[:, 0]

model = LinearRegression()
model.fit(x, y)

print(array)
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")


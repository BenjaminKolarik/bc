import pandas as pd
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from codes.python.execution_timer import measure_execution_time, append_execution_time


def data_load(file_path):
    data = pd.read_excel(file_path)
    data = data[['y', 'x']].dropna()
    return data

def perform_regression(data):
    x = data[['x']]
    y = data['y']
    model = LinearRegression().fit(x, y)
    return model

def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    print(f"Model Evaluation:\nR-squared: {model.score(x, y):.4f}\nMSE: {mse:.4f}")

def plot_regression(data, model, output_dir):
    x = data[['x']]
    y_pred = model.predict(x)

    plt.figure(figsize=(10, 6))
    sns.regplot(x=data['x'], y=data['y'], line_kws={'color': 'red'})
    plt.title('Linear Regression')
    plt.savefig(output_dir + 'linear_regression.png')
    plt.close()

    residuals = data['y'] - y_pred
    plt.figure(figsize=(10, 6))
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals')
    plt.savefig(output_dir + 'qq_residuals.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.residplot(x=data['x'], y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(output_dir + 'residual_plot.png')
    plt.close()

def test_assumptions(data, slope, intercept):
    x = data['x']
    y = data['y']
    y_pred = intercept + slope * x
    residuals = y - y_pred

    # Homoscedasticity test (Breusch-Pagan)
    bp_test = het_breuschpagan(residuals, sm.add_constant(x))
    print(f"\nBreusch-Pagan test: p-value = {bp_test[1]:.4f}")

    # Normality test (Shapiro-Wilk)
    shapiro_test = shapiro(residuals)
    print(f"Shapiro-Wilk test: p-value = {shapiro_test.pvalue:.4f}")

    # Independence test (Durbin-Watson)
    dw_test = durbin_watson(residuals)
    print(f"Durbin-Watson test: statistic = {dw_test:.4f}")


def main():
    os.makedirs("../../output/LR/LR_sklearn", exist_ok=True)
    data = data_load("../../input/LR/LR_1000.xlsx")
    model = perform_regression(data)
    evaluate_model(model, data[['x']], data['y'])
    plot_regression(data, model, "../../output/LR/LR_sklearn/")
    test_assumptions(data, model.coef_[0], model.intercept_)

if __name__ == "__main__":
    result = measure_execution_time(main)

    if isinstance(result, tuple):
        wait_time, execution_time = result
        print(f"\nTotal execution time: {execution_time:.6f} seconds")
        print(f"Waiting time: {wait_time:.6f} seconds")
        print(f"Active execution time: {execution_time - wait_time:.6f} seconds")
        append_execution_time(
            execution_time,
            method="LR_scikit",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
    else:
        execution_time = result
        print(f"\nTotal execution time: {execution_time:.6f} seconds")
        append_execution_time(
            execution_time,
            method="LR_scikit",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
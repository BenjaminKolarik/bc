import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from codes.python.execution_timer import measure_execution_time, append_execution_time

def perform_regression(data):
    x = data['x']
    y = data['y']

    x = sm.add_constant(x)
    print(x)
    model = sm.OLS(y, x).fit()

    return model

def data_load(file_path):
    data = pd.read_excel(file_path)
    data = data[['y', 'x']].dropna()
    return data

def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    print(f"Model Evaluation:\nR-squared: {model.rsquared:.4f}\nAdjusted R-squared: {model.rsquared_adj:.4f}\nMSE: {mse:.4f}")

def plot_regression(data, model, output_dir):
    x = sm.add_constant(data['x'])
    y_pred = model.predict(x)

    plt.figure(figsize=(10, 6))
    sns.regplot(x=data['x'], y=data['y'], line_kws={'color': 'red'})
    plt.title('Linear Regression')
    plt.savefig(output_dir + 'linear_regression.png')
    plt.close()

    residuals = model.resid
    fig = sm.qqplot(residuals, line='s')
    fig.set_size_inches(10, 6)
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

    plt.figure(figsize=(10, 6))
    plt.scatter(data['x'], data['y'], label='Data', color='blue')
    plt.plot(data['x'], y_pred, label='Regression Line', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.savefig(output_dir + 'alt.png')
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

    os.makedirs("../../output/LR/LR_statsmodels", exist_ok=True)

    data = data_load('../../input/LR/LR_100.xlsx')
    model = perform_regression(data)
    print(model.summary())
    evaluate_model(model, sm.add_constant(data['x']), data['y'])
    plot_regression(data, model, "../../output/LR/LR_statsmodels/")

    return

if __name__ == "__main__":
    result = measure_execution_time(main)

    if isinstance(result, tuple):
        wait_time, execution_time = result
        print(f"\nTotal execution time: {execution_time:.6f} seconds")
        print(f"Waiting time: {wait_time:.6f} seconds")
        print(f"Active execution time: {execution_time - wait_time:.6f} seconds")

        append_execution_time(
            execution_time - wait_time,
            method="LR - statsmodels",
            computer_name="Windows Ryzen 9 5900x 32GB",
            excel_file="../../output/execution_times/h/moje/execution_times_python_small.xlsx"
        )
    else:
        execution_time = result
        print(f"\nTotal execution time: {execution_time:.6f} seconds")

        append_execution_time(
            execution_time,
            method="LR - statsmodels",
            computer_name="Windows Ryzen 9 5900x 32GB",
            excel_file="../../output/execution_times/h/moje/execution_times_python_small.xlsx"
        )
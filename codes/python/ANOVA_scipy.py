import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from codes.python.execution_timer import measure_execution_time, append_execution_time

def load_data(file_path):

    data = pd.read_excel(file_path)
    data_clean = data.dropna(subset = ['value'])
    return data_clean

def perform_anova(data):

    groups = [data[data['group'] == group]['value'] for group in data['group'].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"SciPy ANOVA results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

    return f_stat, p_value

def graph(data):

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='value', data=data)
    plt.title('Boxplot of Values by Group')
    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.savefig('../../output/ANOVA/ANOVA_scipy/boxplot.png')
    plt.close()

    group_stats = data.groupby('group')['value'].agg(['mean', 'std', 'count']).reset_index()
    print("\nGroup Statistics:")
    print(group_stats)

    return group_stats

def check_assumptions(data, model):
    residuals = model.resid
    _, p_norm = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test for normality: p-value = {p_norm:.4f}")
    if p_norm > 0.05:
        print("Residuals are normally distributed.")
    else:
        print("Residuals are not normally distributed.")

    plt.figure(figsize=(10, 6))
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals')
    plt.savefig('../../output/ANOVA/ANOVA_scipy/qqplot.png')
    plt.close()

    groups = [data[data['group'] == group]['value'] for group in data['group'].unique()]
    _, p_levene = stats.levene(*groups)
    print(f"Levene's test for homogeneity of variances: p-value = {p_levene:.4f}")
    if p_levene > 0.05:
        print("Variances are homogeneous.")
    else:
        print("Variances are not homogeneous.")

def post_hoc_analysis(data):
    tukey = pairwise_tukeyhsd(data['value'], data['group'], alpha = 0.05)
    print("\nPost-hoc analysis (Tukey's HSD):")
    print(tukey)

def main():

    os.makedirs("../../output/ANOVA/ANOVA_scipy", exist_ok=True)
    data = load_data("../../input/ANOVA/ANOVA_medium.xlsx")
    model, anova_table = perform_anova(data)
    groups_stats = graph(data)
    check_assumptions(data, model)

    if anova_table.iloc[0, 4] < 0.05:
        post_hoc_analysis(data)

if __name__ == "__main__":
    result = measure_execution_time(main)

    if isinstance(result, tuple):
        wait_time, execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")
        print(f"Waiting time: {wait_time:.6f} seconds")
        print(f"Active execution time: {execution_time - wait_time:.6f} seconds")

        append_execution_time(
            execution_time - wait_time,
            method="ANOVA_scipy",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
    else:
        execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")

        append_execution_time(
            execution_time,
            method="ANOVA_scipy",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
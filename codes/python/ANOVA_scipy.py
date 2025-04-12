import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from codes.python.execution_timer import measure_execution_time, append_execution_time


def data_load(file_path):

    data = pd.read_excel(file_path)
    data_clean = data.dropna(subset = ['value'])
    return data_clean

def perform_anova(data):

    groups = [data[data['group'] == group]['value'] for group in data['group'].unique()]
    print(groups, "\n", print(type(groups)))
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

    group_means = data.groupby('group')['value'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='group', y='value', hue='group', data=group_means, palette='viridis')
    plt.title('Barplot of Values by Group')
    plt.xlabel('Groups')
    plt.ylabel('Mean Values')
    plt.savefig('../../output/ANOVA/ANOVA_scipy/barplot.png')
    plt.close()

    group_stats = data.groupby('group')['value'].agg(['mean', 'std', 'count']).reset_index()
    print("\nGroup Statistics:")
    print(group_stats)

    return group_stats

def test_assumptions(data_frame, group_col, value_col):
    model = ols(f'{value_col} ~ C({group_col})', data=data_frame).fit()
    residuals = model.resid

    shapiro_test = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk test for normality: p-value = {shapiro_test.pvalue:.4f}")

    groups = [data_frame[data_frame[group_col] == group][value_col] for group in data_frame[group_col].unique()]
    levene_test = stats.levene(*groups)
    print(f"Levene's test for homogeneity of variances: p-value = {levene_test.pvalue:.4f}")


def post_hoc_analysis(data):

    tukey = pairwise_tukeyhsd(data['value'], data['group'], alpha = 0.05)
    print("\nPost-hoc analysis (Tukey's HSD):")
    print(tukey)

def main():

    os.makedirs("../../output/ANOVA/ANOVA_scipy", exist_ok=True)
    data = data_load('../../input/ANOVA/ANOVA_small.xlsx')
    f_stat, p_value = perform_anova(data)
    groups_stats = graph(data)
    test_assumptions(data, "group", "value")

    if p_value < 0.05:
        print("Post-hoc analysis")
        post_hoc_analysis(data)

if __name__ == "__main__":
    result = measure_execution_time(main)

    if isinstance(result, tuple):
        wait_time, execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")
        print(f"Waiting time: {wait_time:.6f} seconds")
        print(f"Active execution time: {execution_time - wait_time:.6f} seconds")

        append_execution_time(
            execution_time,
            method="ANOVA - scipy",
            computer_name="Windows Ryzen 9 5900x 32GB",
            excel_file="../../output/execution_times/execution_times_python_small.xlsx"
        )
    else:
        execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")

        append_execution_time(
            execution_time,
            method="ANOVA - scipy",
            computer_name="Windows Ryzen 9 5900x 32GB",
            excel_file="../../output/execution_times/execution_times_python_small.xlsx"
        )
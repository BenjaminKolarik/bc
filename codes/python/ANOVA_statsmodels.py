import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

from codes.python.execution_timer import measure_execution_time, append_execution_time


def load_data(file_path):
    data = pd.read_excel(file_path)
    data_clean = data.dropna(subset=['value'])
    return data_clean


def perform_anova(data):
    model = ols('value ~ C(group)', data=data).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)
    print("Statsmodels ANOVA results:")
    print(anova_table)

    return model, anova_table


def graph(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='value', data=data)
    plt.title('Boxplot of Values by Group')
    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.savefig('../../output/ANOVA/ANOVA_statsmodels/boxplot.png')
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
    plt.savefig('../../output/ANOVA/ANOVA_statsmodels/qqplot.png')
    plt.close()

    groups = [data[data['group'] == group]['value'] for group in data['group'].unique()]
    _, p_levene = stats.levene(*groups)
    print(f"Levene's test for homogeneity of variances: p-value = {p_levene:.4f}")
    if p_levene > 0.05:
        print("Variances are homogeneous.")
    else:
        print("Variances are not homogeneous.")


def post_hoc_analysis(data):
    tukey = pairwise_tukeyhsd(data['value'], data['group'], alpha=0.05)
    print("\nPost-hoc analysis (Tukey's HSD):")
    print(tukey)

    plt.figure(figsize=(10, 6))
    tukey.plot_simultaneous()
    plt.tight_layout()
    plt.savefig('../../output/ANOVA/ANOVA_statsmodels/tukey_test.png')
    plt.close()


def main():
    wait_time = 0
    os.makedirs("../../output/ANOVA/ANOVA_statsmodels", exist_ok=True)

    data = load_data("../../input/ANOVA/ANOVA_medium.xlsx")

    model, anova_table = perform_anova(data)

    graph(data)

    check_assumptions(data, model)

    if anova_table.loc['C(group)', 'PR(>F)'] < 0.05:
        post_hoc_analysis(data)

    return wait_time

if __name__ == "__main__":
    result = measure_execution_time(main)

    if isinstance(result, tuple):
        wait_time, execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")
        print(f"Waiting time: {wait_time:.6f} seconds")
        print(f"Active execution time: {execution_time - wait_time:.6f} seconds")

        append_execution_time(
            execution_time - wait_time,
            method="ANOVA_statsmodels",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
    else:
        execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")

        append_execution_time(
            execution_time,
            method="ANOVA_statsmodels",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
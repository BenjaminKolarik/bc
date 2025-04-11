import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

from codes.python.execution_timer import measure_execution_time, append_execution_time


def perform_anova(data):
    model = ols('value ~ C(group)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("Statsmodels ANOVA results:")
    print(anova_table)

    return model, anova_table

def load_data(file_path):
    data = pd.read_excel(file_path)
    data_clean = data.dropna(subset=['value'])
    return data_clean


def graph(data):

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='value', data=data)
    plt.title('Boxplot of Values by Group')
    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.savefig('../../output/ANOVA/ANOVA_statsmodels/boxplot.png')
    plt.close()

    group_means = data.groupby('group')['value'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='group', y='value', hue='group', data=group_means, palette='viridis')
    plt.title('Barplot of Values by Group')
    plt.xlabel('Groups')
    plt.ylabel('Mean Values')
    plt.savefig('../../output/ANOVA/ANOVA_statsmodels/barplot.png')
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

    # Homogeneity of variances test (Levene's test)
    groups = [data_frame[data_frame[group_col] == group][value_col] for group in data_frame[group_col].unique()]
    levene_test = stats.levene(*groups)
    print(f"Levene's test for homogeneity of variances: p-value = {levene_test.pvalue:.4f}")



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
    os.makedirs("../../output/ANOVA/ANOVA_statsmodels", exist_ok=True)

    data = load_data("../../input/ANOVA/ANOVA_medium.xlsx")

    model, anova_table = perform_anova(data)

    group_stats = graph(data)

    test_assumptions(data, "group", "value")

    if anova_table.loc['C(group)', 'PR(>F)'] < 0.05:
        print("Post-hoc analysis")
        post_hoc_analysis(data)

    return

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
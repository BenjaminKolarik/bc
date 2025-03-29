import random as rd
import pandas as pd
import os
import numpy as np


def generate_anova_data(num_samples_per_group, num_groups, excel_file, nan_probability=0.02):

    data = {'group': [], 'value': []}

    for group in range(1, num_groups + 1):
        group_mean = 50 + group * 5

        for _ in range(num_samples_per_group):
            data['group'].append(f"Group_{group}")

            if rd.random() < nan_probability:
                data['value'].append(np.nan)
            else:
                value = round(group_mean + rd.normalvariate(0, 5))
                data['value'].append(value)

    nan_count = sum(pd.isna(v) for v in data['value'])
    total_samples = num_samples_per_group * num_groups

    print(f"Dataset {excel_file}: NaN count - {nan_count} ({nan_count / total_samples:.2%})")

    excel_writer(excel_file, data)


def excel_writer(excel_file, data, output_dir='../../input/ANOVA'):
    df = pd.DataFrame(data)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, excel_file)

    df.to_excel(file_path, index=False)
    print(f"Data written to {file_path}")


generate_anova_data(num_samples_per_group=20, num_groups=5, excel_file="ANOVA_small.xlsx")
generate_anova_data(num_samples_per_group=125, num_groups=8, excel_file="ANOVA_medium.xlsx")
generate_anova_data(num_samples_per_group=1000, num_groups=10, excel_file="ANOVA_large.xlsx")
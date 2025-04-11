import random as rd
import pandas as pd
import os
import numpy as np

def generate_data(num_samples, excel_file, nan_probability=0.02):
    random_numbers_x = []
    for _ in range(num_samples):
        if rd.random() < nan_probability:
            random_numbers_x.append(np.nan)
        else:
            random_numbers_x.append(rd.randint(1, 100))

    random_numbers_y = []
    for x in random_numbers_x:
        if pd.isna(x) or rd.random() < nan_probability:
            random_numbers_y.append(np.nan)
        else:
            random_numbers_y.append(x + 10 + rd.randint(0, 15))

    x_nan_count = sum(pd.isna(x) for x in random_numbers_x)
    y_nan_count = sum(pd.isna(y) for y in random_numbers_y)
    total_nan_count = x_nan_count + y_nan_count

    print(f"Dataset {excel_file}: NaN count - x: {x_nan_count} ({x_nan_count / num_samples:.2%}), "
          f"y: {y_nan_count} ({y_nan_count / num_samples:.2%}), "
          f"total: {total_nan_count} ({total_nan_count / (num_samples * 2):.2%})")

    data = {'y': random_numbers_y, 'x': random_numbers_x}
    excel_writer(excel_file, data)


def excel_writer(excel_file, data, output_dir = '../../input/LR'):
    df = pd.DataFrame(data)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, excel_file)

    df.to_excel(file_path, index=False)
    print(f"Data written to {file_path}")


#generate_data(100, excel_file="LR_100.xlsx")
#generate_data(1000, excel_file="LR_1000.xlsx")
#generate_data(10000, excel_file="LR_10000.xlsx")
generate_data(100000, excel_file="LR_100000.xlsx")
import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook, Workbook
from scipy.stats import f

def data_load():

    data = pd.read_excel("../input/mtcars/LR.xlsx")
    LR_data = data[['y', 'x']]
    LR_array = LR_data.to_numpy()
    sum_values = LR_array[-1, :]
    LR_array = LR_array[:-1, :]


    return LR_array, sum_values

def calculations(array, sums_values):
    y = array[:, 0]
    x = array[:, 1]
    x_average = np.mean(x)
    y_average = np.mean(y)

    xy = x * y
    xy_avg = np.mean(xy)
    y_2 = y ** 2
    x_2 = x ** 2
    sum_y = sums_values[0]
    sum_x = sums_values[1]
    sum_x2 = np.sum(x ** 2)
    sum_y2 = np.sum(y ** 2)
    sum_xy = np.sum(x * y)
    x_x_avg = x - x_average
    y_y_avg = y - y_average
    y_y_avg2 = y_y_avg ** 2
    x_x_avg2 = x_x_avg ** 2
    y_y_avg2_sum = np.sum(y_y_avg2)
    x_x_avg2_sum = np.sum(x_x_avg2)
    length = len(x)
    xx_yy = (x - x_average) * (y - y_average)
    xx_yy_sum = np.sum(xx_yy)
    return x_average, y_average, xy, xy_avg, y_2, x_2, sum_y, sum_x, sum_x2, sum_y2, sum_xy, x_x_avg, y_y_avg, y_y_avg2, x_x_avg2, y_y_avg2_sum, x_x_avg2_sum, length, xx_yy, xx_yy_sum


def regression_coefficients(cov, scatter, y_average, x_average):
    b1 = cov / scatter
    b0 = y_average - (b1 * x_average)
    return b0, b1

def covariance_and_scatter(xy_average, x_average, y_average, x_2):
    x_2_avg = np.mean(x_2)
    cov = xy_average - (x_average * y_average)
    scatter = x_2_avg - (x_average ** 2)
    return cov, scatter

def bal_line_values(b0, b1, x):
    balancing_line = b0 + (b1 * x)
    sum_bal_line = np.sum(balancing_line)
    return balancing_line, sum_bal_line

def squares_sum(y, y_average, bal_line):
    SSM = np.sum((bal_line - y_average) ** 2)
    SSR = np.sum((y - bal_line) ** 2)
    SST = np.sum((y - y_average) ** 2)
    return SSM, SSR, SST

def DF(length):
    return [1, length - 2, length - 1]

def mean_squares(SSR, SSM, df):
    MSR = SSM / df[0]
    MSE = SSR / df[1]
    return MSR, MSE

def validation_F_stat(MSM, MSE, df):
    F = MSM / MSE
    if F > f.ppf(0.95, df[0], df[1]):
        verification = True
    else:
        verification = False
    return verification, F

def write_data_to_excel(y, x, x_y, y_2, x_2, x_x_avg, y_y_avg, xx_yy, x_x_avg2, y_y_avg2, balancing_line, SSM, SSR, SST, df_values, MSA, MSE, F, file_name = 'output_data_LR.xlsx', output_dir = '../output/'):

    file_path = os.path.join(output_dir, file_name)
    error_values = y - balancing_line
    ssr_values = error_values ** 2
    df1 = pd.DataFrame({
        'Dom i': [i for i in range(length)] + ['∑'],
        'Spotreba (v kWh/mesiac) yi': list(y) + [np.sum(y)],
        'Priemer (v m**2) xi': list(x) + [np.sum(x)],
        'x*y': list(x_y) + [np.sum(x_y)],
        'y^2': list(y_2) + [np.sum(y_2)],
        'x^2': list(x_2) + [np.sum(x_2)],
        '(xi - x_avg)': list(x_x_avg) + [np.sum(x_x_avg2)],
        '(yi - y_avg)': list(y_y_avg) + [np.sum(y_y_avg)],
        '(xi - x_avg) - (yi - y_avg)': list(xx_yy) + [np.sum(xx_yy)],
        '(xi - x_avg)^2': list(x_x_avg2) + [np.sum(x_x_avg2)],
        '(yi - y_avg)^2': list(y_y_avg2) + [np.sum(y_y_avg2)],
    })

    df2 = pd.DataFrame({
        'Dom i': [i for i in range(length)] + ['∑'],
        'Spotreba (v kWh/mesiac) yi': list(y) + [np.sum(y)],
        'Priemer (v m**2) xi': list(x) + [np.sum(x)],
        'y^i': list(balancing_line) + [np.sum(balancing_line)],
        'error': list(error_values) + [np.sum(error_values)],
        '(yi - y^i)^2': list(ssr_values) + [np.sum(ssr_values)],
    })
    df3 = pd.DataFrame({
        'Variabilita premennej Y': ['Vysvetlená regresným modelom',
                                    'Nevysvetlená regresným modelom',
                                    'Celková variabilita'],

        'Súčet štvorcov SS': ["SSM = {:.2f}".format(SSM),
                              "SSR = {:.2f}".format(SSR),
                              "SST = {:.2f}".format(SST)],

        'Stupne voľnosti DF': df_values,

        'Priemerný súčet štvorcov MS': ["MSA = {:.2f}".format(MSA),
                                        "MSE = {:.2f}".format(MSE),
                                        float('nan')],
        'F_stat': [
            "F = {:.2f}".format(F),
            float('nan'),
            float('nan')]
    })

    with pd.ExcelWriter(file_path) as writer:
        df1.to_excel(writer, sheet_name='Výpočtová tabuľka1', index=False)
        df2.to_excel(writer, sheet_name='Výpočtová tabuľka2', index=False)
        df3.to_excel(writer, sheet_name='LR', index=False)

    workbook = load_workbook(file_path)
    sheet1 = workbook['Výpočtová tabuľka1']
    sheet2 = workbook['Výpočtová tabuľka2']
    sheet3 = workbook['LR']

    for sheet in [sheet1, sheet2, sheet3]:
        for col in sheet.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00'
                        current_length = len(str(int(cell.value))) + 3
                        if current_length > max_length:
                            max_length = current_length
                    elif len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

                adjusted_width = (max_length + 2) * 1.2
                sheet.column_dimensions[column].width = adjusted_width

    workbook.save(file_path)

    print(f"Data has been written to {file_path}")

data_set, sums = data_load()
y = data_set[:, 0]
x = data_set[:, 1]
x_average, y_average, xy, xy_average, y_2, x_2, sum_y, sum_x, sum_x2, sum_y2, sum_xy, x_x_avg, y_y_avg, y_y_avg2, x_x_avg2, y_y_avg2_sum, x_x_avg2_sum, length, xx_yy, xx_yy_sum = calculations(data_set, sums)
cov, scatter = covariance_and_scatter(xy_average, x_average, y_average, x_2)
b0, b1 = regression_coefficients(cov, scatter, y_average, x_average)
balancing_line, sum_bal_line = bal_line_values(b0, b1, data_set[:, 1])
squares_sum(data_set[:, 0], y_average, balancing_line)
SSM, SSR, SST = squares_sum(data_set[:, 0], y_average, balancing_line)
df = DF(length)
MSM, MSR = mean_squares(SSR, SSM, df)
validation, F_value = validation_F_stat(MSM, MSR, df)
write_data_to_excel(y, x, xy, y_2, x_2, x_x_avg, y_y_avg, xx_yy, x_x_avg2, y_y_avg2, balancing_line, SSM, SSR, SST, df, MSM, MSR, F_value)
"""
print(MSM, MSR)
print(validation, F_value)

print(balancing_line, sum_bal_line)
print(cov, scatter)
print(f"x_average: {x_average}")
print(f"y_average: {y_average}")
print(f"xy: {xy}")
print(f"xy_average: {xy_average}")
print(f"y_2: {y_2}")
print(f"x_2: {x_2}")
print(f"sum_y: {sum_y}")
print(f"sum_x: {sum_x}")
print(f"sum_x2: {sum_x2}")
print(f"sum_y2: {sum_y2}")
print(f"sum_xy: {sum_xy}")
"""
print(f"x_x_avg: {x_x_avg}")
print(f"y_y_avg: {y_y_avg}")
print(f"y_y_avg2: {y_y_avg2}")
print(f"x_x_avg2: {x_x_avg2}")
print(f"y_y_avg2_sum: {y_y_avg2_sum}")
print(f"x_x_avg2_sum: {x_x_avg2_sum}")
print(f"xx_yy: {xx_yy}")
print(f"xx_yy_sum: {xx_yy_sum}")
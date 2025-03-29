import pandas as pd
import numpy as np
from openpyxl import load_workbook

class DataProcessor:
    def __init__(self, array):
        self.array = array
        self.bez_nan = []
        self.suma = []
        self.pocty = []
        self.priemer = []
        self.sum_bez_nan = 0

    def process_data(self):
        for row in self.array:
            valid_values = [i for i in row if not isinstance(i, float) or not np.isnan(i)]
            self.bez_nan.extend(valid_values)
            sumy = sum(i for i in row if not isinstance(i, float) or not np.isnan(i))
            self.suma.append(sumy)
            self.pocty.append(len(valid_values))

        self.bez_nan = [float(i) for i in self.bez_nan]
        self.priemer = [float(self.suma[i] / self.pocty[i]) for i in range(len(self.pocty))]
        self.sum_bez_nan = np.sum(~np.isnan(self.array))

    def get_priemer_and_priemer_c(self):
        priemer_c = sum(self.suma) / sum(self.pocty)
        return self.priemer, priemer_c

class AnovaCalculator:
    def __init__(self, count, priemer, priemer_c, bez_nan):
        self.count = count
        self.priemer = priemer
        self.priemer_c = priemer_c
        self.bez_nan = bez_nan

    def calculate_SSA(self):
        SSA_values = [(self.priemer[i] - self.priemer_c) ** 2 * self.count[i] for i in range(len(self.count))]
        SSA_value = sum(SSA_values)
        return SSA_values, SSA_value

    def calculate_SSE(self):
        start_index = 0
        SSE_values = []

        for i in range(len(self.count)):
            end_index = start_index + self.count[i]
            group_values = self.bez_nan[start_index:end_index]
            tmp = sum((value - self.priemer[i]) ** 2 for value in group_values)
            SSE_values.append(tmp)
            start_index = end_index

        SSE_value = sum(SSE_values)
        return SSE_values, SSE_value

    def calculate_SST(self):
        start_index = 0
        SST_values = []

        for i in range(len(self.count)):
            end_index = start_index + self.count[i]
            group_values = self.bez_nan[start_index:end_index]
            tmp = sum((value - self.priemer_c) ** 2 for value in group_values)
            SST_values.append(tmp)
            start_index = end_index

        SST_value = sum(SST_values)
        return SST_values, SST_value

class Statistics:
    @staticmethod
    def calculate_df(pocty, sum_bez_nan):
        return [float(len(pocty) - 1), float(sum_bez_nan - len(pocty)), float(sum_bez_nan - 1)]

    @staticmethod
    def calculate_F_stat(SSA, SSE, count, sum_bez_nan):
        MSA = SSA / (len(count) - 1)
        MSE = SSE / (sum_bez_nan - len(count))
        F = MSA / MSE
        return MSA, MSE, F

class ExcelWriter:
    def __init__(self, df1, df2, file_name='output_data.xlsx'):
        self.df1 = df1
        self.df2 = df2
        self.file_name = file_name

    def write_to_excel(self):
        with pd.ExcelWriter(self.file_name) as writer:
            self.df1.to_excel(writer, sheet_name='Výpočtová tabuľka', index=False)
            self.df2.to_excel(writer, sheet_name='ANOVA', index=False)

        self.format_excel()

    def format_excel(self):
        workbook = load_workbook(self.file_name)
        sheet1 = workbook['Výpočtová tabuľka']
        sheet2 = workbook['ANOVA']

        for sheet in [sheet1, sheet2]:
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

        workbook.save(self.file_name)
        print(f"Data has been written to {self.file_name}")


data = pd.read_excel("../input/develop_test/tst_x.xlsx")
P_data = data[['Predajňa 1', 'Predajňa 2', 'Predajňa 3']]
P_array = P_data.to_numpy()

processor = DataProcessor(P_array)
processor.process_data()
priemer, priemer_c = processor.get_priemer_and_priemer_c()

anova_calc = AnovaCalculator(processor.pocty, priemer, priemer_c, processor.bez_nan)
SSA_values, SSA_value = anova_calc.calculate_SSA()
SSE_values, SSE_value = anova_calc.calculate_SSE()
SST_values, SST_value = anova_calc.calculate_SST()

df_values = Statistics.calculate_df(processor.pocty, processor.sum_bez_nan)
MSA, MSE, F = Statistics.calculate_F_stat(SSA_value, SSE_value, processor.pocty, processor.sum_bez_nan)

values_column = [','.join(map(str, row)) for row in P_array]

df1 = pd.DataFrame({
    'Priemer': priemer + [priemer_c] + [None] * (len(priemer) - 1),
    'Values': values_column + [None] * (len(priemer) - len(values_column)),
    'SSA Values': SSA_values + [None] * (len(priemer) - len(SSA_values)),
    'SSE Values': SSE_values + [None] * (len(priemer) - len(SSE_values)),
    'SST Values': SST_values + [None] * (len(priemer) - len(SST_values))
})

df2 = pd.DataFrame({
    '': ['Faktor', 'Nahoda', 'Spolu'],
    'Values': [
        "SSA = {:.2f}".format(SSA_value),
        "SSE = {:.2f}".format(SSE_value),
        "SST = {:.2f}".format(SST_value)],
    'df': df_values,
    'MS': ['MSA', 'MSE', float('nan')],
    'MS_v': [MSA, MSE, float('nan')],
    'F_stat': ["F = {:.2f}".format(F), float('nan'), float('nan')]
})

writer = ExcelWriter(df1, df2)
writer.write_to_excel()

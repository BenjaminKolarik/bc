import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from scipy import stats
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from codes.python.execution_timer import measure_execution_time, append_execution_time


class DataLoader:
    @staticmethod
    def load_data(file_path):
        try:
            data = pd.read_excel(file_path)
            print(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    @staticmethod
    def preprocess_data(data):
        if data is None:
            return None
        else:
            data = data.dropna()
            return data


class LinearRegression:
    def __init__(self, data, x_column, y_column):
        self.data = data
        self.x_column = x_column
        self.y_column = y_column
        self.results = None
        self.x_values = None
        self.y_values = None
        self.slope = None
        self.intercept = None
        self.r_value = None
        self.p_value = None
        self.std_err = None
        self.predictions = None
        self.residuals = None

    def run_analysis(self):
        try:
            self.x_values = self.data[self.x_column].values
            self.y_values = self.data[self.y_column].values

            self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(
                self.x_values, self.y_values)

            self.predictions = self.slope * self.x_values + self.intercept
            self.residuals = self.y_values - self.predictions

            rmse = np.sqrt(np.mean(self.residuals ** 2))
            r_squared = self.r_value ** 2
            adjusted_r_squared = 1 - (1 - r_squared) * (len(self.x_values) - 1) / (len(self.x_values) - 1 - 1)
            mean_absolute_error = np.mean(np.abs(self.residuals))

            self.results = {
                "slope": self.slope,
                "intercept": self.intercept,
                "r_value": self.r_value,
                "p_value": self.p_value,
                "std_err": self.std_err,
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
                "rmse": rmse,
                "mae": mean_absolute_error,
                "equation": f"y = {self.intercept:.4f} + {self.slope:.4f}x",
                "x_values": self.x_values,
                "y_values": self.y_values,
                "predictions": self.predictions,
                "residuals": self.residuals,
                "is_significant": self.p_value < 0.05
            }

            return self.results

        except Exception as e:
            print(f"Error in linear regression analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_results_summary(self):
        if self.results is None:
            return "Analysis not yet performed"

        return {
            "Equation": self.results["equation"],
            "R-squared": f"{self.results['r_squared']:.4f}",
            "P-value": f"{self.results['p_value']:.4f}",
            "Significant": "Yes" if self.results['is_significant'] else "No",
            "RMSE": f"{self.results['rmse']:.4f}",
            "MAE": f"{self.results['mae']:.4f}"
        }


class RegressionVisualizer:
    def __init__(self, regression, output_dir='../../output/LR/LR_OOP/graphs'):
        self.regression = regression
        self.data = regression.data
        self.x_column = regression.x_column
        self.y_column = regression.y_column
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def create_plots(self):
        if self.regression.results is None:
            print("No regression results available for plotting")
            return

        self._create_scatter_with_line()
        self._create_residual_plot()
        self._create_qq_plot()
        self._create_histogram_residuals()
        plt.close('all')

    def _create_scatter_with_line(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.x_column, y=self.y_column, data=self.data)

        x_min, x_max = min(self.regression.x_values), max(self.regression.x_values)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = self.regression.slope * x_line + self.regression.intercept
        plt.plot(x_line, y_line, color='red', label=self.regression.results["equation"])

        plt.title(f'Linear Regression: {self.y_column} vs {self.x_column}')
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regression_scatter.png'))

    def _create_residual_plot(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.regression.predictions, self.regression.residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'residual_plot.png'))

    def _create_qq_plot(self):
        plt.figure(figsize=(10, 6))
        sm.qqplot(self.regression.residuals, line='s', ax=plt.gca())
        plt.title('Q-Q Plot of Residuals')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'qq_plot.png'))

    def _create_histogram_residuals(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.regression.residuals, kde=True)
        plt.title('Histogram of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'residual_histogram.png'))

class RegressionTester:
    def __init__(self, regression):
        self.regression = regression
        self.residuals = regression.residuals
        self.x_values = regression.x_values

    def test_assumptions(self):

        # Homoscedasticity test (Breusch-Pagan)
        bp_test = het_breuschpagan(self.residuals, sm.add_constant(self.x_values))
        print(f"\nBreusch-Pagan test: p-value = {bp_test[1]:.4f}")

        # Normality test (Shapiro-Wilk)
        shapiro_test = shapiro(self.residuals)
        print(f"Shapiro-Wilk test: p-value = {shapiro_test.pvalue:.4f}")

        # Independence test (Durbin-Watson)
        dw_test = durbin_watson(self.residuals)
        print(f"Durbin-Watson test: statistic = {dw_test:.4f}")


class RegressionExporter:
    def __init__(self, regression, output_dir='../../output/LR/LR_OOP/excel', file_name='linear_regression_results.xlsx'):
        self.regression = regression
        self.results = regression.results
        self.output_dir = output_dir
        self.file_name = file_name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export_to_excel(self):
        if self.results is None:
            print("No results to export")
            return

        file_path = os.path.join(self.output_dir, self.file_name)

        summary_df = pd.DataFrame({
            "Parameter": ["Slope", "Intercept", "R-value", "R-squared", "Adjusted R-squared",
                          "P-value", "Standard Error", "RMSE", "MAE", "Equation", "Significant"],
            "Value": [self.results['slope'], self.results['intercept'], self.results['r_value'],
                      self.results['r_squared'], self.results['adjusted_r_squared'],
                      self.results['p_value'], self.results['std_err'], self.results['rmse'],
                      self.results['mae'], self.results['equation'],
                      "Yes" if self.results['is_significant'] else "No"]
        })

        data_df = pd.DataFrame({
            "X": self.results['x_values'],
            "Y (Actual)": self.results['y_values'],
            "Y (Predicted)": self.results['predictions'],
            "Residuals": self.results['residuals']
        })

        with pd.ExcelWriter(file_path) as writer:
            summary_df.to_excel(writer, sheet_name='Regression Summary', index=False)
            data_df.to_excel(writer, sheet_name='Data and Predictions', index=False)

        self._format_excel_file(file_path)
        print(f"Data exported to {file_path}")

    def _format_excel_file(self, file_path):
        workbook = load_workbook(file_path)

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            for col in sheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if isinstance(cell.value, float):
                            cell.number_format = '0.0000'
                            current_length = len(str(int(cell.value))) + 5
                            if current_length > max_length:
                                max_length = current_length
                        elif cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass

                adjusted_width = (max_length + 2) * 1.2
                sheet.column_dimensions[column].width = adjusted_width

        image_paths = [
            os.path.join('../../output/LR/LR_OOP/graphs', 'regression_scatter.png'),
            os.path.join('../../output/LR/LR_OOP/graphs', 'residual_plot.png'),
            os.path.join('../../output/LR/LR_OOP/graphs', 'qq_plot.png'),
            os.path.join('../../output/LR/LR_OOP/graphs', 'residual_histogram.png')
        ]

        if any(os.path.exists(path) for path in image_paths):
            sheet = workbook.create_sheet("Graphs")
            row_pos = 1

            for i, img_path in enumerate(image_paths):
                if os.path.exists(img_path):
                    img = Image(img_path)
                    sheet.add_image(img, f'A{row_pos}')
                    row_pos += 20

        workbook.save(file_path)


def main():

    data_loader = DataLoader()
    file_path = '../../input/LR/LR_1000.xlsx'
    data = data_loader.load_data(file_path)
    data = data_loader.preprocess_data(data)

    if data is None:
        print("Failed to load or preprocess data")
        return wait_time

    x_col = "x"
    y_col = "y"

    if x_col not in data.columns or y_col not in data.columns:
        print(f"Columns '{x_col}' and/or '{y_col}' not found in the data")
        print(f"Available columns: {', '.join(data.columns)}")
        return wait_time

    try:
        lr = LinearRegression(data, x_col, y_col)
        results = lr.run_analysis()

        if results is None:
            print("Linear regression analysis failed")
            return wait_time

        visualizer = RegressionVisualizer(lr)
        visualizer.create_plots()

        exporter = RegressionExporter(lr)
        exporter.export_to_excel()

        tester = RegressionTester(lr)
        tester.test_assumptions()

        summary = lr.get_results_summary()
        print("\nLinear Regression Results:")
        print(f"Equation: {summary['Equation']}")
        print(f"R-squared: {summary['R-squared']}")
        print(f"P-value: {summary['P-value']}")
        print(f"Significant: {summary['Significant']}")
        print(f"RMSE: {summary['RMSE']}")
        print(f"MAE: {summary['MAE']}")

    except Exception as e:
        print(f"An error occurred in the linear regression application: {str(e)}")
        import traceback
        traceback.print_exc()

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
            method="LR - OOP",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
    else:
        execution_time = result
        print(f"Total execution time: {execution_time:.6f} seconds")

        append_execution_time(
            execution_time,
            method="LR - OOP",
            computer_name="Windows Ryzen 9 5900x 32GB"
        )
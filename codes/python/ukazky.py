import pandas as pd
import numpy as np

data = pd.read_excel("./input/develop_test/tst.xlsx")

print(data.describe())
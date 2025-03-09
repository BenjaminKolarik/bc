import pandas as pd
import numpy as np

data = pd.read_excel("./input/mtcars/tst.xlsx")

print(data.describe())
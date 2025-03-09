from scipy.stats import f

df1 = 1
df2 = 8
confidence_level = 0.95

f_stat_value = f.ppf(confidence_level, df1, df2)

print(f_stat_value)
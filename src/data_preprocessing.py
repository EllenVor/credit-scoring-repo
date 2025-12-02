import pandas as pd
import numpy as np

def replace_by_mean_2(cols):
    MonthlyIncome=cols[0]
    DebtRatio = cols[1]
    mean_value=26.598777445397225
    if pd.isna(MonthlyIncome):
      return mean_value
    else :
      return DebtRatio

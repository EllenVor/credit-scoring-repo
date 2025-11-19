def replace_by_mean(cols):
    MonthlyIncome =cols[0]
    DebtRatio = cols[1]
    mean_value = 26.598777445
    if pd.isnull(MonthlyIncome):
        return mean_value
    else:
        return DebtRatio
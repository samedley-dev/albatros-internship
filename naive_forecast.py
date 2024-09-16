import pandas as pd

from outlier_processing import processed_df
from data_plotting import visualize

target_var = 'sales'  # the variable to be forecasted

filepath = 'badminton.csv'
df = processed_df(filepath)
weekly_sum = df.groupby(pd.Grouper(key='date', freq='1W')).sum().reset_index()

n = weekly_sum.shape[0]
train_size = int(n * .75)

# preparing a dictionary containing seasonal coefficients
st = [x.week for x in weekly_sum.loc[:, 'date']].index(1)
fn = st + [x.week for x in weekly_sum.loc[st:, 'date']].index(52)
seasonal_change = {
    weekly_sum.loc[i, 'date'].week: weekly_sum.loc[i, target_var] / weekly_sum.loc[i - 1, target_var]
    for i in range(st, fn + 1)
}

# creating a dictionary with the prognosis and 'forecasting'
# by iterating through a for-loop
weekly_prognosis = {
    weekly_sum.loc[i, 'date']: weekly_sum.loc[i, target_var] for i in range(n)
}
for i in range(train_size, n):
    prev_date = weekly_sum.loc[i - 1, 'date']
    curr_date = weekly_sum.loc[i, 'date']
    j = curr_date.week  # j - the week of the current date
    weekly_prognosis[curr_date] = weekly_prognosis[prev_date] * seasonal_change[j]
naive_forecast = list(weekly_prognosis.values())[train_size:]

# plotting the resulting_params
visualize((weekly_sum['date'], weekly_sum[target_var], target_var),
          (weekly_sum.loc[train_size:, 'date'], naive_forecast, f'forecasted {target_var}'),
          title='Naive forecast using the coefficients of seasonal change')

# creating and saving a dataframe storing values from naive_forecast
naive_df = pd.DataFrame(data=naive_forecast, columns=['value'])
naive_df.to_csv('naive_forecast.csv')

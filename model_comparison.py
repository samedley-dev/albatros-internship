import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae

from outlier_processing import processed_df

target_var = 'sales'  # the variable to be forecasted

filepath = 'badminton.csv'
df = processed_df(filepath)
weekly_sum = df.groupby(pd.Grouper(key='date', freq='1W')).sum().reset_index()

n = weekly_sum.shape[0]
train_size = int(n * .75)

naive_forecast = list(pd.read_csv('naive_forecast.csv')['value'])
sarima_forecast = list(pd.read_csv('sarima_forecast.csv')['value'])
prophet_forecast = list(pd.read_csv('prophet_forecast.csv')['value'])

comparison_df = pd.DataFrame(
    {
        'Metric': ['Root MSE', 'MAE', 'MAPE, %'],

        'Naive': [
            np.sqrt(mse(weekly_sum.loc[train_size:, target_var], naive_forecast)),
            mae(weekly_sum.loc[train_size:, target_var], naive_forecast),
            100 * mape(weekly_sum.loc[train_size:, target_var], naive_forecast)
        ],

        'SARIMA': [
            np.sqrt(mse(weekly_sum.loc[train_size:, target_var], sarima_forecast)),
            mae(weekly_sum.loc[train_size:, target_var], sarima_forecast),
            100 * mape(weekly_sum.loc[train_size:, target_var], sarima_forecast)
        ],

        'FB Prophet': [
            np.sqrt(mse(weekly_sum.loc[train_size:, target_var], prophet_forecast)),
            mae(weekly_sum.loc[train_size:, target_var], prophet_forecast),
            100 * mape(weekly_sum.loc[train_size:, target_var], prophet_forecast)
        ]
    }
)

comparison_df = comparison_df.map(lambda x: f'{x:.3f}' if not isinstance(x, str) else x).set_index('Metric')
print(comparison_df)

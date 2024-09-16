import pandas as pd
import numpy as np
import statsmodels.api as sm

from outlier_processing import processed_df
from data_plotting import visualize

target_var = 'sales'  # the variable to be forecasted

filepath = 'badminton.csv'
df = processed_df(filepath)
weekly_sum = df.groupby(pd.Grouper(key='date', freq='1W')).sum().reset_index()

n = weekly_sum.shape[0]
train_size = int(n * .75)

# getting rid of the target_variable
weekly_sum.drop(
    columns=[col for col in ['sales', 'total_sales'] if col != target_var],
    inplace=True
)

# preparing the train/test split
train = weekly_sum.loc[:train_size - 1, :]
test = weekly_sum.loc[train_size:, :]

# training the model and forecasting values
model = sm.tsa.statespace.SARIMAX(np.log(train[target_var]), order=(1, 1, 1),
                                  seasonal_order=(0, 1, 0, 52)).fit(disp=True)
sarima_forecast = np.exp(
    model.predict(start=train.shape[0], end=weekly_sum.shape[0] - 1)
)

# plotting the resulting_params
visualize((weekly_sum['date'], weekly_sum[target_var], target_var),
          (weekly_sum.loc[train_size:, 'date'], sarima_forecast, f'forecasted {target_var}'),
          title='Forecast by Seasonal ARIMA')

# creating and saving a dataframe storing values from sarima_forecast
sarima_df = pd.DataFrame(data=list(sarima_forecast), columns=['value'])
sarima_df.to_csv('sarima_forecast.csv')

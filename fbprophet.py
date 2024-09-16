import pandas as pd
import numpy as np
from pandas import DataFrame
from prophet import Prophet
from mango import Tuner
from scipy.stats import uniform
import holidays
from sklearn.metrics import mean_absolute_percentage_error as mape

from outlier_processing import processed_df
from data_plotting import visualize


def objective(args):
    global train, test, holidays_df

    params_evaluated = []
    results = []

    for params in args:
        m = Prophet(growth='linear', holidays=holidays_df, **params)

        m.fit(train)
        forecast = m.predict(test)
        error = mape(test['y'], forecast['yhat'])

        params_evaluated.append(params)
        results.append(error)

    return params_evaluated, results


target_var = 'sales'  # the variable to be forecasted

filepath = 'badminton.csv'
df = processed_df(filepath)
weekly_sum = df.groupby(pd.Grouper(key='date', freq='1W')).sum().reset_index()

n = weekly_sum.shape[0]
train_size = int(n * .75)

# getting rid of the target variable
weekly_sum.drop(
    columns=[col for col in ['sales', 'total_sales'] if col != target_var],
    inplace=True
)

# renaming the 'date' and 'total_sales' columns for the back-end of
# FB Prophet to run
weekly_sum.rename(columns={'date': 'ds', target_var: 'y'}, inplace=True)

# creating a dataframe consisting of Russian financial holidays as
# a fbprophet hyperparameter
holidays_df: DataFrame = pd.DataFrame(
    {
        'holiday': 'financial_holiday',
        'ds': [pd.to_datetime(date) for date in holidays.financial_holidays('RU', years=[2023, 2024])]
    }
)

# preparing the train/test split
train = weekly_sum.loc[:train_size - 1, :]
test = weekly_sum.loc[train_size:, :]

# hyperparameter optimization
param_space = dict(
    seasonality_prior_scale=uniform(.01, 10), changepoint_prior_scale=uniform(.001, .5),
    holidays_prior_scale=uniform(.01, 10), seasonality_mode=['additive', 'multiplicative'],
    yearly_seasonality=[False, True]
)
tuner = Tuner(param_space, objective, {'initial_random': 10, 'num_iteration': 100})
resulting_params = tuner.minimize()

# training the model and forecasting values
model = Prophet(holidays=holidays_df, **resulting_params['best_params'])
model.fit(train)
prophet_forecast = model.predict(test)

# plotting the resulting_params
visualize((weekly_sum['ds'], weekly_sum['y'], target_var),
          (prophet_forecast['ds'], prophet_forecast['yhat'], f'forecasted {target_var}'),
          title='Forecast by FaceBook Prophet')

# creating and saving a dataframe storing values from prophet_forecast
prophet_df = pd.DataFrame(data=list(prophet_forecast['yhat']), columns=['value'])
prophet_df.to_csv('prophet_forecast.csv')

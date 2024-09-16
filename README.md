# Стажировка в компании "Альбатрос"
Целью стажировки является разработка программы, которая позволит компании минимизировать убытки, связанные с недопоставками и дефицитом товаров. Ключевая задача проекта заключается в анализе текущих товарных запасов и прогнозировании потребностей, чтобы предотвратить дефицит. Разработанное решение будет автоматически рассчитывать оптимальное количество товаров на основе исторических данных и текущих рыночных трендов, что обеспечит компании эффективное управление закупками и повысит качество обслуживания клиентов.

## Личное участие
В составе команды аналитиков отдела продаж мои обязанности включали участие в очищении данных от статистических выбросов в соответствии с установленными стандартами. Я также принимал участие в разработке и применении статистических моделей, алгоритмов машинного обучения и нейронных сетей для прогнозирования продаж. Совместно с коллегами мы занимались валидацией и оценкой качества датасетов для повышения точности прогнозов, а также отбором наиболее эффективных моделей на основе метрик точности.


Кроме того, я был задействован в анализе трендов и выявлении паттернов в данных, что способствовало улучшению прогностических моделей. В рамках команды мы занимались визуализацией результатов анализа, подготовкой отчетов и автоматизацией процессов обработки данных и прогнозирования. Важным аспектом нашей работы было взаимодействие с другими отделами для интеграции данных, а также адаптация моделей к изменениям рыночной среды.

## Рабочий процесс

### 1. Оценка качества датасетов для построения прогнозов

Задача:
* Оценить датасет на наличие дубликатов.
* Проверить, нет ли разрывов в данных, т.е. промежутков времени, о которых мы ничего не знаем.

Если в датасете есть хотя бы один дубликат или разрыв, то он считается невалидным. 

Фрагмент кода из файла `data_validation.py`, реализующий функцию `verify(df)` для установления валидности датасета:

```python
def verify(df):
    n = df.shape[0]  # number of rows
    is_valid = True
    for i in range(1, n):
        prev = pd.Timestamp(df.loc[i - 1, 'date'])
        curr = pd.Timestamp(df.loc[i, 'date'])

        # if the previous date recorded in the dataframe was more than a day before
        #  the current date, then there's a gap
        if (curr - prev) > pd.Timedelta(days=1):
            is_valid = False

        # if the previous date recorded is the same as the current date, it means
        # that there are repetitions in the dataframe
        if curr == prev:
            is_valid = False

    return is_valid
```

### 2. Визуализация данных

Для упрощения процесса визуализации данных была реализована функция `visualize(*args, title=None, method=plt.plot)`, способная выводить графики нескольких наборов данных (`*args`) на ондой фигуре, под названием `title`, построенных при помощи указанного метода `method`, по умолчанию равного `matplotlib.pyplot.plot`. 

Фрагмент кода из файла `data_plotting.py`, содержащий реализацию данной функции:

```python
def visualize(*args, title=None, method=plt.plot):
    plt.figure(figsize=(10, 5))
    for (xData, yData, name) in args:
        method(xData, yData, label=name)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

### 3. Обработка статистических выбросов

В данных могут иметься статистические выбросы. Для их обработки была реализована функция `processed_df(filename)`, возвращающая пандас-датафрейм, содержащий данные о продажах, очищенные от выбросов. 

Ниже приведен код из файла `outlier_processing.py`, реализующий работу данной функции.

```python
def processed_df(filename):
    df = prepare_df(filename)

    # the initial share of preprocessed sales column relative to total_sales column
    initial_share = df['sales'].mean() / df['total_sales'].mean()
    # adding a column that identifies outliers within sales column
    df['is_outlier'] = df['sales'] < (initial_share * df['total_sales'] * 2 / 3)

    # dataframe consisting of non-outliers
    df_good = df.query('is_outlier == False')

    # here we're trying to estimate the pseudooptimal share, which is the
    # most optimal share within those we have, and yet not necessarily is
    # the most optimal share overall;
    # it'll be used as the initial guess for finding the optimal share

    shares = df_good['sales'] / df_good['total_sales']
    min_error, pseudooptimal_share = float('inf'), None
    for share in shares:
        if min_error > mape(df_good['sales'], df_good['total_sales'] * share):
            min_error = mape(df_good['sales'], df_good['total_sales'] * share)
            pseudooptimal_share = share

    # calculating the optimal share
    optimal_share = minimize(lambda x: mape(df_good['sales'], df_good['total_sales'] * x),
                             x0=pseudooptimal_share).x[0]

    df['sales'] = pd.Series(
        [[df.loc[i, 'sales'], df.loc[i, 'total_sales'] * optimal_share * 1.25][int(df.loc[i, 'is_outlier'])]
         for i in range(df.shape[0])]
    )
    df.drop(columns=['is_outlier'], inplace=True)

    return df
```

### 4. Прогнозирование

Прогнозирование будет совершаться понедельно. Фрагмент кода, устанавливающий, по какому датафрейму мы будем прогнозировать значения `target_var` - нашей целевой переменной:

```python
target_var = 'sales'  # the variable to be forecasted

filepath = 'badminton.csv'
df = processed_df(filepath)
weekly_sum = df.groupby(pd.Grouper(key='date', freq='1W')).sum().reset_index()

n = weekly_sum.shape[0]
train_size = int(n * .75)
```

#### 4.1. Наивный метод

Предполагаем, что в этом году сумма продаж на текущей неделе будет относиться к сумме продаж на предыдущей неделе точно так же, как в прошлом году. На этом строим прогноз. Фрагмент кода из файла `naive_forecast.py` приведен ниже.

```python
# preparing a dictionary containing seasonal coefficients
st = [x.week for x in weekly_sum.loc[:, 'date']].index(1)
fn = st + [x.week for x in weekly_sum.loc[st:, 'date']].index(52)
seasonal_change = {
    weekly_sum.loc[i, 'date'].week: weekly_sum.loc[i, target_var] / weekly_sum.loc[i - 1, target_var]
    for i in range(st, fn + 1)
}

# creating a dictionary with the prognosis and 'forecasting'
# by iterating through a for-loop
weekly_prognosis = {weekly_sum.loc[i, 'date']: weekly_sum.loc[i, target_var] for i in range(n)}
for i in range(train_size, n):
    prev_date = weekly_sum.loc[i - 1, 'date']
    curr_date = weekly_sum.loc[i, 'date']
    j = curr_date.week  # j - the week of the current date
    weekly_prognosis[curr_date] = weekly_prognosis[prev_date] * seasonal_change[j]

naive_forecast = list(weekly_prognosis.values())[train_size:]

# creating and saving a dataframe storing values from naive_forecast
naive_df = pd.DataFrame(data=naive_forecast, columns=['value'])
naive_df.to_csv('naive_forecast.csv')
```

#### 4.2. SARIMA

Фрагмент из файла `sarima_forecast.py`, реализующий прогнозирование продаж моделью Seasonal ARIMA:

```python
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

# creating and saving a dataframe storing values from sarima_forecast
sarima_df = pd.DataFrame(data=list(sarima_forecast), columns=['value'])
sarima_df.to_csv('sarima_forecast.csv')
```

#### 4.3. Facebook Prophet

Использование модели Facebook Prophet ограничилась не только обучением модели и предсказанием значений целевой переменной, но и подбором параметров, при которых модель будет получать наиболее точные прогнозы. 

Для этого была реализована функция `objective(args)`, на основе которой была реализована оптимизацию гиперпараметров модели Facebook Prophet:

```python
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
```

Фрагмент из файла `prophet_forecast.py`, реализующий прогнозирование продаж модели Facebook Prophet с оптимизированными гиперпараметрами (при помощи байесовской оптимизации, реализованной методом `mango.Tuner().minimizer())`, приведен ниже.

```python
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

# creating and saving a dataframe storing values from prophet_forecast
prophet_df = pd.DataFrame(data=list(prophet_forecast['yhat']), columns=['value'])
prophet_df.to_csv('prophet_forecast.csv')
```

## 5. Сравнение моделей

Итогом нашей работы было сравнение моделей прогнозирования по трем метрикам:
* Mean Absolute Error;
* Root-Mean-Square Error;
* Mean Absolute Percentage Error.

Реализовано это было в коде программы `model_comparison.py`, фрагмент из которой приведен ниже.

```python
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

comparison_df = comparison_df.map(
    lambda x: f'{x:.3f}' if not isinstance(x, str) else x
).set_index('Metric')
```

Используя данную таблицу, мы подобрали модель, отличившуюся точностью своих прогнозов больше других, - Facebook Prophet.

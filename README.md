# Стажировка в компании "Альбатрос"
Целью стажировки является разработка программы, которая позволит компании минимизировать убытки, связанные с недопоставками и дефицитом товаров. Ключевая задача проекта заключается в анализе текущих товарных запасов и прогнозировании потребностей, чтобы предотвратить дефицит. Разработанное решение будет автоматически рассчитывать оптимальное количество товаров на основе исторических данных и текущих рыночных трендов, что обеспечит компании эффективное управление закупками и повысит качество обслуживания клиентов.

## Личное участие
В составе команды аналитиков отдела продаж мои обязанности включали участие в очищении данных от статистических выбросов в соответствии с установленными стандартами. Я также принимал участие в разработке и применении статистических моделей, алгоритмов машинного обучения и нейронных сетей для прогнозирования продаж. Совместно с коллегами мы занимались валидацией и оценкой качества датасетов для повышения точности прогнозов, а также отбором наиболее эффективных моделей на основе метрик точности.


Кроме того, я был задействован в анализе трендов и выявлении паттернов в данных, что способствовало улучшению прогностических моделей. В рамках команды мы занимались визуализацией результатов анализа, подготовкой отчетов и автоматизацией процессов обработки данных и прогнозирования. Важным аспектом нашей работы было взаимодействие с другими отделами для интеграции данных, а также адаптация моделей к изменениям рыночной среды.

## Аналитика

### 1. Валидация и оценка качества датасетов для построения прогнозов

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

#### 4.1. Наивный метод

Предполагаем, что в этом году сумма продаж на текущей неделе будет относиться к сумме продаж на предыдущей неделе точно так же, как в прошлом году. На этом строим прогноз. Фрагмент кода из файла `naive_forecast.py` приведен ниже.

```python
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
```

#### 4.2. SARIMA

#### 4.3. Facebook Prophet

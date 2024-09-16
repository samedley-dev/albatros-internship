# Стажировка в компании "Альбатрос"
Целью стажировки является разработка программы, которая позволит компании минимизировать убытки, связанные с недопоставками и дефицитом товаров. Ключевая задача проекта заключается в анализе текущих товарных запасов и прогнозировании потребностей, чтобы предотвратить дефицит. Разработанное решение будет автоматически рассчитывать оптимальное количество товаров на основе исторических данных и текущих рыночных трендов, что обеспечит компании эффективное управление закупками и повысит качество обслуживания клиентов.

## Личное участие
В составе команды аналитиков отдела продаж мои обязанности включали участие в очищении данных от статистических выбросов в соответствии с установленными стандартами. Я также принимал участие в разработке и применении статистических моделей, алгоритмов машинного обучения и нейронных сетей для прогнозирования продаж. Совместно с коллегами мы занимались валидацией и оценкой качества датасетов для повышения точности прогнозов, а также отбором наиболее эффективных моделей на основе метрик точности.

Кроме того, я был задействован в анализе трендов и выявлении паттернов в данных, что способствовало улучшению прогностических моделей. В рамках команды мы занимались визуализацией результатов анализа, подготовкой отчетов и автоматизацией процессов обработки данных и прогнозирования. Важным аспектом нашей работы было взаимодействие с другими отделами для интеграции данных, а также адаптация моделей к изменениям рыночной среды.

## Аналитика

### 1. Валидация и оценка качества датасетов для построения прогнозов.

Задача:
* Оценить датасет на наличие дубликатов.
* Проверить, нет ли разрывов в данных, т.е. промежутков времени, о которых мы ничего не знаем.

Если в датасете есть хотя бы один дубликат или разрыв, то он считается невалидным. Для установления валидности датасета используется функция `verify` из файла `data_validation.py`.
Фрагмент кода:


```
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

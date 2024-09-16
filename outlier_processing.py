import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy.optimize import minimize


def prepare_df(filename):
    df = pd.read_csv(filename)

    # changing the names of the header columns
    df.rename(
        columns={'startDate': 'date', 'sellerMetric.value': 'sales', 'platformMetric.value': 'total_sales'},
        inplace=True
    )

    # changing the elements of the date column to datetime
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'])

    return df


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

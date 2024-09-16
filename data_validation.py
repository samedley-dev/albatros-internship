import pandas as pd


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


def verify(df):
    n = df.shape[0]  # the number of rows
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

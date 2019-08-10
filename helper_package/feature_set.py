"""
Function(s) to obtain mu and sigma feature set
"""

import pandas as pd
import sys


def get_feature_set(df1):
    """
    Given a DataFrame with 'Year_Week' and 'Return' values for a stock,
    this function returns a new DataFrame (df3) with new columns for
    'Mean_Return' (mu) and for 'Std_Return' (standard deviation, sigma)
    :param df1: Stock DataFrame
    :return df3: DataFrame with Mean_Return and Std_Return columns
    """

    try:
        mean_return_df = df1[['Year_Week', 'Return']].groupby(
            'Year_Week').mean().reset_index()
        mean_return_df.rename(columns={'Return': 'Mean_Return'}, inplace=True)

        std_return_df = df1[['Year_Week', 'Return']].groupby(
            'Year_Week').std().reset_index()
        std_return_df.rename(columns={'Return': 'Std_Return'}, inplace=True)

        df2 = pd.merge(mean_return_df, std_return_df, on='Year_Week')
        df3 = pd.merge(df1, df2, on='Year_Week')

        # replace 'None' value with 0 for Std_Return
        values = {'Std_Return': 0.0}
        df3.fillna(value=values, inplace=True)

    except KeyError as ke:
        print(ke)
        sys.exit()

    return df3

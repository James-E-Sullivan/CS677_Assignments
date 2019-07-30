import pandas as pd
import os
import numpy as np
from Assignment2 import week_labeling as wl


def get_ticker_df():

    ticker = 'SBUX'
    input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
    filename = os.path.join(input_dir, 'logistic_regression_gradient_descent.pdf')
    ticker_file = os.path.join(input_dir, ticker + '.csv')
    data = pd.read_csv(ticker_file)

    # assign labels using week_labeling formula from assignment 2
    data = wl.assign_labels(data)

    # Set Return to Percent Return
    data['Return'] = data['Return'].apply(lambda e: e * 100)

    data['Class'] = data['Color'].apply(lambda a: 1 if a == 'Green' else 0)
    data = get_feature_set(data)
    return data


def get_feature_set(df1):

    mean_return_df = df1[['Year_Week', 'Return']].groupby('Year_Week').mean().reset_index()
    mean_return_df.rename(columns={'Return': 'Mean_Return'}, inplace=True)

    std_return_df = df1[['Year_Week', 'Return']].groupby('Year_Week').std().reset_index()
    std_return_df.rename(columns={'Return': 'Std_Return'}, inplace=True)

    df2 = pd.merge(mean_return_df, std_return_df, on='Year_Week')
    df3 = pd.merge(df1, df2, on='Year_Week')

    return df3



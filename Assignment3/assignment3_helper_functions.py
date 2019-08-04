import pandas as pd
import os
import numpy as np
import sys
from Assignment2 import week_labeling as wl


def get_ticker_df():
    """
    Obtains ticker data and reads it into a DataFrame; assigns labels
    to each trading week; sets returns to percent returns; obtains a
    binary label for labels; fills 'None' of Std_Return with 0; and returns
    modified DataFrame.
    :return data: DataFrame of stock data, slightly modified
    """

    try:
        ticker = 'SBUX'
        input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
        ticker_file = os.path.join(input_dir, ticker + '.csv')
        data = pd.read_csv(ticker_file)

    except FileNotFoundError as e:
        print(e)
        sys.exit()

    # assign labels using week_labeling formula from assignment 2
    data = wl.assign_labels(data)

    # Set Return to Percent Return
    data['Return'] = data['Return'].apply(lambda e: e * 100)

    # create a binary column (1 or 0) corresponding to green/red
    data['Class'] = data['Color'].apply(lambda a: 1 if a == 'Green' else 0)
    data = get_feature_set(data)

    # Replace Std_Return NaN values with 0
    data.fillna(value={'Std_Return': 0}, inplace=True)

    return data


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

    except KeyError as ke:
        print(ke)
        sys.exit()

    return df3


def get_acc(vector):
    """
    Returns 1's for accurate prediction, otherwise returns 0
    :param vector: vector containing (actual, predicted) values
    :return: 1 or 0
    """
    if vector[0] == vector[1]:
        return 1
    else:
        return 0


def tp(vector):
    """
    Compares actual vs predicted color values.
    Returns 1 only if result is a true positive
    :param vector: vector containing (actual, predicted) values
    :return: 1 or 0
    """
    if vector[0] is 'Green' and vector[1] is 'Green':
        return 1
    else:
        return 0


def fp(vector):
    """
    Compares actual vs predicted color values.
    Returns 1 only if result is a false positive
    :param vector: vector containing (actual, predicted) values
    :return: 1 or 0
    """
    if vector[0] is 'Red' and vector[1] is 'Green':
        return 1
    else:
        return 0


def tn(vector):
    """
    Compares actual vs predicted color values.
    Returns 1 only if result is a true negative
    :param vector: vector containing (actual, predicted) values
    :return: 1 or 0
    """
    if vector[0] is 'Red' and vector[1] is 'Red':
        return 1
    else:
        return 0


def fn(vector):
    """
    Compares actual vs predicted color values.
    Returns 1 only if result is a false negative
    :param vector: vector containing (actual, predicted) values
    :return: 1 or 0
    """
    if vector[0] is 'Green' and vector[1] is 'Red':
        return 1
    else:
        return 0

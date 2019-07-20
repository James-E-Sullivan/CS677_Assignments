# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os
import math
import numpy as np
import pandas as pd

#
ticker='SBUX'
input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
ticker_file = os.path.join(input_dir, ticker + '.csv')
output_file = os.path.join(input_dir, ticker + '_updated.csv')

df = pd.read_csv(ticker_file)   # DataFrame of SBUX ticker

mean_return = 100.0 * df['Return'].mean()   # mean of values in 'Return' column
std_return = 100.0 * df['Return'].std()     # std of values in 'Return' column

low_bound = mean_return - 2 * std_return
upper_bound = mean_return + 2 * std_return


# please fix the computation of last digit <--- what does this mean?
df['Open'] = df['Open'].round(2)

# obtains int(last char in str)
df['last_digit'] = df['Open'].apply(lambda x: int(str(x)[-1]))
df['count'] = 1
df['year_count'] = 1

digits_total = df.groupby(['last_digit'])['count'].sum()
actual = 100 * digits_total / len(df)

predicted = np.array([10,10,10,10,10,10,10,10,10,10])

'''
max_abs_error = np.max(abs(actual - predicted)).round(2)
median_abs_error = np.median(abs(actual - predicted)).round(2)
mean_abs_error = np.mean(abs(actual - predicted)).round(2)
mse = np.mean((actual - predicted)**2).round(2)  # root mean squared error
'''


def max_abs_error(a, p):
    """
    Returns Max Absolute Error of an actual dataset compared to predicted
    :param a: actual values
    :param p: predicted values
    """

    return np.max(abs(a - p).round(2))


def median_abs_error(a, p):
    """
    Returns Median Absolute Error of actual dataset compared to predicted
    :param a: actual values
    :param p: predicted values
    """

    return np.median(abs(a - p)).round(2)


def mean_abs_error(a, p):
    """
    Returns Mean Absolute Error of actual dataset compared to predicted
    :param a: actual values
    :param p: predicted values
    """

    return np.mean(abs(a - p)).round(2)


def mse(a, p):
    """
    Returns the Root Mean Square Error
    :param a: actual values
    :param p: predicted values
    """

    return np.mean((a - p)**2).round(2)


# new code below

yearly_digits_total = df.groupby(['Year', 'last_digit'])['count'].sum()
yearly_actual = 100 * yearly_digits_total / df.groupby(['Year'])['year_count'].sum()

year_list = df['Year'].unique()

yearly_actual['max'] = max_abs_error(yearly_actual[2014], predicted)


error_data = {}  # to hold error calculations, by year


for year in year_list:

    try:
        error_data[year] = [max_abs_error(yearly_actual[year], predicted),
                            median_abs_error(yearly_actual[year], predicted),
                            mean_abs_error(yearly_actual[year], predicted),
                            mse(yearly_actual[year], predicted)]
    except KeyError as e:
        print(e)

# create error calculation df
error_df = pd.DataFrame(error_data, index=['max_absolute_error',
                                           'median_absolute_error',
                                           'mean_absolute_error',
                                           'RSME'])

# print transposed (swap axes) error calculation table
print(error_df.T)

# -*- coding: utf-8 -*-
"""
Original @author: epinsky
Created on Mon Nov  5 14:37:29 2018
Updated July 17, 2019
Original script reads ticker file and constructs a Pandas DataFrame

James Sullivan
Class: CS677 - Summer 2
Date: 7/20/2019
Homework 2: Last Digit Distribution Questions 1-3

1. What is the most frequent digit?
2. What is the least frequent digit?
3. Compute the following 4 error metrics for your data:
    (a) max absolute error
    (b) median absolute error
    (c) mean absolute error
    (d) root mean squared error (RMSE)

    Summarize findings in a table (*1* row for each year) and discuss
    your results
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


# ---------- James Sullivan's Code Below ----------

# Print digits with max and min values to console
print("\nQuestion 1: What is the most frequent digit?")
print(digits_total.idxmax())

print("\nQuestion 2: What is the least frequent digit?")
print(digits_total.idxmin())


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


yearly_digits_total = df.groupby(['Year', 'last_digit'])['count'].sum()
yearly_actual = 100 * yearly_digits_total / df.groupby(
                ['Year'])['year_count'].sum()

year_list = df['Year'].unique()

error_data = {}  # to hold error calculations, by year


for year in year_list:

    try:
        # calculate error values and place into a dict
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
print("\nQuestion 3")
print(error_df.T)

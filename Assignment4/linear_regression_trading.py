"""
James Sullivan
Class: CS677 - Summer 2
Date: 8/3/2019
Homework: Day Trading with Linear Regression Questions 1-6

Takes a window of W days, and given the adj_close prices of P1, P2, PW, for
days t=1, 2,..., W, we estimate the closing price PW+1 for day W+1 using
linear regression. We choose W based on the profitability of our strategy.

*** This assignment was completed while following the assignment instructions
in its original format ***

Questions

1. Take W=5,6,...,30 and consider data for yr 1. For each W in specified range,
   compute avg profit/loss per trade and plot it: (x=w values, y= profit/loss).
   What is the optimal value W* of W?

2. Use the value of W* from yr 1 and consider yr 2. For every day in yr 2,
   take the previous W* days, compute linear regression and compute the value
   of r^2 for yr 2. What is the avg r^2? How well does it explain price
   movements?

3. Take optimal W* from yr 1 and use it to implement the window trading
   strategy (found in window_strategy.py). How many "long position" and
   "short position" transactions did you have in yr 2?

4. What is the avg profit/loss per "long position" trade and per "short
   position" trade in yr 2?

5. What is the avg number of days for long position and short position
   transactions in year 2?

6. Are these results very different from those in year 1 for this value of W?
"""

import pandas as pd
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from helper_package import assignment4_helper_functions as hf
import Assignment4.window_strategy as ws

hf.show_more_df()   # allow df output to display more rows/columns
df = hf.fix_column_names(hf.get_ticker_df())     # get DataFrame of ticket file


def create_w_list(w_min, w_max, step=1):
    """
    Create list of w values (used as 'windows' for estimation)
    :param w_min: Minimum w value
    :param w_max: maximum w value
    :param step: step for w values in list
    :return output_list: list of w values
    """
    output_list = []    # list of w values
    for i in range(w_min, w_max + 1):
        output_list.append(i)
        i += step

    return output_list


def adj_close_lin_reg(data, w):
    """
    Takes Stock DataFrame with daily adjusted close price values, returns
    Stock DataFrame with profit and r^2 value columns.
    :param data: Stock DataFrame with daily adjusted close price values.
    :param w: The number of days (the window) used to create linear
    regression line and predict the adj_close price of day w+1
    :return: data DF with additional column values, including r^2
     and actual profit values.
    """

    # get Adj_Close values from data
    data_set = data.Adj_Close.values

    # first window end index
    end_w = copy.copy(w)    # copy to avoid resetting w's value

    pred_price_list = []   # list to hold predicted price
    r_squared_list = []    # list to hold r^2 values

    # append None for indices less than w
    for i in range(end_w):
        pred_price_list.append(None)
        r_squared_list.append(None)

    # loops through data set, less the value of w
    for start_w in range((len(data_set) - w)):

        x = np.arange(start_w, end_w)   # w values on x-axis
        y = data_set[start_w: end_w]    # adj_close values w/in window

        x_2 = x[:, np.newaxis]          # flip axis of array x
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(x_2, y)

        r_squared = lin_reg.score(x_2, y)   # r^2 value
        slope = lin_reg.coef_[0]            # slope
        intercept = lin_reg.intercept_      # y-intercept

        day = end_w # the "day #" of the dataset (x value to predict y)
        pred_return = (slope * day) + intercept  # y=mx+b

        pred_price_list.append(pred_return)     # add predicted return to list
        r_squared_list.append(r_squared)        # add r^2 value to list

        end_w += 1  # increment end of window

    column_name = 'w_' + str(w)             # column names

    # add predicted prices
    data['next_pred_price'] = pred_price_list

    # shift pred prices so the column = next day's predicted price
    data.next_pred_price = data.next_pred_price.shift(periods=-1)

    # add r^2 values
    data['r_squared'] = r_squared_list

    data_p_l = ws.window_strategy(data, w)  # apply window strategy to data
    data.merge(data_p_l.profit_loss)  # merge profit/loss values with data

    # rename profit_loss column name to reflect the w_value
    data.rename(columns={'profit_loss': column_name + '_profit'}, inplace=True)
    data.drop(columns='next_pred_price', inplace=True)

    return data


def pred_multiple_w(data, w_values):
    """
    Predicts profit loss of several w values
    :param data: Stock DataFrame with daily adj_close values
    :param w_values: list of w values
    :return: data with profit/loss columns for each w in w_values
    """
    for w_value in w_values:
        new_df = adj_close_lin_reg(data, w_value)
        data.merge(new_df)

    return data


def find_mean_profit(data, w_values):
    """
    Given data with columns of profit/loss for a given set of w_values,
    function returns a list of the mean profit values (corr. to w_values)
    :param data: Stock DataFrame with profit/loss values
    :param w_values: List of w_values
    :return: Returns list of mean profits corresponding to w_values
    """
    mean_profit_list = []
    for w in w_values:
        column_name = 'w_' + str(w) + '_profit'
        mean_profit = data[column_name].mean()
        mean_profit_list.append(mean_profit)
    return mean_profit_list


def get_optimal_w(data, w_values):
    """
    Given a stock DataFrame with daily adjusted close values, and a list
    of w values, returns the w value (optimal_w) with best profits.
    :param data: DataFrame with daily adjusted close values
    :param w_values: List of chosen w values
    :return optimal_w: w value that gives best profits
    """
    data_with_profit = pred_multiple_w(data, w_list)
    mean_profits = find_mean_profit(data_with_profit, w_values)

    plt.scatter(w_values, mean_profits)
    plt.grid(True)
    plt.title('mean profits for each w value')
    plt.xlabel('w')
    plt.ylabel('mean profits')
    plot_dir = '../plots'
    output_file = os.path.join(plot_dir, 'w_mean_profit' + '.pdf')
    plt.savefig(output_file)

    mean_profit_df = pd.DataFrame(columns={'w_value', 'mean_profit'})
    mean_profit_df.w_value = w_values
    mean_profit_df.mean_profit = mean_profits

    max_profit = mean_profit_df['mean_profit'].max()

    # this seems like a roundabout way of getting the optimal w, but it works
    optimal_w = mean_profit_df.loc[mean_profit_df.mean_profit == max_profit].w_value.values[0]

    return optimal_w


def short_or_long_trx(vector):
    """
    Returns 1 if there is a transaction and a short/long position is held
    :param vector: vector[0] is transaction and vector[1] position
    :return: 1 or 0
    """
    if vector[0] is not None and vector[1] == 1:
        return 1
    else:
        return 0


def short_or_long_profit(vector):
    """
    Returns profit value if the position vector is 1, else 0. Used to fill in
    column with profit/loss values for one transaction type.
    :param vector: vector[0] is profit, vector[1] is position type
    :return: profit/loss or 0
    """
    if vector[1] == 1:
        return vector[0]
    else:
        return 0


# creates subsets of stock DataFrame for 2017 and 2018
df_2017 = df[df.Year == 2017].reset_index(drop=True)
df_2018 = df[df.Year == 2018].reset_index(drop=True)

w_minimum = 5       # minimum w value
w_maximum = 30      # maximum w value
w_list = create_w_list(w_minimum, w_maximum)    # list of w values

# answers assignment questions using 2017 as yr1 and 2018 as yr2
if __name__ == '__main__':

    # ---------- Question 1 ----------
    optimal_w_value = get_optimal_w(df_2017, w_list)
    print('\n__________Question 1__________')
    print('Optimal W value (for W=5, 6,..., 30): ', optimal_w_value)

    # ---------- Question 2 ----------
    df_2018_lin_reg = adj_close_lin_reg(df_2018, optimal_w_value)
    mean_r_squared = df_2018_lin_reg.r_squared.mean()

    print('\n__________Question 2__________')
    print('2018 Mean r^2 value: ', round(mean_r_squared, 6))

    # ---------- Question 3 ----------

    profit_col_name = 'w_' + str(10) + '_profit'

    # column to track short transactions
    df_2018_lin_reg['short_trx'] = df_2018_lin_reg[[
        profit_col_name, 'short_count']].apply(short_or_long_trx, axis=1)

    # column to track long transactions
    df_2018_lin_reg['long_trx'] = df_2018_lin_reg[[
        profit_col_name, 'long_count']].apply(short_or_long_trx, axis=1)

    # calculate number of long and short transactions
    short_transactions = df_2018_lin_reg.short_trx.sum()
    long_transactions = df_2018_lin_reg.long_trx.sum()

    print('\n__________Question 3__________')
    print('Number of long position transactions: ', long_transactions)
    print('Number of short position transactions: ', short_transactions)

    # ---------- Question 4 ----------

    # column to track short profits
    df_2018_lin_reg['short_profit'] = df_2018_lin_reg[[
        profit_col_name, 'short_trx']].apply(short_or_long_profit, axis=1)

    # column to track long profits
    df_2018_lin_reg['long_profit'] = df_2018_lin_reg[[
        profit_col_name, 'long_trx']].apply(short_or_long_profit, axis=1)

    # calculate average short & long transaction profits
    short_profit_avg = df_2018_lin_reg.short_profit.sum() / short_transactions
    long_profit_avg = df_2018_lin_reg.long_profit.sum() / long_transactions

    print('\n__________Question 4__________')
    print('Average profit/loss per long position trade: ', round(long_profit_avg, 4))
    print('Average profit/loss per short position trade: ', round(short_profit_avg, 4))

    # ---------- Question 5 ----------

    # calculate number of days that short and long positions were held
    short_days = df_2018_lin_reg.short_count.sum()
    long_days = df_2018_lin_reg.long_count.sum()

    # calc avg days per short and long transaction
    short_avg_days = short_days / short_transactions
    long_avg_days = long_days / long_transactions

    print('\n__________Question 5__________')
    print('Average number of days for long position: ', round(long_avg_days, 6))
    print('Average number of days for short position: ', round(short_avg_days, 6))

    # ---------- Question 6 ----------
    # reset df_2017
    df_2017 = df[df.Year == 2017].reset_index(drop=True)
    df_2017_lin_reg = adj_close_lin_reg(df_2017, optimal_w_value)

    # calculate mean r^2 value for 2017
    mean_r_square_2017 = df_2017_lin_reg.r_squared.mean()

    # column to track short transactions
    df_2017_lin_reg['short_trx'] = df_2017_lin_reg[[
        profit_col_name, 'short_count']].apply(short_or_long_trx, axis=1)

    # column to track long transactions
    df_2017_lin_reg['long_trx'] = df_2017_lin_reg[[
        profit_col_name, 'long_count']].apply(short_or_long_trx, axis=1)

    # calculate number of short and long transactions for 2017
    short_trx_2017 = df_2017_lin_reg.short_trx.sum()
    long_trx_2017 = df_2017_lin_reg.long_trx.sum()

    # column to track short profits
    df_2017_lin_reg['short_profit'] = df_2017_lin_reg[[
        profit_col_name, 'short_trx']].apply(short_or_long_profit, axis=1)

    # column to track long profits
    df_2017_lin_reg['long_profit'] = df_2017_lin_reg[[
        profit_col_name, 'long_trx']].apply(short_or_long_profit, axis=1)

    # calc avg profits for long and short transactions
    short_profit_avg_2017 = df_2017_lin_reg.short_profit.sum() / short_trx_2017
    long_profit_avg_2017 = df_2017_lin_reg.long_profit.sum() / long_trx_2017

    # calculate days at short and long positions
    short_days_2017 = df_2017_lin_reg.short_count.sum()
    long_days_2017 = df_2017_lin_reg.long_count.sum()

    # calculate avg no. of short and long days per transaction
    short_avg_days_2017 = short_days_2017 / short_trx_2017
    long_avg_days_2017 = long_days_2017 / long_trx_2017

    # output same answers as 2018, without separating into multiple questions
    # could have made function to output both yr1 and yr2 this way
    print('\n__________Question 6__________')
    print('Results for 2017')
    print('Average r^2 value: ', round(mean_r_square_2017, 6))
    print('Number of long position transactions: ', long_trx_2017)
    print('Number of short position transactions: ', short_trx_2017)
    print('Avg profit/loss per long position trade: ', round(
        long_profit_avg_2017, 4))
    print('Avg profit/loss per short position trade: ', round(
        short_profit_avg_2017, 4))
    print('Avg no. days for long position: ', round(long_avg_days_2017, 6))
    print('Avg no. days for short position: ', round(short_avg_days_2017, 6))

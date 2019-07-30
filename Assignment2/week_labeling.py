"""
James Sullivan
Class: CS677 - Summer 2
Date: 7/23/2019
Homework 2: Week Labeling Questions 1-2

Preface: Take stock data and assign colors for two years (yr 1 & 2). For other
         years, put color "undefined." Assign colors to blocks of 5 days at a
         time, based on a defined rule.

Rule:    If the annualized return of a given 10 day block is greater than or
         equal to 10%, color the block GREEN. IF the annualized return is less
         than 10%, color the block RED.

         10% was chosen because it is the approximate averaged annual return
         of the S&P 500 since its inception, NOT adjusted for inflation.

         Annualized Return = ((1 + cumulative return) ^ (365/days held)) - 1

Strategy:
    1. For 1st "green" week, invest $100 by buys shares of stock at opening
       price of 1st trading day that week.
    2. If next week is "red":
           (a) if you have a position this week (green week), sell your shares
               at the adjusted closing price of last trading day of this week
           (b) if you have no position this week (red week), do nothing
    3. If next week is "green":
           (a) if you have a position this week (green week), you do nothing
           (b) if you have no position this week (red week), you buy shares
               of stock at the opening price of next week (invest ALL money)
    4. Ignore trading costs and assume that you are able to buy or sell at
       open or adjusted closing prices
Questions

1. Implement a buy-and-hold strategy for year 2. Invest $100 on the first
   trading day at opening price and sell at the last trading day at the adjusted
   closing price.

2. Implement a trading strategy based on your labels for year 2 and compare the
   performance with the "buy-and-hold" strategy. Which strategy results in a
   larger amount at the end of the year?
"""

import pandas as pd
import numpy as np
import os


ticker='SBUX'
input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)


""""
try:
    # create new df with cumulative returns for each week
    week_return_df = df[['Year_Week', 'Return']].groupby(
        'Year_Week').sum().reset_index()

    # rename 'Return' to 'Return_Week' and merge back into original df
    week_return_df.rename(columns={'Return': 'Return_Week'}, inplace=True)
    df = pd.merge(df, week_return_df, on='Year_Week')

    # create new df with the length of each week (by days)
    week_size_df = df[['Year_Week', 'Return']].groupby(
        'Year_Week').count().reset_index()

    # rename 'Return' to 'Week_Size' and merge back into original df
    week_size_df.rename(columns={'Return': 'Week_Size'}, inplace=True)
    df = pd.merge(df, week_size_df, on='Year_Week')

    # create new df w/ the date of the first day of each week
    week_start_df = df[['Year_Week', 'Date']].groupby(
        'Year_Week').min().reset_index()
    week_start_df.rename(columns={'Date': 'Week_Start'}, inplace=True)

    # create new df w/ the date of the last day of each week
    week_end_df = df[['Year_Week', 'Date']].groupby(
        'Year_Week').max().reset_index()
    week_end_df.rename(columns={'Date': 'Week_End'}, inplace=True)

    # merge week_start_df and week_end_df, then merge new df into original df
    week_bounds_df = pd.merge(week_start_df, week_end_df, on='Year_Week')
    df = pd.merge(df, week_bounds_df, on='Year_Week')

except KeyError as e:
    print(e)
    print("Key value not found")
"""


def assign_labels(data):

    week_return_data = data[['Year_Week', 'Return']].groupby('Year_Week').sum().reset_index()

    week_return_data.rename(columns={'Return': 'Return_Week'}, inplace=True)
    data = pd.merge(data, week_return_data, on='Year_Week')

    # create new df with the length of each week (by days)
    week_size_data = data[['Year_Week', 'Return']].groupby(
        'Year_Week').count().reset_index()

    # rename 'Return' to 'Week_Size' and merge back into original df
    week_size_data.rename(columns={'Return': 'Week_Size'}, inplace=True)
    data = pd.merge(data, week_size_data, on='Year_Week')

    # create new df w/ the date of the first day of each week
    week_start_data = data[['Year_Week', 'Date']].groupby(
        'Year_Week').min().reset_index()
    week_start_data.rename(columns={'Date': 'Week_Start'}, inplace=True)

    # create new df w/ the date of the last day of each week
    week_end_data = data[['Year_Week', 'Date']].groupby(
        'Year_Week').max().reset_index()
    week_end_data.rename(columns={'Date': 'Week_End'}, inplace=True)

    # merge week_start_df and week_end_df, then merge new df into original df
    week_bounds_data = pd.merge(week_start_data, week_end_data, on='Year_Week')
    data = pd.merge(data, week_bounds_data, on='Year_Week')

    # calculate annualized return for each week
    data['Annual_Return'] = data[['Return_Week', 'Week_Size']].apply(
        annualized_return, axis=1)

    # calculate percent annualized return for each week
    data['Annual_Return_%'] = data['Annual_Return'].apply(lambda x: x * 100)

    # assign color based on percent annualized return value
    data['Color'] = data['Annual_Return_%'].apply(
        lambda x: 'Green' if x >= 10 else 'Red')

    return data


def annualized_return(vec):
    """
    Returns the annualized return value of a given cumulative return over
    a specified length of time. Must be performed on array where 1st col
    is the cumulative return, and the 2nd col is the day length associated
    with the cumulative return value.
    :param vec: Vector from DataFrame
    :return: Annualized Return value
    """

    try:
        r = vec[0]
        length = vec[1]
        return (1 + r) ** (365 / length) - 1

    except KeyError as ke:
        print(ke)
        print('annualized_return() could not find keys')

""""
try:
    # calculate annualized return for each week
    df['Annual_Return'] = df[['Return_Week', 'Week_Size']].apply(
        annualized_return, axis=1)

    # calculate percent annualized return for each week
    df['Annual_Return_%'] = df['Annual_Return'].apply(lambda x: x * 100)

    # assign color based on percent annualized return value
    df['Color'] = df['Annual_Return_%'].apply(
        lambda x: 'Green' if x >= 10 else 'Red')

except ValueError as color_exception:
    print(color_exception)
"""


def buy_and_hold(data):
    """
    Buy stock on 1st day and sell stock on the last day
    :param data: DataFrame of stock data
    :return: Returns the final funds after following strategy
    """
    initial_funds = 100.00  # funds start at $100.00
    initial_price = data.iloc[0]['Open']
    final_price = data.iloc[len(data) - 1]['Adj Close']
    final_funds = final_price * (initial_funds / initial_price) # price * shares
    return final_funds


def color_strategy(data):
    """
    Follows strategy described in the docstring at the top of this file
    :param data: DataFrame of stock data
    :return: Returns the final funds after following strategy
    """

    funds = 100.00  # funds start at $100.00
    shares = 0.0    # shares start at $0.00
    current_color = 'Red'   # initialize color to red


    # get week start date and open value on that date
    week_data = data[['Year_Week', 'Color', 'Week_Start', 'Open']].groupby(
        'Year_Week').first().reset_index()

    # get week end date and adjusted close value on that date
    week_end_data = data[['Year_Week', 'Week_End', 'Adj Close']].groupby(
        'Year_Week').last().reset_index()

    # merge DataFrames w/ week_start and week_end info
    week_data = pd.merge(week_data, week_end_data, on='Year_Week')

    # purchase funds on first day of first week, if 1st week color is green
    if week_data.iloc[0]['Color'] == 'Green':
        shares = funds / data.iloc[0]['Open']
        funds = 0.00
        current_color = 'Green'


    # iterate through weeks
    for i in range(len(week_data)-1):

        next_color = week_data.iloc[i+1]['Color']      # next week's color
        open_price = week_data.iloc[i]['Open']              # open price
        adj_close_price = week_data.iloc[i]['Adj Close']    # close price

        # if there is no price, we should be at last row
        if open_price is None or adj_close_price is None:
            break

        elif next_color == 'Red':

            # want to sell all stocks if current week is Green
            if current_color is 'Green':
                sell_price = adj_close_price
                funds = shares * sell_price
                shares = 0.00

        elif next_color == 'Green':

            # want to buy max amt of stocks if current week is Red
            if current_color is 'Red':
                buy_price = open_price
                shares = funds / buy_price
                funds = 0.00

        current_color = next_color  # set color to next week's color

    # If shares haven't been converted to funds by the end of time period
    if shares > 0:

        # convert shares to funds at final adjusted closing price
        final_value = week_data.iloc[len(week_data)-1]['Adj Close'] * shares

    # all shares already sold
    else:
        final_value = funds

    return final_value




if __name__ == "__main__":

    df = assign_labels(df)
    print(df)

    print("\nQuestion 1")
    # Buy and hold strategy applied to 2015
    buy_and_hold_final = buy_and_hold(df[df['Year'] == 2015]).round(2)
    print("Buy & Hold for 2015: " + "$" + str(buy_and_hold_final))

    print("\nQuestion 2")
    # Color strategy applied to 2015
    color_strat_final = color_strategy(df[df['Year'] == 2015]).round(2)
    print("Color Strategy for 2015: " + "$" + str(color_strat_final))


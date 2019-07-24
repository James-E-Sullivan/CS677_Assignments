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
                                    OR
         Annualized Return = ((1 + cumulative return) ^ (365/5)) - 1

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
#input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
input_dir = r'C:\Users\james.sullivan\Documents\Personal Documents - Temporary (Delete)\BU MET\CS677\Datasets'
ticker_file = os.path.join(input_dir, ticker + '.csv')

df = pd.read_csv(ticker_file)

pd.set_option('display.max_columns', 7)

#df['Return'] = 100.0 * df['Return']     # Percent return

y1_start_date = '2014-01-02'
y1_end_date = '2014-12-31'

y2_start_date = '2015-01-02'
y2_end_date = '2015-12-31'

size_2014 = len(df[df['Year'] == 2014])
size_2015 = len(df[df['Year'] == 2015])

year_list = df['Year'].unique()



year_week_df = df[['Year_Week', 'Return']].groupby('Year_Week').sum().reset_index()
year_week_df.rename(columns={'Return': 'Return_Week'}, inplace=True)
df = pd.merge(df, year_week_df, on='Year_Week')

week_size_df = df[['Year_Week', 'Return']].groupby('Year_Week').count().reset_index()
week_size_df.rename(columns={'Return': 'Week_Size'}, inplace=True)
df = pd.merge(df, week_size_df, on='Year_Week')


df_test = df[['Year_Week', 'Date']].groupby('Year_Week').min().reset_index()
df_test.rename(columns={'Date': 'Week_Start'}, inplace=True)

df_test_2 = df[['Year_Week', 'Date']].groupby('Year_Week').max().reset_index()
df_test_2.rename(columns={'Date': 'Week_End'}, inplace=True)

week_bounds_df = pd.merge(df_test, df_test_2, on='Year_Week')
df = pd.merge(df, week_bounds_df, on='Year_Week')


def annualized_return(vec):
    r = vec[0]
    length = vec[1]
    return (1 + r) ** (365 / length) - 1


df['Annual_Return'] = df[['Return_Week', 'Week_Size']].apply(annualized_return, axis=1)
df['Annual_Return_%'] = df['Annual_Return'].apply(lambda x: x * 100)


df['Color'] = df['Annual_Return_%'].apply(lambda x: 'Green' if x >= 10 else 'Red')


def buy_and_hold(data):

    initial_funds = 100.00  # funds start at $100.00
    initial_price = data.iloc[0]['Open']
    final_price = data.iloc[len(data) - 1]['Adj Close']
    final_funds = final_price * (initial_funds / initial_price)  # price * shares
    return final_funds


def color_strategy(data):

    funds = 100.00  # funds start at $100.00
    shares = 0.0
    current_color = 'Red'   # initialize color to red

    week_data = data[['Year_Week', 'Color', 'Week_Start', 'Open']].groupby('Year_Week').first().reset_index()

    week_end_data = data[['Year_Week', 'Week_End', 'Adj Close']].groupby('Year_Week').last().reset_index()

    week_data = pd.merge(week_data, week_end_data, on='Year_Week')

    if week_data.iloc[0]['Color'] == 'Green':
        shares = funds / data[0]['Open']
        funds = 0.00
        current_color = 'Green'

    for i in range(len(week_data)-1):

        next_color = week_data.iloc[i+1]['Color']
        open_price = week_data.iloc[i]['Open']
        adj_close_price = week_data.iloc[i]['Adj Close']

        if next_color == 'Red':

            if current_color is 'Green':
                sell_price = adj_close_price
                funds = shares * sell_price
                shares = 0.00

        elif next_color == 'Green':

            if current_color is 'Red':
                buy_price = open_price
                shares = funds / buy_price
                funds = 0.00

        """
        print('\n ---------- Iteration', i, '----------')
        print("Current Color: ", current_color)
        print("Next Color: ", next_color)
        print("Open Price: ", open_price)
        print("Close Price: ", adj_close_price)
        print("Funds After: ", funds)
        print("Shares After: ", shares)
        """

        current_color = next_color

    if shares > 0:
        final_value = week_data.iloc[len(week_data)-1]['Adj Close'] * shares

        print("Final value: ", final_value)

    else:
        final_value = funds

    return final_value


print("Buy & Hold for 2015: ", buy_and_hold(df[df['Year'] == 2015]).round(2))
print("Color Strategy for 2015: ", color_strategy(df[df['Year'] == 2015]).round(2))


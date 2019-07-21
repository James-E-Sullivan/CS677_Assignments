"""
James Sullivan
Class: CS677 - Summer 2
Date: 7/20/2019
Homework: Normality of Returns Question # 1-3

1. Compute # days w/ positive and negative returns for each year (of 5)

2. For each year, compute avg daily returns (u) and compute the % of days
   with returns greater than u and the proportion of days with returns
   less than u. Does it change from year to year?

3. For each year, compute the mean and std of daily returns. Compute # days
   that the abs(returns) > 2 std from the mean.
"""

import pandas as pd
import numpy as np
import os

ticker='SBUX'
input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
ticker_file = os.path.join(input_dir, ticker + '.csv')

df = pd.read_csv(ticker_file)
df['Return'] = 100.0 * df['Return']


# ---------- Question 1 ----------
print('\nQuestion 1')
print('Days with positive returns: ', df['Return'].gt(0).sum())
print('Days with negative returns: ', df['Return'].lt(0).sum())


# ---------- Question 2 ----------

# Calculate mean returns for each year
mean_returns = df.groupby(['Year'])['Return'].mean()

# Trade_Days values set to 1 to sum for annual trade day calculation
df['Trade_Days'] = 1
annual_total_trade_days = df.groupby(['Year'])['Trade_Days'].sum()

# Create list of unique years from df
year_list = df['Year'].unique()

# Create temp lists for days above/below yearly mean returns
above_mean_list = []
below_mean_list = []

# Compare Return values to mean returns for each year, add values to temp list
for year in year_list:

    above_mean_list.extend(
        df.loc[df['Year'] == year]['Return'].transform(
            lambda x: 1 if x > mean_returns[year] else 0))

    below_mean_list.extend(
        df.loc[df['Year'] == year]['Return'].transform(
            lambda x: 1 if x < mean_returns[year] else 0))

# Add days above/below mean list into df
df['Above_Mean'] = above_mean_list
df['Below_Mean'] = below_mean_list

# Compute annual days above and below annual mean values
annual_days_above_mean = df.groupby('Year')['Above_Mean'].sum()
annual_days_below_mean = df.groupby('Year')['Below_Mean'].sum()

# Create df containing Question 2 Answers
df_q2 = pd.DataFrame(columns=[
    'year', 'trading days', 'u', '% days < u', '% days > u'])
df_q2['year'] = year_list
df_q2['trading days'] = list(annual_total_trade_days)
df_q2['u'] = list(mean_returns)
df_q2['% days < u'] = list((100 * annual_days_below_mean /
                            annual_total_trade_days).round(2))
df_q2['% days > u'] = list((100 * annual_days_above_mean /
                            annual_total_trade_days).round(2))

# Output answers to console
print("\nQuestion 2 - Table")
print(df_q2)


# ---------- Question 3 ----------

# Calculate standard deviation of returns for each year
std_returns = df.groupby('Year')['Return'].std()

# Calculate mean +- 2 * std of returns for each year
mean_minus_2_std = mean_returns - (2 * std_returns)
mean_plus_2_std = mean_returns + (2 * std_returns)

# create temp lists for days above/below mean +- 2std
minus_2_std_list = []
plus_2_std_list = []

# Compare Return values to mean (+-) 2STD for each year, add values to temp list
for year in year_list:

    minus_2_std_list.extend(
        df.loc[df['Year'] == year]['Return'].transform(
            lambda x: 1 if x < mean_minus_2_std[year] else 0))

    plus_2_std_list.extend(
        df.loc[df['Year'] == year]['Return'].transform(
            lambda x: 1 if x > mean_plus_2_std[year] else 0))

# add days above/below mean+-2std to df
df['Above_2STD'] = plus_2_std_list
df['Below_2STD'] = minus_2_std_list

# compute annual days above/below mean+-2std for each year
annual_days_above_2STD = df.groupby('Year')['Above_2STD'].sum()
annual_days_below_2STD = df.groupby('Year')['Below_2STD'].sum()

# Create df containing Question 3/4 Answers
df_q3 = pd.DataFrame(columns=[
    'year', 'trading days', 'u', 'o', '% days < u-2o', '% days > u+2o'])
df_q3['year'] = year_list
df_q3['trading days'] = list(annual_total_trade_days)
df_q3['u'] = list(mean_returns)
df_q3['o'] = list(std_returns)
df_q3['% days < u-2o'] = list((100 * annual_days_below_2STD /
                               annual_total_trade_days).round(2))
df_q3['% days > u+2o'] = list((100 * annual_days_above_2STD /
                               annual_total_trade_days).round(2))

# Output answers to console
print("\nQuestion 3&4 - Table")
print(df_q3)

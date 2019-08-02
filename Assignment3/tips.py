"""
James Sullivan
Class: CS677 - Summer 2
Date: 7/31/2019
Homework: Tips Question# 1-8

1. What is the average tip (% of meal cost) for lunch and dinner?

2. Average tip for each day of the week?

3. When are tips the highest (day and time)?

4. Compute the correlation between meal prices and tips

5. Is there any relationship between tips and size of the group?

6. What percentage of people are smoking?

7. Assume that rows in the tips.csv file are arranged in time.
   Are tips increasing with time in each day?

8. Is there any difference in correlation between tip amounts from
   smokers and non-smokers?
"""

import pandas as pd
import numpy as np
import os

# allow df prints to display each column
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)

input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
filename = os.path.join(input_dir, 'tips.csv')
df = pd.read_csv(filename)


def get_tip_pct(vector):
    """
    Get percentage given a tip value and a meal price.
    :param vector: Vector containing tip and meal price values
    :return: Percentage of meal price given as tip
    """
    return (vector[0] / vector[1]) * 100


def multiply_vectors(vector):
    """
    Multiplies two vector values
    :param vector:
    :return: Multiplied value
    """
    return int(vector[0]) * int(vector[1])


if __name__ == '__main__':

    # ----------Question 1----------
    # calculate tips as percentage of total meal cost
    df['tip_%'] = df[['tip', 'total_bill']].apply(get_tip_pct, axis=1)

    # get average tip% for each time, put into new df
    tip_avg_df = df[['time', 'tip_%']].groupby('time').mean().reset_index()

    # separate lunch and dinner into separate df's
    dinner_tips = tip_avg_df.loc[tip_avg_df['time'] == 'Dinner']
    lunch_tips = tip_avg_df.loc[tip_avg_df['time'] == 'Lunch']

    # output average lunch and dinner tips to console
    print('\n__________Question 1__________')
    print('\nAverage Lunch Tip Percentage: ', round(lunch_tips.iloc[0][1], 2))
    print('Average Dinner Tip Percentage: ', round(dinner_tips.iloc[0][1], 2))

    # ----------Question 2----------
    # calculate avg tip for each day of the week
    tip_avg_day_df = df[['day', 'tip_%']].groupby('day').mean().reset_index()
    tip_avg_day_df['tip_%'] = tip_avg_day_df['tip_%'].apply(lambda x: round(
        x, 2))

    print('\n__________Question 2__________')
    print("Average tips per day: \n", tip_avg_day_df)

    # ----------Question 3----------
    # find name of time with maximum tip
    max_tip_time = tip_avg_df['tip_%'].max()
    time_of_max_tip_df = tip_avg_df.loc[tip_avg_df['tip_%'] == max_tip_time,
                                        'time']

    # find name of day with maximum daily tip
    max_tip_day = tip_avg_day_df['tip_%'].max()
    day_of_max_tip = tip_avg_day_df.loc[tip_avg_day_df['tip_%'] == max_tip_day,
                                        'day']

    # display when tips are highest (for day and time)
    print('\n__________Question 3__________')
    print("\nTime of max tips: ", time_of_max_tip_df[1])
    print('Day of max tips: ', day_of_max_tip[0])

    # ----------Question 6----------
    # create binary class for smoker/non-smoker
    df['smoker_class'] = df['smoker'].apply(lambda x: 1 if x == 'Yes' else 0)

    # get percentage of smokers

    # multiply the number of smoker groups by the number within the group
    df['total_smokers'] = df[['smoker_class', 'size']].apply(multiply_vectors,
                                                             axis=1)

    # obtain total smokers and total number of people
    total_smokers = df['total_smokers'].sum()
    total_people = df['size'].sum()

    # get percentage of smokers
    smokers_pct = (total_smokers / total_people) * 100

    print('\n__________Question 6__________')
    print('Percentage of Smokers: ' + str(round(smokers_pct, 2)) + '%')

    # ----------Calculations for Question 7----------
    # assume that rows in tips.csv are arranged in time
    def day_change(vector):
        return 1 if vector[0] != vector[1] else 0


    # set column for day of next bill
    df['next_day'] = df['day'].shift(periods=-1)

    # if next day is different, set to 1
    df['day_change'] = df[['day', 'next_day']].apply(day_change, axis=1)
    df['time_unit'] = 0  # initialize column to 0
    bill_count = 0       # the number of bills in a day, starts at 0

    for i in range(len(df)-1):
        df.iloc[i, 12] = bill_count  # sets time_unit to bill number for day

        if df.iloc[i, 11] == 1:     # if next day is different
            bill_count = 0          # set daily bill count to 0

        else:
            bill_count += 1         # if next day is same, increment bill count

    # ----------Question 4, 5, 7 and 8----------
    # create non-smoker column
    df['non_smoker_class'] = df['smoker_class'].apply(
        lambda x: 1 if x is 0 else 0)

    # compute correlation between meal prices and tips
    correlation_matrix = df.corr()
    tip_vs_time = correlation_matrix.loc['tip_%']['time_unit']
    price_tip_corr = correlation_matrix.loc['tip_%']['total_bill']
    size_tip_corr = correlation_matrix.loc['tip_%']['size']
    smoke_tip_corr = correlation_matrix.loc['tip_%']['smoker_class']
    nonsmoke_tip_corr = correlation_matrix.loc['tip_%']['non_smoker_class']

    print('\n__________Question 4, 5, 7 and 8__________')
    # Question 4
    print('\nCorrelation between meal prices and tips: ')
    print(round(price_tip_corr, 4))

    # Question 5
    print('\nCorrelation between tips and size of group: ')
    print(round(size_tip_corr, 4))

    # Question 8
    print('\nCorrelation between smokers and tip size: ')
    print(round(smoke_tip_corr, 4))

    # Question 8
    print('\nCorrelation between non-smokers and tip size: ')
    print(round(nonsmoke_tip_corr, 4))

    print('Correlation in tip amounts form non-smokers and smokers'
          ' is inversely proportional.')

    # Question 7
    print('\nCorrelation between tip_% and the time of day (based on # bills)')
    print(round(tip_vs_time, 4))

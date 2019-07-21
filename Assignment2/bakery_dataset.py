"""
James Sullivan
Class: CS677 - Summer 2
Date: 7/21/2019
Homework: Bakery Dataset
Questions # 1-9

1. What is the busiest hour/weekday/period (in terms of transactions?
2. What is the most profitable hour/weekday/period (in terms of revenue)?
3. What is the most and least popular item?
4. Assume 1 barrista can handle 50 transactions/day, how many barristas are
   needed for each day of the weeK?
5. Divide all items into 3 groups (drinks, food, unknown). What is the avg
   price of a drink and a food item?
6. Does this coffee shop make more money from selling drinks or from selling
   food?
7. What are the top 5 most popular items for each day of the week? Does this
   list stay the same from day to day?
8. What are the bottom 5 least popular items for each day of the week? Does
   this stay the same from day to day?
9. How many drinks are there per transaction?
"""

import os
import pandas as pd
import numpy as np

input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
input_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

df = pd.read_csv(input_file)

# adding full date to df to check for unique dates
df['Date'] = df['Year'].map(str) + "-" + df['Month'].map(str) + "-" +\
             df['Day'].map(str)

# ---------- Question 1 ----------

# Obtain no. of transactions per hour/weekday/period
transaction_hour = df.groupby(['Hour'])['Transaction'].count()
transaction_weekday = df.groupby(['Weekday'])['Transaction'].count()
transaction_period = df.groupby(['Period'])['Transaction'].count()

print("\n__________Question 1__________")
print("Busiest Hour: ", transaction_hour.idxmax())
print("Busiest Weekday: ", transaction_weekday.idxmax())
print("Busiest Period: ", transaction_period.idxmax())


# ---------- Question 2 ----------

revenue_hour = df.groupby(['Hour'])['Item_Price'].sum()
revenue_weekday = df.groupby(['Weekday'])['Item_Price'].sum()
revenue_period = df.groupby(['Period'])['Item_Price'].sum()

print("\n__________Question 2__________")
print("Most Profitable Hour: ", revenue_hour.idxmax())
print("Most Profitable Weekday: ", revenue_weekday.idxmax())
print("Most Profitable Period: ", revenue_period.idxmax())


# ---------- Question 3 ----------

item_sales = df.groupby(['Item'])['Transaction'].count()

least_popular_value = min(item_sales)
least_pop_items = list(item_sales[item_sales == least_popular_value].index)

most_pop_value = max(item_sales)
most_pop_items = list(item_sales[item_sales == most_pop_value].index)

print("\n__________Question 3__________")
print("Most Popular Item(s): ")
for item in most_pop_items:
    print("   *", item)

print("\nLeast Popular Item(s): ")
for item in least_pop_items:
    print("   *", item)


# ---------- Question 4 ----------

barrista_rate = 50

#weekday_total_dict = {'Sunday': }

# Obtain number of weeks
unique_dates = df.groupby(['Weekday', 'Date'])['Transaction'].count()

#print(unique_dates.reset_index)

print(unique_dates.count(level='Weekday'))

week_count = len(df['Date'].unique()) // 7
#print(week_count)
#print(transaction_weekday)


#print(week_count)
#avg_transactions_weekday =
#print(avg_transactions_weekday)
#print(transaction_weekday // barrista_rate)

"""
James Sullivan
Class: CS677 - Summer 2
Date: 7/21/2019
Homework: Bakery Dataset
Questions # 1-9

1. What is the busiest hour/weekday/period (in terms of transactions)?
2. What is the most profitable hour/weekday/period (in terms of revenue)?
3. What is the most and least popular item?
4. Assume 1 barista can handle 50 transactions/day, how many baristas are
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
import math

input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
input_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

df = pd.read_csv(input_file)

# adding full date to df to check for unique dates
df['Date'] = df['Year'].map(str) + "-" + df['Month'].map(str) + "-" +\
             df['Day'].map(str)

# Obtain transaction count, grouped by Weekday and unique date
unique_weekday_transactions = df.groupby(
    ['Weekday', 'Date'])['Transaction'].count()

# Obtain transaction count, grouped by hour and unique date
unique_hour_transactions = df.groupby(['Hour', 'Date'])['Transaction'].count()

# Obtain transaction count, grouped by period and unique date
unique_period_transactions = df.groupby(
    ['Period', 'Date'])['Transaction'].count()

# Calculate the number of times each hour/weekday/period is in data
hour_count = unique_hour_transactions.count(level='Hour')
weekday_count = unique_weekday_transactions.count(level='Weekday')
period_count = unique_period_transactions.count(level='Period')


if __name__ == "__main__":

    # ---------- Question 1 ----------

    # Obtain total no. of transactions for each hour/weekday/period
    total_transaction_hour = df.groupby(['Hour'])['Transaction'].count()
    total_transaction_weekday = df.groupby(['Weekday'])['Transaction'].count()
    total_transaction_period = df.groupby(['Period'])['Transaction'].count()

    # Obtain # trans. per hr/weekday/period, adjusted for unique hr/day/period
    adj_transaction_hour = total_transaction_hour / hour_count
    adj_transaction_weekday = total_transaction_weekday / weekday_count
    adj_transaction_period = total_transaction_period / period_count

    print("\n__________Question 1__________")
    print("Busiest Hour: ", adj_transaction_hour.idxmax())
    print("Busiest Weekday: ", adj_transaction_weekday.idxmax())
    print("Busiest Period: ", adj_transaction_period.idxmax())

    # ---------- Question 2 ----------

    # Obtain total revenue for each hour/weekday/period
    total_revenue_hour = df.groupby(['Hour'])['Item_Price'].sum()
    total_revenue_weekday = df.groupby(['Weekday'])['Item_Price'].sum()
    total_revenue_period = df.groupby(['Period'])['Item_Price'].sum()

    # Revenue for each hr/weekday/period, adjusted for unique hr/day/period
    adj_revenue_hour = total_revenue_hour / hour_count
    adj_revenue_weekday = total_revenue_weekday / weekday_count
    adj_revenue_period = total_revenue_period / period_count

    print("\n__________Question 2__________")
    print("Most Profitable Hour: ", adj_revenue_hour.idxmax())
    print("Most Profitable Weekday: ", adj_revenue_weekday.idxmax())
    print("Most Profitable Period: ", adj_revenue_period.idxmax())

    # ---------- Question 3 ----------

    # Obtain number of transactions per item
    item_sales = df.groupby(['Item'])['Transaction'].count()

    # get list of least popular items
    least_popular_value = min(item_sales)
    least_pop_items = list(item_sales[item_sales == least_popular_value].index)

    # get list of most popular items
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

    # Number of transactions a barista can complete in one day
    barista_rate = 50

    # Calculate float number of required baristas per day
    float_barista = adj_transaction_weekday / barista_rate

    # Get ceiling values of floats to find actual no. baristas required
    int_barista = float_barista.apply(lambda x: math.ceil(x))

    weekday_list = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday', 'Saturday']

    print("\n__________Question 4__________")
    print("Number of Baristas required for each weekday: ")

    for day in weekday_list:
        print(format(day + ": ", "<11s"), format(int_barista[day], ">3d"))

    # ---------- Question 5 ----------

    # Classify Item based on len(str) b/c I am not going to manually classify
    df['Item_Type'] = df['Item'].apply(
        lambda x: 'Food' if len(x) > 5 else 'Drink')

    # get mean prices of each item type
    mean_item_type_prices = df.groupby(['Item_Type'])['Item_Price'].mean()

    print("\n__________Question 5__________")
    print("Mean Drink Price: " + "$" +
          str(mean_item_type_prices['Drink'].round(2)))
    print("Mean Food Price: " + "$" +
          str(mean_item_type_prices['Food'].round(2)))

    # ---------- Question 6 ----------

    # get total revenue for each item type
    total_item_type_revenue = df.groupby(['Item_Type'])['Item_Price'].sum()
    total_drink_revenue = total_item_type_revenue['Drink']
    total_food_revenue = total_item_type_revenue['Food']

    if total_drink_revenue > total_food_revenue:
        highest_type_revenue = 'Drinks'

    elif total_drink_revenue < total_food_revenue:
        highest_type_revenue = 'Food'

    else:
        highest_type_revenue = 'Food & Drinks'

    print("\n__________Question 6__________")
    print("This coffee shop makes more money selling: ", highest_type_revenue)

    # ---------- Question 7 -----------

    # Obtain revenue of each item, grouped by weekday
    weekday_item_revenue = df.groupby(['Weekday', 'Item'])['Item_Price'].sum()

    print("\n__________Question 7__________")
    print("Top 5 Selling Items for Each Weekday")
    for day in weekday_list:
        print("\n" + day + ": ")

        # get list of top 5 selling items for weekday
        top_five_items = list((weekday_item_revenue[day].nlargest(5)).index)

        for item in top_five_items:
            print("   *", item)

    # ---------- Question 8 ----------

    print("\n__________Question 8__________")
    print("Bottom 5 Selling Items for Each Weekday")
    for day in weekday_list:
        print("\n" + day + ": ")

        # get list of bottom 5 selling items for weekday
        bot_five_items = list((weekday_item_revenue[day].nsmallest(5)).index)

        for item in bot_five_items:
            print("   *", item)

    # ---------- Question 9 ----------

    try:

        # Count number of total transactions and total drinks sold
        total_transactions = df['Transaction'].count()
        total_drinks_sold = \
            df.loc[df['Item_Type'] == 'Drink']['Transaction'].count()

        # calculate drinks per transaction
        drinks_per_transaction = total_drinks_sold / total_transactions

        print("\n__________Question 9__________")
        print("Drinks per transaction: ", drinks_per_transaction.round(2))

    except ValueError as drinks_value_error:
        print(drinks_value_error)



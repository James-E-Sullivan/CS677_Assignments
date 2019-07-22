# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 02:18:30 2019

@author: AvantikaDG and Eugene Pinsky
"""
import os
import pandas as pd
import numpy as np

file_name = os.path.join(r'C:\Users','epinsky','bu','python','data_science_with_Python','datasets','BreadBasket_DMS_output.csv')
df = pd.read_csv(file_name)


# Busiest
print(df.groupby('Hour')['Transaction'].nunique())
print("11 is the busiest hour with 1445 transactions\n.")

print(df.groupby('Weekday')['Transaction'].nunique())
print("Saturday is the busiest day of the week with 2068 transactions.\n")

print(df.groupby('Period')['Transaction'].nunique())
print("Afternoon is the busiest period of the day with 5307 transactions.\n")

# Profitable
print(df.groupby('Hour')['Item_Price'].sum().sort_values(ascending = False))
print("11 is the most profitable hour with $21453.44 worth of transactions\n.")

print(df.groupby('Weekday')['Item_Price'].sum().sort_values(ascending = False))
print("Saturday is the most profitable day of the week with $31531.83 worth of transactions.\n")

print(df.groupby('Period')['Item_Price'].sum().sort_values(ascending = False))
print("Afternoon is the most profitable period of the day with $81299.97 worth of transactions.\n")

# Popular
print(df['Item'].value_counts())
print("Coffee is the most popular item.")
print("The least popular items are:\n1.Olum & polenta\n2.Adjustment\n3.Polenta\n4.Bacon\n5.Raw bars\n6.Chicken sand\n7.Gift Voucher\n8.The BART")

trans_day = pd.DataFrame(df.groupby(['Year', 'Month', 'Day', 'Weekday'])['Transaction'].nunique())
print("Number of barristas required per day:")
max_day = trans_day.groupby('Weekday')['Transaction'].max().to_frame()
barristas = (np.ceil(max_day.Transaction/50)).astype(int)
print("Maximum Barristas per day:")
print(barristas)

# ------------------------------------------



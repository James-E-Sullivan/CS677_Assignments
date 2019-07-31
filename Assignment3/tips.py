import pandas as pd
import numpy as np
import os

# allow df prints to display each column
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)

#input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
input_dir = r'C:\Users\james.sullivan\Documents\Personal Documents - Temporary (Delete)\BU MET\CS677\Datasets'
filename = os.path.join(input_dir, 'tips.csv')
df = pd.read_csv(filename)


def get_tip_pct(vector):
    return (vector[0] / vector[1]) * 100


# calculate tips as percentage of total meal cost
df['tip_%'] = df[['tip', 'total_bill']].apply(get_tip_pct, axis=1)
print("\n", df)

# get average tip% for each time
tip_avg_df = df[['time', 'tip_%']].groupby('time').mean().reset_index()
print("\n", tip_avg_df)

# calculate avg tip for each day of the week
tip_avg_day_df = df[['day', 'tip_%']].groupby('day').mean().reset_index()
print("\n", tip_avg_day_df)

# display when tips are highest (for day and time)
time_max_tip = tip_avg_df['tip_%'].max()
print("\nTime of max tips: ", df.loc[df['tip_%'] == time_max_tip, 'time'])

# create binary class for smoker/non-smoker
df['smoker_class'] = df['smoker'].apply(lambda x: 1 if x == 'Yes' else 0)



# assume that rows in tips.csv are arranged in time


def day_change(vector):
    return 1 if vector[0] != vector[1] else 0


df['next_day'] = df['day'].shift(periods=-1)
df['day_change'] = df[['day', 'next_day']].apply(day_change, axis=1)
df['time_unit'] = 0
bill_count = 0

for i in range(len(df)-1):
    df.iloc[i, 11] = bill_count

    if df.iloc[i, 10] == 1:
        bill_count = 0

    else:
        bill_count += 1

print(df[['time_unit', 'tip_%']].groupby('time_unit').mean().reset_index())

# create non-smoker column
df['non_smoker_class'] = df['smoker_class'].apply(lambda x: 1 if x is 0 else 0)

print(df)

# compute correlation between meal prices and tips
correlation_matrix = df.corr()
print(correlation_matrix['tip_%'])


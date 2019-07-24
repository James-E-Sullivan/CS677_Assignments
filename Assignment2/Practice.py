import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

ticker='SBUX'
input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
ticker_file = os.path.join(input_dir, ticker + '.csv')

df = pd.read_csv(ticker_file)
df['Return'] = 100.0 * df['Return']

"""
kde_join = sns.jointplot(x='Month', y='Return', data=df, kind='kde')

plt.show()
"""

mean_df = df[['Year', 'Return']].groupby('Year').mean().reset_index()
mean_df.rename(columns={'Return':'Mean_Return'}, inplace=True)
df = pd.merge(df, mean_df, on='Year')


mean_series = df.groupby(['Year'])['Return'].mean()

print(mean_df)
print(df.head())

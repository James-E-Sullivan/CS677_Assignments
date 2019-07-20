# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
"""
# run this  !pip install pandas_datareader

import os

from Strategies import strategy_1, strategy_2, strategy_3, \
    strategy_4, buy_and_hold

ticker='SBUX'
input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:   
    with open(ticker_file) as f:

        # changed this section, as it first resulted in an error
        lines = f.read().splitlines()       # split file by row (line)
        line_values = []                    # list to hold row values

        # split each row (delimited by commas) into a list
        for line in lines:
            line_values.append(line.split(","))

    print('opened file for ticker: ', ticker)

    # output results of strategies 1-4 and the buy-and-hold strategy
    strategy_1(line_values)
    buy_and_hold(line_values)
    strategy_2(line_values)
    strategy_3(line_values)
    strategy_4(line_values)

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)












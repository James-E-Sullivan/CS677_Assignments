import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sys


def get_ticker_df():
    """
    Obtains ticker data and reads it into a DataFrame
    :return data: DataFrame of stock data, unmodified
    """

    try:
        ticker = 'SBUX'
        input_dir = '../datasets'
        ticker_file = os.path.join(input_dir, ticker + '.csv')
        data = pd.read_csv(ticker_file)

    except FileNotFoundError as e:
        print(e)
        sys.exit(0)

    return data


def fix_column_names(data):

    data.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    return data


def output_plot(name):

    plot_dir = '../plots'
    output_file = os.path.join(plot_dir, name + '.pdf')
    plt.savefig(output_file)


def show_more_df():
    """allow df prints to display each column"""
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 50)

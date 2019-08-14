"""
General helper functions for plot and DataFrames for all assignments
"""
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
        input_dir = os.sep.join(os.path.dirname(os.path.realpath(__file__)).
                                split(os.sep)[:-2])
        ticker_file = os.path.join(input_dir, 'CS677_Assignments', 'datasets', ticker + '.csv')

        data = pd.read_csv(ticker_file)

    except FileNotFoundError as e:
        print(e)
        sys.exit(0)

    return data


def fix_column_names(data):
    """
    Renames 'Adj Close' column in DataFrame to 'Adj_Close' in order to
    reference the column with data.Adj_Close syntax
    :param data: DataFrame with a column named 'Adj Close'
    :return: DataFrame with 'Adj CLose' renamed 'Adj_Close'
    """
    try:
        data.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

    except KeyError as ke:
        print(ke)
        print("'Adj Close' is not a valid key")
    return data


def output_plot(name):
    """
    Outputs matplotlib plot into 'plots' relative directory
    with a given name, as a pdf.
    :param name: the file name for the plot pdf
    """
    plot_dir = os.sep.join(os.path.dirname(os.path.realpath(__file__)).
                           split(os.sep)[:-2])
    output_file = os.path.join(plot_dir, 'CS677_Assignments', 'plots', name + '.pdf')

    plt.savefig(output_file)


def show_more_df():
    """allow df prints to display each column"""
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 50)

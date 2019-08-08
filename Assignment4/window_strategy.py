"""
Contains function used to apply Assignment 4's Linear Regression window
trading strategy to a dataset, with a given w value.
"""

import pandas as pd
import numpy as np
import copy


def window_strategy(data, w):
    """
    Assumes that predicted prices have already been added as column to data.
    Strategy takes long or short positions, based on the next day's
    price. Returns data DF with profit/loss column, and short and long
    count columns.
    :param data: DataFrame with actual adjusted close prices, and predicted
    prices based on linear regression (using given w value)
    :param w: w value used in linear regression formula
    :return: data DF with profit/loss, and short and long count columns
    """
    position = 'NA'     # default position

    # stock values start at 0
    long_stock = 0.0
    short_stock = 0.0

    # prices initialized at 0
    long_price = 0.00
    short_price = 0.00

    i = w  # start at day of w value
    data['profit_loss'] = None      # initialize column w/ None
    profit_list = data.profit_loss.values   # list to be changed by strat

    # initialize short and long count columns w/ 0
    data['short_count'] = 0
    data['long_count'] = 0

    # short and long count lists to be changed by strategy
    short_list = data.short_count.values
    long_list = data.long_count.values

    # loop through days in dataset apply trading strategy
    while i < len(data):

        # set variables for row, current price, and next day's pred price
        current_row = data.iloc[i]
        current_price = current_row.Adj_Close
        pred_price = current_row.next_pred_price

        # add count to long_list or short_list, depending on position type
        if position == 'Long':
            long_list[i] = 1
        elif position == 'Short':
            short_list[i] = 1

        # Assignment 1 Window strategy (as I understood it)
        # Part 1
        if pred_price > current_price:

            # (a)
            if position is 'NA':
                long_stock = 100.00 / current_price
                long_price = copy.copy(current_price)
                position = 'Long'

            # (b) - do nothing

            # (c)
            elif position is 'Short':
                p_l_total = short_stock * (short_price - current_price)
                profit_list[i] = p_l_total
                short_price = 0.00
                position = 'NA'

        # Part 2
        elif pred_price < current_price:

            # (a)
            if position is 'NA':
                short_stock = 100.00 / current_price
                short_price = copy.copy(current_price)
                position = 'Short'

            # (b) - do nothing

            # (c)
            if position is 'Long':
                p_l_total = long_stock * (long_price - current_price)
                profit_list[i] = p_l_total
                long_price = 0.00
                position = 'NA'

        i += 1  # increment i to move to next day

    # re-fill profit/loss, and short and long counts with values found
    # by applying strategy
    data.profit_loss = profit_list
    data.short_count = short_list
    data.long_count = long_list

    return data


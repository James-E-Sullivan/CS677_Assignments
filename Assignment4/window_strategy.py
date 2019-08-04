import pandas as pd
import numpy as np
import copy


def window_strategy(data, w):

    position = 'NA'
    long_stock = 0.0
    short_stock = 0.0

    long_price = 0.00
    short_price = 0.00



    i = w  # start at day after w value
    data['profit_loss'] = None
    profit_list = data.profit_loss.values

    data['short_count'] = 0
    data['long_count'] = 0

    short_list = data.short_count.values
    long_list = data.long_count.values

    while i < len(data):

        current_row = data.iloc[i]

        current_price = current_row.Adj_Close
        pred_price = current_row.next_pred_price

        if position == 'Long':
            long_list[i] = 1
        elif position == 'Short':
            short_list[i] = 1


        '''
        print('\n', i)
        print('Current: ', current_price)
        print('Predicted: ', pred_price)
        '''

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

        i += 1

    data.profit_loss = profit_list
    data.short_count = short_list
    data.long_count = long_list

    return data


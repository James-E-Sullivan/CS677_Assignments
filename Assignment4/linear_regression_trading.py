"""
James Sullivan
Class: CS677 - Summer 2
Date: 8/3/2019
Homework: Day Trading with Linear Regression Questions 1-6
"""

import pandas as pd
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from helper_package import assignment4_helper_functions as hf
import Assignment4.window_strategy as ws

hf.show_more_df()   # allow df output to display more rows/columns
df = hf.fix_column_names(hf.get_ticker_df())     # get DataFrame of ticket file


def create_w_list(w_min, w_max, step=1):
    """
    Create list of w values (used as 'windows' for estimation)
    :param w_min: Minimum w value
    :param w_max: maximum w value
    :param step: step for w values in list
    :return output_list: list of w values
    """
    output_list = []    # list of w values

    for i in range(w_min, w_max + 1):
        output_list.append(i)
        i += step

    return output_list


def test_lin_reg():

    x = np.array([1,2,3,4,6,8])
    y = np.array([1,1,5,8,3,5])
    x_2 = x[:, np.newaxis]
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(x_2, y)

    print(lin_reg.score(x_2, y))
    print(lin_reg.coef_)            # slope
    print(lin_reg.intercept_)       # intercept

    plt.scatter(x, y)
    plot_dir = '../plots'
    output_file = os.path.join(plot_dir, 'test1' + '.pdf')
    plt.savefig(output_file)


def adj_close_lin_reg(data, w):

    # get Adj_Close values from data
    data_set = data.Adj_Close.values

    # first window end index
    end_w = copy.copy(w)    # copy to avoid resetting w's value

    pred_price_list = []   # list to hold predicted price
    r_squared_list = []

    # append None for indices less than w
    for i in range(end_w):
        pred_price_list.append(None)
        r_squared_list.append(None)

    for start_w in range((len(data_set) - w)):

        x = np.arange(start_w, end_w)
        y = data_set[start_w: end_w]

        x_2 = x[:, np.newaxis]
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(x_2, y)

        r_squared = lin_reg.score(x_2, y)
        slope = lin_reg.coef_[0]
        intercept = lin_reg.intercept_

        day = end_w
        pred_return = (slope * day) + intercept

        '''
        # for testing
        print('\nWindow: ', x)
        print('Values: ', y)
        print('R^2: ', lin_reg.score(x_2, y))
        print('Slope: ', lin_reg.coef_[0])
        print('Intercept: ', lin_reg.intercept_)
        print('Predicted Return for day=' + str(day) + ': ', pred_return, '\n')

        # further testing
        if start_w == 0:
            plt.scatter(x, y)
            plot_dir = '../plots'
            output_file = os.path.join(plot_dir, 'lin_reg_data_test' + '.pdf')
            plt.savefig(output_file)
        '''

        pred_price_list.append(pred_return)
        r_squared_list.append(r_squared)

        end_w += 1  # increment end of window

    column_name = 'w_' + str(w)             # column names

    data['next_pred_price'] = pred_price_list

    data['r_squared'] = r_squared_list
    data.next_pred_price = data.next_pred_price.shift(periods=-1)

    data_p_l = ws.window_strategy(data, w)
    data.merge(data_p_l.profit_loss)
    data.rename(columns={'profit_loss': column_name + '_profit'}, inplace=True)
    data.drop(columns='next_pred_price', inplace=True)

    return data


def pred_multiple_w(data, w_values):

    for w_value in w_values:
        new_df = adj_close_lin_reg(data, w_value)
        data.merge(new_df)

    #data.drop(columns='next_pred_price', inplace=True)

    return data


def find_mean_profit(data, w_values):
    mean_profit_list = []
    for w in w_values:
        column_name = 'w_' + str(w) + '_profit'
        mean_profit = data[column_name].mean()
        mean_profit_list.append(mean_profit)

    return mean_profit_list


def get_optimal_w(data, w_values):

    data_with_profit = pred_multiple_w(data, w_list)
    mean_profits = find_mean_profit(data_with_profit, w_values)

    plt.scatter(w_values, mean_profits)
    plot_dir = '../plots'
    output_file = os.path.join(plot_dir, 'w_mean_profit' + '.pdf')
    plt.savefig(output_file)

    mean_profit_df = pd.DataFrame(columns={'w_value', 'mean_profit'})
    mean_profit_df.w_value = w_values
    mean_profit_df.mean_profit = mean_profits

    max_profit = mean_profit_df['mean_profit'].max()

    # this seems like a roundabout way of getting the optimal w, but it works
    optimal_w = mean_profit_df.loc[mean_profit_df.mean_profit == max_profit].w_value.values[0]

    return optimal_w


df_2017 = df[df.Year == 2017].reset_index(drop=True)
df_2018 = df[df.Year == 2018].reset_index(drop=True)

w_minimum = 5
w_maximum = 30
start_day = w_maximum + 1
w_list = create_w_list(w_minimum, w_maximum)




if __name__ == '__main__':
    #print(adj_close_lin_reg(df_2017, 5))
    #output_data = pred_multiple_w(df_2017, w_list)
    #print(output_data)
    #print(find_mean_profit(output_data, w_list))

    # ---------- Question 1 ----------
    optimal_w_value = get_optimal_w(df_2017, w_list)
    print('\n__________Question 1__________')
    print('Optimal W value (for W=5, 6,..., 30): ', optimal_w_value)

    # ---------- Question 2 ----------
    df_2018_lin_reg = adj_close_lin_reg(df_2018, optimal_w_value)
    mean_r_squared = df_2018_lin_reg.r_squared.mean()

    print('\n__________Question 2__________')
    print('2018 Mean r^2 value: ', round(mean_r_squared, 6))
    print('r^2 value indicates that linear regression does'
          ' not perfectly predict price movements.')
    #print(df_2018_lin_reg)




"""
James Sullivan
Class: CS677 - Summer 2
Date: 8/11/2019
Homework: Random Forest Questions #1-4

Implement a random forest classifier. For each week, your feature set is
(mu, sigma) for that week. Use your labels (you will have 52 labels per year -
1 for each week) from year 1 to train your classifier and predict labels for
year 2. Recall that there are two hyper-parameters in the random forest
classifier:

    1. N - number of (sub)trees to use
    2. d - max depth of each subtree

Questions:
1. Take N=1,...,10 and d = 1,2,...,5. For each value of N and d, construct a
   random tree classifier (use "entropy" as splitting criteria - this is the
   default). Use your year 1 labels as a training set and compute the error
   rate for year 2. Plot your error rates and find the best combination of N
   and d.
2. Using the optimal values from year 1, compute the confusion matrix for
   year 2.
3. What is the true positive rate (TPR) and true negative rate (TNR) for year 2
4. Implement a trading strategy based on your labels for year 2 and compare
   the performance with the "buy-and-hold" strategy. Which strategy results in
   a larger amount at the end of the year.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper_package import helper_functions as hf
from helper_package import feature_set as fs
from helper_package import confusion_matrix_calcs as cm
from helper_package import assign_labels as al

# raises numpy errors/warnings so they can be caught by try/except
np.seterr(all='raise')

# allow df console output to display more columns
hf.show_more_df()

# get DataFrame of stock ticker info from csv file
df = hf.fix_column_names(hf.get_ticker_df())

df = al.assign_color_labels(df)  # assign color labels
df = fs.get_feature_set(df)      # add mean and std return columns for DF


def random_forest_predict(df1, df2, N, d):
    """
    Random Forest classification of labels (colors)
    :param df1: Training set (DataFrame)
    :param df2: Prediction set (DataFrame)
    :param N: number of (sub)trees to use
    :param d: max depth of each subtree
    :return: df2 with predicted label (binary) and color columns
    """

    try:

        x = df1[['Mean_Return', 'Std_Return']].values
        y = df1.binary_label.values
        x_2 = df2[['Mean_Return', 'Std_Return']].values

        rf_classifier = RandomForestClassifier(n_estimators=N, max_depth=d,
                                               criterion='entropy')
        rf_classifier.fit(x, y)

        pred_df = copy.copy(df2)  # copies df2 to ensure that orig isn't changed

        pred_df['pred_label'] = rf_classifier.predict(x_2)
        pred_df['pred_color'] = pred_df.pred_label.apply(
            lambda a: 'Green' if a is 1 else 'Red')
        return pred_df

    except KeyError as ke:
        print(ke)
        print('Key does not exist')


# create DataFrames for 2017 and 2018 from df
df_2017 = df.loc[df.Year == 2017].reset_index()
df_2018 = df.loc[df.Year == 2018].reset_index()

# hard-coded n_list and d_list values to create N/d pairs
n_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
d_list = [1, 2, 3, 4, 5]

# get combinations of n_list and d_list
n_d_combinations = [(n, d) for n in n_list for d in d_list]


if __name__ == '__main__':

    # list for N & d error values
    nd_errors = []

    # obtain error rate for each N/d combo and append it to nd_errors
    for nd_pair in n_d_combinations:

        pred_2018 = random_forest_predict(
            df_2017, df_2018, nd_pair[0], nd_pair[1])

        pred_2018['acc_count'] = pred_2018[[
            'binary_label', 'pred_label']].apply(cm.get_acc, axis=1)

        accuracy = pred_2018.acc_count.sum() / pred_2018.acc_count.count()
        error_rate = 1 - accuracy
        nd_errors.append(error_rate)

    # create figure for error rates of N&d values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = []  # x value list (for N)
    ys = []  # y value list (for d)

    # append N and d values to xs and ys lists
    for nd in n_d_combinations:
        xs.append(nd[0])  # N value
        ys.append(nd[1])  # d value

    zs = nd_errors  # z value list references the list of errors (nd_errors)
    ax.scatter(xs, ys, zs)  # create scatter plot of N, d, and nd_errors

    # add labels and titles to axes
    ax.set_xlabel('N value')
    ax.set_ylabel('d value')
    ax.set_zlabel('error rate')
    ax.set_title('error rates for N & d value pairs')

    # get relative project directory and save figure to output file
    plot_dir = os.sep.join(os.path.dirname(os.path.realpath(__file__)).
                           split(os.sep)[:-2])
    plot_name = 'nd_plot'
    output_file = os.path.join(
        plot_dir, 'CS677_Assignments', 'plots', plot_name + '.pdf')
    plt.savefig(output_file)

    # create a data frame for N/d pair values and their respective error rates
    df_nd_err = pd.DataFrame(columns={'nd_pair', 'error_rate'})
    df_nd_err.nd_pair = n_d_combinations  # fill nd_pair with n_d_combinations
    df_nd_err.error_rate = nd_errors  # fill error_rate with nd_errors
    min_err = df_nd_err.error_rate.min()  # find minimum error rate

    # gets first n&d pair with minimum error rate
    optimal_nd = df_nd_err.loc[df_nd_err.error_rate == min_err].nd_pair.values[0]

    # use random forest classifier to predict 2018 labels
    df_2018 = random_forest_predict(
        df_2017, df_2018, optimal_nd[0], optimal_nd[1])

    # ---------- Question 1 ----------
    # find best combination of N and d
    df_2018['acc_counter'] = df_2018[
        ['binary_label', 'pred_label']].apply(cm.get_acc, axis=1)
    accuracy = df_2018.acc_counter.sum() / df_2018.acc_counter.count()
    percent_accuracy = accuracy * 100
    print('__________Question 1__________')
    print('Optimal combination of N and d: ')
    print('N =', optimal_nd[0])
    print('d =', optimal_nd[1])
    print('Error rate of optimal N & d pair: ', min_err)
    print('Accuracy of optimal N & d pair when used to predict 2018 labels: ',
          percent_accuracy)

    # ---------- Question 2 ----------
    # add columns for true/false positive/negative values
    df_2018['tp'] = df_2018[['binary_label', 'pred_label']].apply(cm.tp, axis=1)
    df_2018['fp'] = df_2018[['binary_label', 'pred_label']].apply(cm.fp, axis=1)
    df_2018['tn'] = df_2018[['binary_label', 'pred_label']].apply(cm.tn, axis=1)
    df_2018['fn'] = df_2018[['binary_label', 'pred_label']].apply(cm.fn, axis=1)

    # compute sum of true/false positive/negative values
    tp_sum = df_2018.tp.sum()
    fp_sum = df_2018.fp.sum()
    tn_sum = df_2018.tn.sum()
    fn_sum = df_2018.fn.sum()

    # calculate confusion matrix with sum values
    confusion_matrix_2018 = cm.confusion_matrix(tp_sum, fp_sum, tn_sum, fn_sum)

    print('\n__________Question 2__________')
    print('2018 Confusion Matrix (using optimal N & d pair): \n',
          confusion_matrix_2018)

    # ---------- Question 3 ----------
    # calculate true positive and true negative rates

    try:
        tpr = tp_sum / (tp_sum + fp_sum)
    except FloatingPointError:
        # dividing by 0 will result in FloatingPointError
        tpr = 0

    try:
        tnr = tn_sum / (tn_sum + fp_sum)
    except FloatingPointError:
        # dividing by 0 will result in FloatingPointError
        tnr = 0

    # print tpr and tnr values
    print('\n__________Question 3__________')
    print('True Positive Rate: ', tpr)
    print('True Negative Rate: ', tnr)

    # ---------- Question 4 ----------

    # get final funds for 2018 using color strategy and using buy & hold
    color_strat_final = al.color_strategy(df_2018)
    buy_and_hold_final = al.buy_and_hold(df_2018)

    print('\n__________Question 4__________')
    print('Color Strategy Final Funds: $' + str(round(color_strat_final, 2)))
    print('Buy & Hold Strategy Final Funds: $' + str(
        round(buy_and_hold_final, 2)))

    if color_strat_final > buy_and_hold_final:
        best_strat = 'Color Strategy'
    else:
        best_strat = 'Buy & Hold Strategy'

    print('Which strategy results in larger amount at the end of the year?',
          best_strat)


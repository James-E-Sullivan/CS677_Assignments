"""
James Sullivan
Class: CS677 - Summer 2
Date: 8/10/2019
Homework: Naive Bayesian Questions #1-4

Implement a Naive Bayesian classifier. For each week, your feature set is
(mu, sigma) for that week. Use your labels (you will have 52 labels per year -
1 for each week) from year 1 to train your classifier and predict labels for
year 2. Use Gaussian Naive Bayesian (this is the default).

1. Implement a Gaussian Naive Bayesian classifier and compute its accuracy
   for year 2
2. Compute the confusion matrix for year 2
3. What is the true positive rate (TPR) and true negative rate (TNR) for year 2
4. Implement a trading strategy based on your labels for year 2 and compare
   the performance with the "buy-and-hold" strategy. Which strategy results in
   a larger amount at the end of the year.
"""

import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
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


def nb_predict(df1, df2):
    """
    Gaussian Naive Bayesian classification of labels (colors)
    :param df1: Training set (DataFrame)
    :param df2: Prediction set (DataFrame)
    :return: df2 with predicted label (binary) and color columns
    """
    x = df1[['Mean_Return', 'Std_Return']].values
    y = df1.binary_label.values
    x_2 = df2[['Mean_Return', 'Std_Return']].values

    NB_classifier = GaussianNB().fit(x, y)
    df2['pred_label'] = NB_classifier.predict(x_2)
    df2['pred_color'] = df2.pred_label.apply(lambda a: 'Green' if a is 1 else 'Red')

    return df2


df_2017 = df.loc[df.Year == 2017].reset_index()
df_2018 = df.loc[df.Year == 2018].reset_index()


if __name__ == '__main__':

    df_2018 = nb_predict(df_2017, df_2018)

    # ---------- Question 1 ----------
    df_2018['acc_counter'] = df_2018[['binary_label', 'pred_label']].apply(cm.get_acc, axis=1)
    accuracy = df_2018.acc_counter.sum() / df_2018.acc_counter.count()
    percent_accuracy = accuracy * 100
    print('__________Question 1__________')
    print('Percent accuracy for 2018: ', round(percent_accuracy, 2))

    # ---------- Question 2 ----------
    compare_vector = df_2018[['binary_label', 'pred_label']]

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
    print('2018 Confusion Matrix: \n', confusion_matrix_2018)

    # ---------- Question 3 ----------
    # calculate true positive and true negative rates

    '''
    tpr = tp_sum / (tp_sum + fp_sum)
    tnr = tn_sum / (tn_sum + fp_sum)
    '''

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
    print('True Positive Rate: ', round(tpr, 4))
    print('True Negative Rate: ', round(tnr, 4))

    # ---------- Question 4 ----------
    color_strat_final = al.color_strategy(df_2018)
    buy_and_hold_final = al.buy_and_hold(df_2018)
    print('\n__________Question 4__________')
    print('Color Strategy Final Funds: $' + str(round(color_strat_final, 2)))
    print('Buy & Hold Strategy Final Funds: $' + str(round(buy_and_hold_final, 2)))




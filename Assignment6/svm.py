"""
James Sullivan
Class: CS677 - Summer 2
Date: 8/17/2019
Homework: SVM Questions #1-6

Implement a support vector machine (SVM) classifier. For each week, your
feature set is (mu, sigma) for that week. Use your labels (you will have 52
labels per year - 1 for each week) from year 1 to train your classifier and
predict labels for year 2.

1. Implement a linear SVM. What is the accuracy of your SVM for year 2?
2. Compute the confusion matrix for year 2?
3. What is true positive rate and true negative rate for year 2?
4. Implement a Gaussian SVM and compute its accuracy for year 2. Is it better
   than linear SVM (use default values for parameters)?
5. Implement a polynomial SVM for degree 2 and compute its accuracy. Is it
   better than linear SVM?
6. Implement a trading strategy based on your labels (from linear SVM) for year
   2 and compare the performance with the "buy & hold" strategy. Which
   strategy results in a larger amount at the end of the year?
"""

import pandas as pd
import numpy as np
import os
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from helper_package import helper_functions as hf
from helper_package import feature_set as fs
from helper_package import confusion_matrix_calcs as cm
from helper_package import assign_labels as al


def linear_svm(df1, df2):
    """
    SVM classification (linear, gaussian, and polynomial SVM) of labels
    (colors).
    :param df1: Training Set (DataFrame)
    :param df2: Prediciton Set (DataFrame)
    :return: df2 with predicted (binary) labels for linear, gaussian, and
    polynomial SVM classified labels. Also includes color labels using linear
    SVM.
    """

    # mean and std deviation of yr1 weekly returns
    x = df1[['Mean_Return', 'Std_Return']].values

    # scale yr1 data
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    y = df1.binary_label.values  # training (1st yr) label values

    # mean and std deviation of yr2 weekly returns
    x_2 = df2[['Mean_Return', 'Std_Return']].values

    # scale yr2 data
    scaler.fit(x_2)
    x_2 = scaler.transform(x_2)

    # create SVM classifiers using linear, gaussian, and polynomial kernels
    l_svm_classifier = svm.SVC(kernel='linear')
    l_svm_classifier.fit(x, y)

    g_svm_classifier = svm.SVC(kernel='rbf')
    g_svm_classifier.fit(x, y)

    p_svm_classifier = svm.SVC(kernel='poly', degree=2)
    p_svm_classifier.fit(x, y)

    # add column for predicted label (0 or 1) for linear SVM
    df2['l_svm_pred'] = l_svm_classifier.predict(x_2)

    # predict color for lin SVM - used in color strategy
    df2['pred_color'] = df2.l_svm_pred.apply(
        lambda a: 'Green' if a is 1 else 'Red')

    # add column for only predicted label (0 or 1) for gaussian and poly SVM
    df2['g_svm_pred'] = g_svm_classifier.predict(x_2)
    df2['p_svm_pred'] = p_svm_classifier.predict(x_2)

    return df2  # yr2 df with added prediction columns


if __name__ == '__main__':

    # raises numpy errors/warnings so they can be caught by try/except
    np.seterr(all='raise')

    # allow df console output to display more columns
    hf.show_more_df()

    # get DataFrame of stock ticker info from csv file
    df = hf.fix_column_names(hf.get_ticker_df())

    df = al.assign_color_labels(df)  # assign color labels
    df = fs.get_feature_set(df)  # add mean and std return columns for DF

    # create DataFrames for 2017 and 2018 from df
    df_2017 = df.loc[df.Year == 2017].reset_index()
    df_2018 = df.loc[df.Year == 2018].reset_index()

    # predict labels with linear_svm()
    df_2018 = linear_svm(df_2017, df_2018)

    # add column in df_2018 for accuracy of predicted SVM values (1 if accurate)
    df_2018['linear_acc'] = df_2018[
        ['binary_label', 'l_svm_pred']].apply(cm.get_acc, axis=1)
    df_2018['gaussian_acc'] = df_2018[
        ['binary_label', 'g_svm_pred']].apply(cm.get_acc, axis=1)
    df_2018['poly_acc'] = df_2018[
        ['binary_label', 'p_svm_pred']].apply(cm.get_acc, axis=1)

    # compute accuracy for linear, gaussian, and polynomial SVM
    linear_accuracy = df_2018.linear_acc.sum() / df_2018.linear_acc.count()
    gaussian_accuracy = df_2018.gaussian_acc.sum() / df_2018.\
        gaussian_acc.count()
    poly_accuracy = df_2018.poly_acc.sum() / df_2018.poly_acc.count()

    print('\n__________Question 1__________')
    print('Percent Accuracy for 2018 (Linear SVM): ',
          round(linear_accuracy * 100, 4))

    # ---------- Question 2 ----------

    # add columns for true/false positive/negative rates (for linear SVM)
    df_2018['tp'] = df_2018[['binary_label', 'l_svm_pred']].apply(cm.tp, axis=1)
    df_2018['fp'] = df_2018[['binary_label', 'l_svm_pred']].apply(cm.fp, axis=1)
    df_2018['tn'] = df_2018[['binary_label', 'l_svm_pred']].apply(cm.tn, axis=1)
    df_2018['fn'] = df_2018[['binary_label', 'l_svm_pred']].apply(cm.fn, axis=1)

    # compute sum of true/false positive/negative values
    tp_sum = df_2018.tp.sum()
    fp_sum = df_2018.fp.sum()
    tn_sum = df_2018.tn.sum()
    fn_sum = df_2018.fn.sum()

    # compute confusion matrix
    confusion_matrix_2018 = cm.confusion_matrix(tp_sum, fp_sum, tn_sum, fn_sum)

    print('\n__________Question 2__________')
    print('Confusion Matrix for 2018 (Linear SVM): \n', confusion_matrix_2018)

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

    # ---------- Question 4 ----------
    print('\n__________Question 4__________')
    print('Percent Accuracy for 2018 (Gaussian SVM): ',
          round(gaussian_accuracy * 100, 4))

    if linear_accuracy > gaussian_accuracy:
        print('Gaussian SVM is less accurate than Linear SVM')
    elif linear_accuracy < gaussian_accuracy:
        print('Gaussian SVM is more accurate than Linear SVM')
    elif linear_accuracy == gaussian_accuracy:
        print('Gaussian SVM and Linear SVM are equally accurate')

    # ---------- Question 5 ----------
    print('\n___________Question 5__________')
    print('Percent Accuracy for 2018 (Polynomial SVM): ',
          round(poly_accuracy * 100, 4))

    if linear_accuracy > poly_accuracy:
        print('Polynomial SVM is less accurate than Linear SVM')
    elif linear_accuracy < poly_accuracy:
        print('Polynomial SVM is more accurate than Linear SVM')
    elif linear_accuracy == poly_accuracy:
        print('Polynomial SVM and Linear SVM are equally accurate')

    # ---------- Question 6 -----------

    # get final funds for 2018 using color strategy & buy and hold
    color_strat_final = al.color_strategy(df_2018)
    buy_and_hold_final = al.buy_and_hold(df_2018)

    # output final fund values for both strategies
    print('\n__________Question 6__________')
    print('Color Strategy Final Funds: $' + str(round(color_strat_final, 2)))
    print('Buy & Hold Strategy Final Funds: $' + str(
        round(buy_and_hold_final, 2)))

    # determines best strategy based on greatest final funds
    if color_strat_final > buy_and_hold_final:
        best_strat = 'Color Strategy'
    elif color_strat_final < buy_and_hold_final:
        best_strat = 'Buy & Hold Strategy'
    else:
        best_strat = 'Both strategies are equal'

    # output best strategy to console
    print('Which strategy results in larger amount at the end of the year?',
          best_strat)


"""
James Sullivan
Class: CS677 - Summer 2
Date: 7/31/2019
Homework: Logistic Regression Question# 1-5

1. What is the equation for logistic regression that your classifier
   found from year 1 data?

2. What is the accuracy for year 2?

3. Compute the confusion matrix for year 2.

4. What is the true positive rate and true negative rate for year 2?

5. Implement a trading strategy based on labels from year 2 and compare
   performance with a buy-and-hold strategy. Which strategy results in
   a larger amount at the end of the year?
"""

import pandas as pd
import numpy as np
import os
from Assignment2 import week_labeling as wl
from Assignment3 import mu_sigma_feature_set as fs

# allow df prints to display each column
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)

# obtain DataFrame of ticker file
data = fs.get_ticker_df()

# split data into DataFrames for 2017 and 2018
df_2017 = data[data['Year'] == 2017].reset_index()
df_2018 = data[data['Year'] == 2018].reset_index()

# obtain smaller df of 2017 data as a training set
training_set = df_2017[[
    'Year_Week', 'Mean_Return', 'Std_Return', 'Color', 'Class']].groupby(
    'Year_Week').max().reset_index()

# obtain x and y values for logitic regression
x = training_set[['Mean_Return', 'Std_Return']].values
y = training_set['Class'].values

N = len(y)
learning_rate = 0.01


def sigmoid(z):
    """
    sigmoid function for a given value z
    :param z: input value
    :return: output of sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-z))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def add_intercept(x):

    intercept = np.ones((x.shape[0], 1))
    return np.concatenate((intercept, x), axis=1)


threshold = 0.1
n = 5
iterations = 10000

x = add_intercept(x)
weights = np.zeros(x.shape[1])


def compute_weights(x, weights, iterations, learning_rate, debug_step=1000):
    """
    Compute weights for regression
    :param x: feature-set
    :param weights: blank np array
    :param iterations: number of iterations to be used
    :param learning_rate: learning rate of logistic regression
    :param debug_step: Step for debugging if output message is to be
    displayed
    :return:
    """
    for i in range(iterations):
        y_pred = np.dot(x, weights)
        phi = sigmoid(y_pred)
        gradient = np.dot(x.T, (phi-y))/N
        weights = weights - learning_rate * gradient
        if i % debug_step==0:
            y_pred = np.dot(x, weights)
            phi = sigmoid(y_pred)
    return weights


print('\n__________Question 1__________')

# weights np array filled with computed weights
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=1000)

# output equation
print('Equation for logistic regression from 2017: ')
print('y = ' + str(round(weights[0], 4)) + ' + (' + str(round(weights[1], 4))
      + ' * mu) + (' + str(round(weights[2], 4)) + ' * sigma)')
print('z(y) = 1 / (1 + e^(-y))')


def predict_y(mu, sigma, weight_list):
    """
    Apply to mu and sigma values in df
    :param mu: mean
    :param sigma: standard deviation
    :param weight_list: Weights
    :return: Predicted y value for lin reg
    """
    return weight_list[0] + (weight_list[1] * mu) + \
        (weight_list[2] * sigma)


print('\n__________Question 2__________')

# predict color labels for 2018
df_2018['Linear_Y'] = df_2018.apply(
    lambda b: predict_y(b['Mean_Return'], b['Std_Return'], weights), axis=1)

df_2018['Logistic_Y'] = df_2018[['Linear_Y']].apply(sigmoid, axis=1)

df_2018['Predicted_Color'] = df_2018['Logistic_Y'].apply(
    lambda c: 'Green' if c > 0.5 else 'Red')


# Smaller df with predicted 2018 values, grouped by 'Year_Week'
pred_2018 = df_2018[
    ['Year_Week', 'Logistic_Y', 'Color', 'Predicted_Color']].groupby(
    ['Year_Week']).max()

pred_2018['Accuracy'] = pred_2018[['Color', 'Predicted_Color']].apply(
    fs.get_acc, axis=1)


# calculate accuracy of predicted values for 2018
correct_count = pred_2018['Accuracy'].sum()
total_weeks = pred_2018['Accuracy'].count()
accuracy = correct_count / total_weeks

percent_accuracy = accuracy * 100
print('Percent Accuracy for 2018 predicted values', round(percent_accuracy, 2))


print('\n__________Question 3__________')

# calculate true/false positive/negative values for 2018 predictions
tp_sum = pred_2018[['Color', 'Predicted_Color']].apply(fs.tp, axis=1).sum()
fp_sum = pred_2018[['Color', 'Predicted_Color']].apply(fs.fp, axis=1).sum()
tn_sum = pred_2018[['Color', 'Predicted_Color']].apply(fs.tn, axis=1).sum()
fn_sum = pred_2018[['Color', 'Predicted_Color']].apply(fs.fn, axis=1).sum()

# calculate and output confusion matrix
confusion_2018 = np.array([[tn_sum, fp_sum], [fn_sum, tp_sum]])
print("\nConfusion Matrix: \n", confusion_2018)

print('\n__________Question 4__________')

# calculate true positive rate and true negative rate
tpr = tp_sum / (tp_sum + fn_sum)
tnr = tn_sum / (tn_sum + fp_sum)

# print tpr and tnr values
print("\nTrue Positive Rate: ", tpr)
print("True Negative Rate: ", tnr)

print('\n___________Question 5___________')

# set df_2018 color value to predicted color so color_strat_log uses pred values
df_2018['Color'] = df_2018['Predicted_Color']

color_strat_log = round(wl.color_strategy(df_2018), 2)
print("Color Strategy for 2018: " + "$" + str(color_strat_log))

buy_and_hold_final = round(wl.buy_and_hold(df_2018), 2)
print("Buy & Hold for 2018: " + "$" + str(buy_and_hold_final))




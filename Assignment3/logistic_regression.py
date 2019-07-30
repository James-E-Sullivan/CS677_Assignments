import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Assignment2 import week_labeling as wl
from Assignment3 import mu_sigma_feature_set as fs

data = fs.get_ticker_df()

df_2017 = data[data['Year'] == 2017].reset_index()
df_2018 = data[data['Year'] == 2018].reset_index()

training_set = df_2017[[
    'Year_Week', 'Mean_Return', 'Std_Return', 'Color', 'Class']].groupby(
    'Year_Week').max().reset_index()

#print(training_set.head())


x = training_set[['Mean_Return', 'Std_Return']].values
y = training_set['Class'].values

N = len(y)
learning_rate = 0.01


def sigmoid(z):
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


def compute_weights(x,weights,iterations, learning_rate, debug_step=1000):
    for i in range(iterations):
        y_pred = np.dot(x, weights)
        phi = sigmoid(y_pred)
        gradient = np.dot(x.T, (phi-y))/N
        weights = weights - learning_rate * gradient
        if i % debug_step==0:
            y_pred = np.dot(x, weights)
            phi = sigmoid(y_pred)
            #print('i:', i, 'loss: ', loss(phi, y_pred))
    #print('rate: ', learning_rate, ' iterations ', iterations, ' weights: ', weights)
    return weights

# compute weights
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=1000)

print(weights[0])


def predict_y(mu, sigma, weight_list):
    """
    Apply to mu and sigma values in df
    :param feature_values: (mu, sigma)
    :param weight_list: Weights
    :return: Predicted y value for lin reg
    """
    return weight_list[0] + (weight_list[1] * mu) + \
        (weight_list[2] * sigma)


def get_acc(vector):

    if vector[0] is vector[1]:
        return 1
    else:
        return 0


def tp(vector):

    if vector[0] is 'Green' and vector[1] is 'Green':
        return 1
    else:
        return 0


def fp(vector):

    if vector[0] is 'Red' and vector[1] is 'Green':
        return 1
    else:
        return 0


def tn(vector):

    if vector[0] is 'Red' and vector[1] is 'Red':
        return 1
    else:
        return 0


def fn(vector):

    if vector[0] is 'Green' and vector[1] is 'Red':
        return 1
    else:
        return 0


#df_2018['Feature_Values'] = df_2018[['Mean_Return', 'Std_Return']].apply(lambda b: (b[0], b[1]))


df_2018['Linear_Y'] = df_2018.apply(lambda b: predict_y(b['Mean_Return'], b['Std_Return'], weights), axis=1)
df_2018['Logistic_Y'] = df_2018[['Linear_Y']].apply(sigmoid, axis=1)
df_2018['Predicted_Color'] = df_2018['Logistic_Y'].apply(lambda c: 'Green' if c > 0.5 else 'Red')




# Smaller df with predicted 2018 values, grouped by 'Year_Week'
pred_2018 = df_2018[['Year_Week', 'Logistic_Y', 'Color', 'Predicted_Color']].groupby(['Year_Week']).max()

pred_2018['Accuracy'] = pred_2018[['Color', 'Predicted_Color']].apply(get_acc, axis=1)

#print(predicted_2018)

correct_count = pred_2018['Accuracy'].sum()
total_weeks = pred_2018['Accuracy'].count()

accuracy = correct_count / total_weeks
percent_accuracy = accuracy * 100
print(accuracy.round(2))

"""
pred_2018['TP'] = pred_2018[['Color', 'Predicted_Color']].apply(tp, axis=1)
pred_2018['FP'] = pred_2018[['Color', 'Predicted_Color']].apply(fp, axis=1)
pred_2018['TN'] = pred_2018[['Color', 'Predicted_Color']].apply(tn, axis=1)
pred_2018['FN'] = pred_2018[['Color', 'Predicted_Color']].apply(fn, axis=1)
"""

tp_sum = pred_2018[['Color', 'Predicted_Color']].apply(tp, axis=1).sum()
fp_sum = pred_2018[['Color', 'Predicted_Color']].apply(fp, axis=1).sum()
tn_sum = pred_2018[['Color', 'Predicted_Color']].apply(tn, axis=1).sum()
fn_sum = pred_2018[['Color', 'Predicted_Color']].apply(fn, axis=1).sum()

print(pred_2018)

""""
confusion_2018 = np.array([[pred_2018['TN'].sum(), pred_2018['FP'].sum()],
                           [pred_2018['FN'].sum(), pred_2018['TP'].sum()]])
"""

confusion_2018 = np.array([[tn_sum, fp_sum], [fn_sum, tp_sum]])

tpr = tp_sum / (tp_sum + fn_sum)
tnr = tn_sum / (tn_sum + fp_sum)
#test_array = np.array([0, 1], [])

print("\nConfusion Matrix: \n", confusion_2018)

print("\nTrue Positive Rate: ", tpr)
print("\nTrue Negative Rate: ", tnr)

df_2018['Color'] = df_2018['Predicted_Color']

color_strat_final = wl.color_strategy(df_2018).round(2)
print("Color Strategy for 2018: " + "$" + str(color_strat_final))

buy_and_hold_final = wl.buy_and_hold(df_2018).round(2)
print("Buy & Hold for 2018: " + "$" + str(buy_and_hold_final))




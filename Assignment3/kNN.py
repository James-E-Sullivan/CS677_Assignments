"""
James Sullivan
Class: CS677 - Summer 2
Date: 7/31/2019
Homework: kNN Question# 1-5

1. Take k=3, 5, 7, 9, 11. For each value of k compute the accuracy of
   your kNN classifier on year 1 data. **On x axis you plot k and on
   y-axis you plot accuracy** (Is this asking us to make a plot as
   part of the question? I did not do this part). What is the optimal
   value of k for year 1?

2. Use the optimal value of k from year 1 to predict labels for year 2.
   What is your accuracy?

3. Using the optimal value for k from year 1, compute the confusion matrix
   for year 2. What is your accuracy?

4. What is the true positive rate and true negative rate for year 2?

5. Implement a trading strategy based on labels from year 2 and compare
   performance with a buy-and-hold strategy. Which strategy results in
   a larger amount at the end of the year?
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

# obtain smaller df of 2017 data, grouped by week
input_2017 = df_2017[['Year_Week', 'Mean_Return', 'Std_Return', 'Color',
                      'Class']].groupby('Year_Week').max().reset_index()

# obtain X and Y values for kNN classification
X = input_2017[['Mean_Return', 'Std_Return']].values
Y = input_2017['Class'].values

k_values = [3, 5, 7, 9, 11]  # list of k values to test

# obtain smaller df of 2018 data, grouped by week
output_2018 = df_2018[['Year_Week', 'Mean_Return', 'Std_Return', 'Color',
                       'Class']].groupby('Year_Week').max().reset_index()

# initialize accuracy_list for accuracy values
accuracy_list = []

print('k value accuracy for 2017: ')

# obtain accuracy for each k in k_values
for k in k_values:

    # get kNN classifier for k, fit to X and Y from above
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X, Y)

    # predict 2017 labels with X and kNN_classifier (to find best k value)
    new_instance = input_2017[['Mean_Return', 'Std_Return']].values
    prediction = knn_classifier.predict(new_instance)

    # add predictions for each k value as new column to 2017 df
    column_name = 'k_' + str(k)
    input_2017[column_name] = prediction

    # create accuracy column in df (1 for correct prediction, else 0)
    accuracy_name = column_name + '_acc'
    input_2017[accuracy_name] = input_2017[['Class', column_name]].apply(
        fs.get_acc, axis=1)

    # calculate accuracy for each k value (ratio not percentage)
    correct_count = input_2017[accuracy_name].sum()
    total_weeks = input_2017[accuracy_name].count()
    accuracy = correct_count / total_weeks

    # append accuracy to accuracy_list
    accuracy_list.append(accuracy)

    # output accuracy and k values to console
    print(accuracy_name + ':', round(accuracy, 4))


def kNN_predict(df1, k1, X1, Y1):
    """
    Add predicted class and color values to a DataFrame using
    kNN classifier.
    :param df1: input DataFrame
    :param k1: k value for kNN
    :param X1: X value from training set
    :param Y1: Y value from training set
    :return df1: df w/ appended predicted class & color
    """

    # get kNN classifier for k, fit to X and Y
    knn_classifier1 = KNeighborsClassifier(n_neighbors=k1)
    knn_classifier1.fit(X1, Y1)

    # predict label based on df1 feature-set
    new_df_features = df1[['Mean_Return', 'Std_Return']].values
    new_df_prediction = knn_classifier1.predict(new_df_features)

    # add predicted class and labels to df1
    df1['Pred_Class'] = new_df_prediction
    df1['Pred_Color'] = df1['Pred_Class'].apply(
        lambda x: 'Green' if x is 1 else 'Red')

    return df1


# create df for 2017 k and accuracy values
k_acc_df = pd.DataFrame()
k_acc_df['k'] = k_values
k_acc_df['accuracy'] = accuracy_list

# get max k value(s) from k_acc_df
k_acc_max = k_acc_df.loc[k_acc_df['accuracy'] == k_acc_df['accuracy'].max()]


print('\n__________Question 1__________')

# optimal_k is set to first k value with maximum accuracy
optimal_k = int(k_acc_max.iloc[0][0])
print('Optimal k value for year 1: ', optimal_k)


print('\n__________Question 2__________')

# predict labels for year 2
output_2018.update(kNN_predict(output_2018, optimal_k, X, Y))
output_2018['Accuracy'] = output_2018[['Color', 'Pred_Color']].apply(
    fs.get_acc, axis=1)

accuracy_2018 = output_2018['Accuracy'].sum() / output_2018['Accuracy'].count()
print('Accuracy for 2018: ', round(accuracy_2018, 4))


print('\n__________Question 3__________')

# calculate true/false positive/negative values for 2018 predictions
tp_sum = output_2018[['Color', 'Pred_Color']].apply(fs.tp, axis=1).sum()
fp_sum = output_2018[['Color', 'Pred_Color']].apply(fs.fp, axis=1).sum()
tn_sum = output_2018[['Color', 'Pred_Color']].apply(fs.tn, axis=1).sum()
fn_sum = output_2018[['Color', 'Pred_Color']].apply(fs.fn, axis=1).sum()

# calculate confusion matrix
confusion_2018 = np.array([[tn_sum, fp_sum], [fn_sum, tp_sum]])
print('\nConfusion Matrix for 2018: \n', confusion_2018)


print('\n__________Question 4__________')

# calculate true positive rate and true negative rate
tpr = tp_sum / (tp_sum + fn_sum)
tnr = tn_sum / (tn_sum + fp_sum)

# print tpr and tnr values
print('\nTrue Positive Rate: ', round(tpr, 4))
print('True Negative Rate: ', round(tnr, 4))


print('\n__________Question 5__________')

# set df_2018 color value to predicted color so color_strat_knn uses pred values
df_2018['Color'] = output_2018['Pred_Color']

color_strat_knn = round(wl.color_strategy(df_2018), 2)
print('Color Strategy for 2018: ' + '$' + str(color_strat_knn))

buy_and_hold_knn = round(wl.buy_and_hold(df_2018), 2)
print('Buy & Hold for 2018: ' + '$' + str(buy_and_hold_knn))

print('\nBuy and Hold Strategy results in a larger'
      ' amount at the end of the year.')


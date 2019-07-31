import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from Assignment2 import week_labeling as wl
from Assignment3 import mu_sigma_feature_set as fs

# allow df prints to display each column
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)

data = fs.get_ticker_df()

df_2017 = data[data['Year'] == 2017].reset_index()
df_2018 = data[data['Year'] == 2018].reset_index()

input_2017 = df_2017[['Year_Week', 'Mean_Return', 'Std_Return', 'Color']].groupby('Year_Week').max().reset_index()
input_2017['Class'] = input_2017['Color'].apply(lambda x: 1 if x is 'Green' else 0)

X = input_2017[['Mean_Return', 'Std_Return']].values
Y = input_2017['Class'].values


#print(X)
#print(Y)

k_values = [3, 5, 7, 9, 11]

output_2018 = df_2018[['Year_Week', 'Mean_Return', 'Std_Return', 'Color']].groupby('Year_Week').max().reset_index()
output_2018['Class'] = output_2018['Color'].apply(lambda x: 1 if x is 'Green' else 0)


accuracy_list = []






# obtain accuracy for each k in k_values
for k in k_values:

    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X, Y)
    #new_instance = output_2018[['Mean_Return', 'Std_Return']].values

    new_instance = input_2017[['Mean_Return', 'Std_Return']].values

    prediction = knn_classifier.predict(new_instance)
    column_name = 'k_' + str(k)

    #actual_values = output_2018['Class'].values
    actual_values = input_2017['Class'].values

    #output_2018[column_name] = prediction
    input_2017[column_name] = prediction

    accuracy_name = column_name + '_acc'

    #output_2018[accuracy_name] = output_2018[['Class', column_name]].apply(fs.get_acc, axis=1)
    input_2017[accuracy_name] = input_2017[['Class', column_name]].apply(fs.get_acc, axis=1)

    #correct_count = output_2018[accuracy_name].sum()
    #total_weeks = output_2018[accuracy_name].count()

    correct_count = input_2017[accuracy_name].sum()
    total_weeks = input_2017[accuracy_name].count()

    accuracy = correct_count / total_weeks

    accuracy_list.append(accuracy)

    print(accuracy_name + ':', accuracy.round(2))


def kNN_predict(df1, k1, X1, Y1):

    knn_classifier = KNeighborsClassifier(n_neighbors=k1)
    knn_classifier.fit(X1, Y1)

    new_instance = df1[['Mean_Return', 'Std_Return']].values

    prediction = knn_classifier.predict(new_instance)

    df1['Pred_Class'] = prediction
    df1['Pred_Color'] = df1['Pred_Class'].apply(lambda x: 'Green' if x is 1 else 'Red')

    return df1

#k_acc_df = pd.DataFrame(accuracy_list, columns=['k', 'accuracy'])
k_acc_df = pd.DataFrame()
k_acc_df['k'] = k_values
k_acc_df['accuracy'] = accuracy_list

k_acc_max = k_acc_df.loc[k_acc_df['accuracy'] == k_acc_df['accuracy'].max()]


print(k_acc_df)
print(k_acc_max)

optimal_k = int(k_acc_max.iloc[0][0])

print('Optimal k value for year 1: ', optimal_k)


# predict labels for year 2


output_2018.update(kNN_predict(output_2018, optimal_k, X, Y))
output_2018['Accuracy'] = output_2018[['Color', 'Pred_Color']].apply(fs.get_acc, axis=1)

accuracy_2018 = output_2018['Accuracy'].sum() / output_2018['Accuracy'].count()
print('Accuracy for 2018: ', accuracy_2018.round(2))

tp_sum = output_2018[['Color', 'Pred_Color']].apply(fs.tp, axis=1).sum()
fp_sum = output_2018[['Color', 'Pred_Color']].apply(fs.fp, axis=1).sum()
tn_sum = output_2018[['Color', 'Pred_Color']].apply(fs.tn, axis=1).sum()
fn_sum = output_2018[['Color', 'Pred_Color']].apply(fs.fn, axis=1).sum()

confusion_2018 = np.array([[tn_sum, fp_sum], [fn_sum, tp_sum]])

print('\nConfusion Matrix for 2018: \n', confusion_2018)

tpr = tp_sum / (tp_sum + fn_sum)
tnr = tn_sum / (tn_sum + fp_sum)

print('\nTrue Positive Rate: ', tpr.round(2))
print('\nTrue Negative Rate: ', tnr.round(2))

df_2018['Color'] = output_2018['Pred_Color']

color_strat_knn = wl.color_strategy(df_2018).round(2)
print('Color Strategy for 2018: ' + '$' + str(color_strat_knn))

buy_and_hold_knn = wl.buy_and_hold(df_2018).round(2)
print('Buy & Hold for 2018: ' + '$' + str(buy_and_hold_knn))

#output_2018['']

#print(output_2018)


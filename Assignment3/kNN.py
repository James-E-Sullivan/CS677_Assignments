import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from Assignment3 import mu_sigma_feature_set as fs

data = fs.get_ticker_df()

df_2017 = data[data['Year'] == 2016].reset_index()
df_2018 = data[data['Year'] == 2017].reset_index()

input_2017 = df_2017[['Year_Week', 'Mean_Return', 'Std_Return', 'Color']].groupby('Year_Week').max().reset_index()
input_2017['Class'] = input_2017['Color'].apply(lambda x: 1 if x is 'Green' else 0)

X = input_2017[['Mean_Return', 'Std_Return']].values
Y = input_2017['Class'].values

#le = LabelEncoder()
#Y = le.fit_transform(input_2017['Color'].values)
print(X)
print(Y)

k_values = [3, 5, 7, 9, 11]

output_2018 = df_2018[['Year_Week', 'Mean_Return', 'Std_Return', 'Color']].groupby('Year_Week').max().reset_index()
output_2018['Class'] = output_2018['Color'].apply(lambda x: 1 if x is 'Green' else 0)


accuracy_list = []


for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X, Y)
    new_instance = output_2018[['Mean_Return', 'Std_Return']].values
    prediction = knn_classifier.predict(new_instance)
    column_name = 'k_' + str(k)

    actual_values = output_2018['Class'].values

    output_2018[column_name] = prediction

    accuracy_name = column_name + '_acc'

    output_2018[accuracy_name] = output_2018[['Class', column_name]].apply(fs.get_acc, axis=1)
    correct_count = output_2018[accuracy_name].sum()
    total_weeks = output_2018[accuracy_name].count()

    accuracy = correct_count / total_weeks


    #k_acc_df.append([k, accuracy])
    #k_acc_df['k'] = k

    print(accuracy_name + ':', accuracy.round(2))

print(k_values)

k_acc_df = pd.DataFrame((k_values, accuracy_list), columns=['k', 'accuracy'])

print(k_acc_df)


#output_2018['']

print(output_2018)

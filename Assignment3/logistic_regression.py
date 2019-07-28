import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Assignment2 import week_labeling as wl

ticker = 'SBUX'
input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
filename = os.path.join(input_dir, 'logistic_regression_gradient_descent.pdf')
ticker_file = os.path.join(input_dir, ticker + '.csv')
data = pd.read_csv(ticker_file)

# assign labels using week_labeling formula from assignment 2
data = wl.assign_labels(data)


def get_feature_set(df1):

    mean_return_df = df1[['Year_Week', 'Return']].groupby('Year_Week').mean().reset_index()
    mean_return_df.rename(columns={'Return': 'Mean_Return'}, inplace=True)

    std_return_df = df1[['Year_Week', 'Return']].groupby('Year_Week').std().reset_index()
    std_return_df.rename(columns={'Return': 'Std_Return'}, inplace=True)

    df2 = pd.merge(mean_return_df, std_return_df, on='Year_Week')
    df3 = pd.merge(df1, df2, on='Year_Week')

    return df3


data['Class'] = data['Color'].apply(lambda a: 1 if a == 'Green' else 0)
data = get_feature_set(data)

df_2017 = data[data['Year'] == 2017].reset_index()
df_2018 = data[data['Year'] == 2018].reset_index()

training_set = df_2017[[
    'Year_Week', 'Mean_Return', 'Std_Return', 'Color', 'Class']].groupby(
    'Year_Week').max().reset_index()
print(training_set.head())

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
            print('i:', i, 'loss: ', loss(phi, y_pred))
    print('rate: ', learning_rate, ' iterations ', iterations, ' weights: ', weights)
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


#df_2018['Feature_Values'] = df_2018[['Mean_Return', 'Std_Return']].apply(lambda b: (b[0], b[1]))


df_2018['Linear_Y'] = df_2018.apply(lambda b: predict_y(b['Mean_Return'], b['Std_Return'], weights), axis=1)
df_2018['Logistic_Y'] = df_2018[['Linear_Y']].apply(sigmoid, axis=1)
df_2018['Predicted_Color'] = df_2018['Logistic_Y'].apply(lambda c: 'Green' if c > 0.5 else 'Red')


print(df_2018.head())

print(df_2018[['Year_Week', 'Mean_Return', 'Color', 'Predicted_Color']].groupby(['Year_Week']).max())


"""
# predict:

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()


df = training_set[training_set['Color']=='Red']
plt.scatter(df['Year_Week'].values, df['Mean_Return'].values, color='red',
            s= 100, label='Class 0')

df = training_set[training_set['Color']=='Green']
plt.scatter(df['Year_Week'].values, df['Mean_Return'].values, color='green',
            s= 100, label='Class 1')


for i in range(len(training_set)):
    x_text = training_set['Year_Week'].iloc[i]
    y_text = training_set['Mean_Return'].iloc[i]
    id_text = training_set['Mean_Return'].iloc[i]
    plt.text(x_text, y_text, id_text, fontsize=14)


#plt.xlim(0, 5)
plt.ylim(df['Mean_Return'].min(),df['Mean_Return'].max())


h_1, h_2 = 4.5, 6.5


iterations=100
weights = np.zeros(x.shape[1])
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=25)
f_1 = (-weights[0]-weights[1]*h_1)/weights[2]
f_2 = (-weights[0]-weights[1]*h_2)/weights[2]
plt.plot([h_1,h_2], [f_1, f_2], color='gray', label=str(iterations) + ' iterations',
         lw=1)

iterations=250
weights = np.zeros(x.shape[1])
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=100)
f_1 = (-weights[0]-weights[1]*h_1)/weights[2]
f_2 = (-weights[0]-weights[1]*h_2)/weights[2]
plt.plot([h_1,h_2], [f_1, f_2], color='magenta', label=str(iterations) + ' iterations',
         lw=1)

iterations=1000
weights = np.zeros(x.shape[1])
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=1000)
f_1 = (-weights[0]-weights[1]*h_1)/weights[2]
f_2 = (-weights[0]-weights[1]*h_2)/weights[2]
plt.plot([h_1,h_2], [f_1, f_2], color='blue', label=str(iterations) + ' iterations', lw=2)


plt.xlabel('Year_Week')
plt.ylabel('Mean_Return')
plt.legend(loc='upper left')
plt.text(5.5,14,'learn.rate = ' + str(learning_rate), fontsize=14)

root_name = 'logistic_regression_gradient_descent_'+str(learning_rate)
root_name = root_name.replace('.', '_')

filename = os.path.join(input_dir,root_name + '_new.pdf')
plt.savefig(filename)
plt.show()

"""



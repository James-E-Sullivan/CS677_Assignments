# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:46:58 2018

@author: epinsky
"""

# sigmoid function
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_dir = r'C:\Users\james\BU MET\CS677\Datasets'
filename = os.path.join(input_dir,'logistic_regression_gradient_descent.pdf')


data = pd.DataFrame(
        {'id': [ 1,2,3,4,5,6,7,8],
        'Label': ['green', 'green', 'green', 'green', 'red', 'red', 'red', 'red'],
        'Height': [5, 5.5, 5.33, 5.75, 6.00, 5.92,  5.58, 5.92],
        'Weight': [100, 150, 130, 150, 180, 190, 170, 165], 
        'Foot': [6, 8, 7, 9, 13, 11, 12, 10]},
         columns = ['id', 'Height', 'Weight', 'Foot', 'Label']
        )

data['Class'] = data['Label'].apply(lambda x: 1 if x=='green' else 0)
x = data[['Height', 'Foot']].values
y = data['Class'].values

N = len(y)
learning_rate = 0.01

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def loss(h, y):
    return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()

def add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)


threshold = 0.1
n=5
iterations=10000
# initialize the weights

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


# predict:

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()


df = data[data['Label']=='red']
plt.scatter(df['Height'].values, df['Foot'].values, color='red',
            s= 100, label='Class 0')

df = data[data['Label']=='green']
plt.scatter(df['Height'].values, df['Foot'].values, color='green',
            s= 100, label='Class 1')

for i in range(len(data)):
    x_text = data['Height'].iloc[i] + 0.05
    y_text = data['Foot'].iloc[i] + 0.2
    id_text = data['id'].iloc[i]
    plt.text(x_text, y_text, str(id_text), fontsize=14)


plt.xlim(4,7)
plt.ylim(4,15)


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



plt.xlabel('Height')
plt.ylabel('Foot')
plt.legend(loc='upper left')
plt.text(5.5,14,'learn.rate = ' + str(learning_rate), fontsize=14)

root_name = 'logistic_regression_gradient_descent_'+str(learning_rate)
root_name = root_name.replace('.', '_')

filename = os.path.join(input_dir,root_name + '_new.pdf')
plt.savefig(filename)
plt.show()    












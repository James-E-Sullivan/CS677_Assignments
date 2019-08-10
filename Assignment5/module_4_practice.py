"""
This example is useless because Y was not defined
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame(
    {'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     'Weather': ['sunny', 'rainy', 'sunny', 'rainy', 'sunny', 'overcast',
                 'sunny', 'overcast', 'rainy', 'rainy'],
     'Temperature': ['hot', 'mild', 'cold', 'cold', 'cold', 'mild', 'hot',
                     'hot', 'hot', 'mild'],
     'Wind': ['low', 'high', 'low', 'high', 'high', 'low', 'low', 'high',
              'high', 'low'],
     'Play': ['no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no',
              'yes']},
    columns=['Day', 'Weather', 'Temperature', 'Wind', 'Play'])

input_data = data[['Weather', 'Temperature', 'Wind']]
dummies = [pd.get_dummies(data[c]) for c in input_data.columns]
binary_data = pd.concat(dummies, axis=1)
classifier_dummies = [pd.get_dummies(data.Play)]
binary_labels = pd.concat(classifier_dummies, axis=1)

X = binary_data[0:10].values
Y = binary_labels[0:10].values
le = LabelEncoder()

NB_classifier = MultinomialNB().fit(X, Y)
new_instance = np.asmatrix([0, 0, 1, 1, 0, 0, 0, 1])
prediction = NB_classifier.predict(new_instance)

print(prediction)




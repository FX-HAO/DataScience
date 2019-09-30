#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import metrics

# def fizz_buzz_encode(i):
#     if   i % 15 == 0: return np.array([0, 0, 0, 1])
#     elif i % 5  == 0: return np.array([0, 0, 1, 0])
#     elif i % 3  == 0: return np.array([0, 1, 0, 0])
#     else:             return np.array([1, 0, 0, 0])
def fizz_buzz_encode(i):
    if i % 15 == 0: return 1
    if i % 5 == 0: return 2
    if i % 3 == 0: return 3
    else: return 4

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

NUM_DIGITS = 16

X = np.array([binary_encode(i, NUM_DIGITS) for i in range(2 ** NUM_DIGITS)])
Y = np.array([fizz_buzz_encode(i) for i in range(2 ** NUM_DIGITS)])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, verbose=1)
y_pred = model.predict(X_test).astype('int')
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

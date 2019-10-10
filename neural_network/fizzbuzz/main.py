#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import metrics

def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

NUM_DIGITS = 16

X = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
Y = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=1)
y_pred = model.predict_classes(X_test)
score = model.evaluate(X_test, y_test, batch_size=2000)
print(score)

X = np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
y = model.predict_classes(X)
result = []
for i in range(len(y)):
    if y[i] == 3:
        result.append('fizzbuzz')
    elif y[i] == 2:
        result.append('buzz')
    elif y[i] == 1:
        result.append('fizz')
    elif y[i] == 0:
        result.append(str(i+1))

print(result)

#!/usr/bin/env python3

# Issue: https://www.kaggle.com/puneet6060/intel-image-classification

import os
import numpy as np
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(path):
    X = []
    Y = []
    i = 0
    for category in os.listdir(path):
        if '.' in category:
            continue
        for file in os.listdir(path + category):
            if file.split('.')[1] != 'jpg':
                continue

            i += 1
            img_array = np.array(load_img(path + category + '/' + file, grayscale=True))
            if img_array.shape == (150, 150):
                X.append(img_array)
                Y.append(category)
                if i % 100 == 0:
                    print(i)
            else:
                print("skip %d" % i)
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

X_train, Y_train = load_data('seg_train/')
X_test, Y_test = load_data('seg_test/')

encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train)
Y_test = encoder.transform(Y_test)

X_train = X_train.reshape(-1, 150, 150, 1)
X_test = X_test.reshape(-1, 150, 150, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255

Y_train_one_hot = to_categorical(Y_train)
Y_test_one_hot = to_categorical(Y_test)

X_train, X_valid, label_train, label_valid = train_test_split(X_train, Y_train_one_hot, test_size=0.2, random_state=13)
print(X_train.shape, X_valid.shape, label_train.shape, label_valid.shape)


# Model the Data
batch_size = 64
epochs = 10
num_classes = 6

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(150,150,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fashion_train = fashion_model.fit(X_train, label_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, label_valid))

test_eval = fashion_model.evaluate(X_test, Y_test_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
# Test loss: 0.7748877938862951
# Test accuracy: 0.7731373310089111
# Hmm... not good enough


# plot loss and accuracy
import matplotlib.pyplot as plt

accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.figure()
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()


# predict test
predicted_classes = fashion_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

correct = np.where(predicted_classes==Y_test)[0]
print("Found %d correct labels" % len(correct))
plt.figure()
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(150, 150), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], Y_test[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=Y_test)[0]
print("Found %d incorrect labels" % len(incorrect))
plt.figure()
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(150, 150), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], Y_test[incorrect]))
    plt.tight_layout()

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(Y_test, predicted_classes, target_names=target_names))

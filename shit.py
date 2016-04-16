__author__ = 'Nikita and Sasha'

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.reshape(X_train.astype('float32'), (X_train.shape[0], 28 * 28))/255
X_test = np.reshape(X_test.astype('float32'), (X_test.shape[0], 28 * 28))/255
model = Sequential()
model.add(Dense(init='uniform', input_dim=784, output_dim=392))
model.add(Activation('relu'))
model.add(Dense(init='uniform', input_dim=392, output_dim=784))
model.add(Activation('relu'))
sgd = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=sgd, loss='mse')
model.fit(X_train, X_train, nb_epoch=100, batch_size=5000, verbose=1, show_accuracy=1)
model.layers[0].get_weights()
model_two = Sequential()
model_two.add(Dense(25, init='uniform', input_dim=784))
model_two.add(Activation('relu'))
model_two.add(Dense(10, init='uniform', input_dim=784))
model_two.add(Activation('softmax'))
model_two.compile(optimizer=sgd, loss='categorical_crossentropy')
model_two.fit(X_train, Y_train, nb_epoch=500, batch_size=5000, verbose=1, show_accuracy=1,
              validation_data=(X_test, Y_test))

from keras.preprocessing import sequence
import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.pkl",
                                                      nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      test_split=0.1)
k = 0
for i in range(X_train.shape[0]):
    k = max(k, len(X_train[i]))
for i in range(X_test.shape[0]):
    k = max(k, len(X_test[i]))

X_train = sequence.pad_sequences(X_train, k)
X_test = sequence.pad_sequences(X_test, k)

max_features = max(numpy.max(X_train), numpy.max(X_test))
maxlen = k
batch_size = 32

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, show_accuracy= True)

# result = model.predict_proba(X)

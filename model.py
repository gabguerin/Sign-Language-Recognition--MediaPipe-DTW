from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
import os
import time


class Seq2seqClassifier(Sequential):
    def __init__(self, input_shape, num_classes):
        Sequential.__init__(self)
        self.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
        self.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
        self.add(LSTM(64, return_sequences=False, activation='relu'))
        self.add(Dense(64, activation='relu'))
        self.add(Dense(32, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))

    def train(self, X, y, name=str(time.time()), epochs=200):
        self.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.fit(X, y, epochs=epochs)
        path = os.path.join("weights", name + '.h5')
        self.save(path)

    def load(self, name):
        path = os.path.join("weights", name + '.h5')
        self.load_weights(name + '.h5')
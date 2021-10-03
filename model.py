from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
import os


class Seq2seqClassifier(Sequential):
    def __init__(self, input_shape, num_classes):
        Sequential.__init__(self)
        self.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
        self.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
        self.add(LSTM(64, return_sequences=False, activation='relu'))
        self.add(Dense(64, activation='relu'))
        self.add(Dense(32, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))



    def train(self, X, y, epochs=200):
        self.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.fit(X, y, epochs=epochs)

        name = f"nseq_{X.shape[1]}__nkpts_{X.shape[2]}__nclass_{y.shape[1]}.h5"
        path = os.path.join("weights", name)
        self.save(path)

    def load(self, name):
        if len(name) == 0:
            name = os.listdir("weights")[0]
        path = os.path.join("weights", name)
        self.load_weights(path)
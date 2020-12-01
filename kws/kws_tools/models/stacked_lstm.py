from tensorflow.keras.layers import  Dropout
from tensorflow.keras.layers import Activation, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2




class StackedLSTM:
    @staticmethod
    def build(samples,timesteps, features, depth, classes):


        model= Sequential(name='stacked_lstm')
        input_shape = (samples, timesteps, features)

        model.add(LSTM(30, return_sequences=True, batch_input_shape=input_shape, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        model.add(LSTM(200, return_sequences=True, kernel_regularizer=l2(0.01)))
        model.add(Dense(30, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.summary()
        return model


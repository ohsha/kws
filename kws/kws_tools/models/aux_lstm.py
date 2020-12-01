from tensorflow.keras.layers import Activation, Dense,  LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


class AuxLSTM:
    @staticmethod
    def build(samples, timesteps, features, depth, classes, stateful=False):
        f1 = 128
        f2 = 30

        model= Sequential(name='AuxLSTM')
        input_shape = (timesteps, features)
        batch_shape = (samples, timesteps, features)

        model.add(TimeDistributed(Dense(f1, activation='relu'),
                                  batch_input_shape=batch_shape, input_shape=input_shape, name='TimeDist_128'))
        model.add(TimeDistributed(Dense(f2, activation='relu'), name='TimeDist_30'))
        model.add(LSTM(150, return_sequences=True, dropout=0.3,kernel_regularizer=l2(0.01),
                       stateful=stateful, name='LSTM_150'))
        model.add(LSTM(50, return_sequences=True, dropout=0.3,kernel_regularizer=l2(0.01),
                       stateful=stateful, name='LSTM_50'))
        #model.add(Dense(30, activation='relu', kernel_regularizer=l2(0.001), name='FC_30'))
        #model.add(Dropout(0.3, name='Dropout'))
        model.add(LSTM(classes,return_sequences=True, stateful=stateful, name='FC_11', activation=None))
        model.add(Activation("softmax", name='Softmax'))

        model.summary()
        return model


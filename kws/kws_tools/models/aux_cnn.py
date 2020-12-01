from tensorflow.keras.layers import BatchNormalization, Conv2D, Input
from tensorflow.keras.layers import  Flatten, Dropout
from tensorflow.keras.layers import Activation, Dense, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


class AuxCNN:
    @staticmethod
    def build(width, height, depth, classes):
        f1 = 128
        f2 = 30

        model= Sequential(name='AuxiliaryNetwork')
        input_shape = (height, width, depth,)

        model.add(Input(shape=input_shape))
        model.add(Conv2D(filters=f1, kernel_size=(1, 256), padding="valid", kernel_regularizer=l2(0.01)))
        model.add(Activation("relu"))
        model.add(Reshape((99, f1, 1)))

        model.add(Dropout(0.4))
        model.add(Conv2D(filters=f2, kernel_size=(1,f1), padding="valid", kernel_regularizer=l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Flatten())
        model.add(Dense(200, kernel_regularizer=l2(0.01)))
        model.add(Activation("relu"))

        model.add(Dropout(0.3))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.summary()
        return model


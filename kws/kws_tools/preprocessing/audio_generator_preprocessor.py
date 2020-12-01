
from ..audio_data_generator import AudioDataGenerator
import numpy as np

class AudioGeneratorPreprocessor():

    def __init__(self,featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 roll_range=0.,
                 brightness_range=None,
                 zoom_range=0.,
                 shift=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 noise=None,
                 validation_split=0.0):



        self.aug = AudioDataGenerator(featurewise_center=featurewise_center,
                 samplewise_center=samplewise_center,
                 featurewise_std_normalization=featurewise_std_normalization,
                 samplewise_std_normalization=samplewise_std_normalization,
                 zca_whitening=zca_whitening,
                 zca_epsilon=zca_epsilon,
                 roll_range=roll_range,
                 brightness_range=brightness_range,
                 zoom_range=zoom_range,
                 shift=shift,
                 fill_mode=fill_mode,
                 cval=cval,
                 horizontal_flip=horizontal_flip,
                 rescale=rescale,
                 preprocessing_function=preprocessing_function,
                 data_format=data_format,
                 noise=noise,
                 validation_split=validation_split)





    def preprocessor(self, data):

        data = data[np.newaxis, :, :]
        data = next(self.aug.flow(data, batch_size=1))

        return data.reshape(data.shape[1], 1)



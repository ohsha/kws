import librosa
import numpy as np

class MFCCPreprocessor:

    def __init__(self, n_mfcc=21, n_mels=21,  include_delta=0, ref=np.max):

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.ref = ref
        self.include_delta = include_delta


    def preprocessor(self, data, rate):
        S = librosa.feature.melspectrogram(data, sr=rate, n_mels=self.n_mels)
        log_S = librosa.power_to_db(S, ref=self.ref)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=self.n_mfcc)
        if self.include_delta == 1:
            return librosa.feature.delta(mfcc)
        elif self.include_delta == 2:
            return librosa.feature.delta(mfcc, order=2, mode='nearest' )
        else:
            return mfcc

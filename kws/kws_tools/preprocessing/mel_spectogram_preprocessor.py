import librosa
import numpy as np



class MelSpectogramPreprocessor:

    def __init__(self, n_mels, include_log=True, ref=np.max):

        self.n_mels = n_mels
        self.include_log = include_log
        self.ref = ref


    def preprocessor(self, data, rate):

        S = librosa.feature.melspectrogram(data, sr=rate, n_mels=self.n_mels)

        if self.include_log:

            log_S = librosa.power_to_db(S, ref=self.ref)
            return log_S
        else:
            return S


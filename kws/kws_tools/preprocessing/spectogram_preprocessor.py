from scipy import signal
import numpy as np


class SpectogramPreprocessor:

    def __init__(self, window_size=28, step_size=10, eps=1e-10, include_log=True):
        self.window_size = window_size
        self.step_size = step_size
        self.eps = eps
        self.include_log = include_log

    def preprocessor(self, data, rate):
        nperseg = int(self.window_size * rate // 1e3)
        noverlap = int(self.step_size * rate //1e3)
        freqs, times, spec = signal.spectrogram(data, fs=rate, window='hann',
                                                nperseg=nperseg, noverlap=noverlap,
                                                detrend=False)
        if self.include_log:
            return np.log(spec.astype(np.float32) + self.eps)

        else:
            return spec.astype(np.float32)
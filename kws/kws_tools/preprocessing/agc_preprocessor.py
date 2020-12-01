import numpy as np

class AGCPreprocessor:

    def __init__(self, max_pcm=32768, min_gain=0.5, max_gain=4, alpha=1.0001):
        self.MAX_PCM = max_pcm
        self.MIN_GAIN = min_gain
        self.MAX_GAIN = max_gain
        self.ALPHA = alpha
        self.PEAK_THRESHOLD = min_gain * max_pcm


    def preprocessor(self, audio_data):


        audio_after_agc = []
        peaks = []
        parts_to_count = 3
        j = 0
        gain = self.MAX_GAIN
        for point in audio_data:
            peak = np.abs(point)
            peaks.append(peak)
            if len(peaks) > parts_to_count:
                peak = np.max(peaks[j - parts_to_count : j])
            j += 1

            if peak * gain > self.PEAK_THRESHOLD:
                gain = self.PEAK_THRESHOLD / peak
            else:
                gain = min(gain * self.ALPHA, self.MAX_GAIN)

            audio_after_agc.append(point * gain)

        return np.array(audio_after_agc)




import numpy as np

class FramePreprocessor:

    def __init__(self, f_size, steps):
        self.f_size = f_size
        self.step = steps

    def preprocessor(self, data, rate=16000):
        frames = []
        for i in range(0, len(data)-self.step, self.step):
            frame = data[i : i + self.f_size]
            frames.append(frame)

        return np.array(frames)
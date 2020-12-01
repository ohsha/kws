import numpy as np
from scipy.io import wavfile


class WavLoader:
    """
    This class

    """


    def __init__(self, preprocessors=None):

        self.preprocessors = preprocessors


    def _padding(self, data, rate):

        if len(data) < rate:
            data = np.pad(data, (0, max(0, rate - len(data))))

        else:

            data = data[:rate]

        return data

    def load(self, wav_paths, extraction_method=None, with_scale=False, verbose=-1):


            data_list = []
            rate_list = []
            samples_list = []
            labels_list = []

            for (i, wav_path) in enumerate(wav_paths):

                rate, data = wavfile.read(wav_path)
                if len(data) != rate:

                    data = self._padding(data, rate)

                if with_scale:
                    data = data / 256.0

                data = data.astype(np.float32)

                # extraction_method should be a lambda function
                # for example: extraction_method = lambda p: p.split(os.path.sep)[-2]
                if extraction_method is None:

                    label = wav_path.split('_')[-1].split('.')[0]
                else:
                    label = extraction_method(wav_path)

                #  run over all preprocessor
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        samples = p.preprocessor(data, rate)

                        samples_list.append(samples)

                data_list.append(data)
                rate_list.append(rate)
                labels_list.append(label)

                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print ("[INFO] processed {}/{}".format((i + 1), len(wav_paths)))

            if samples_list is not None:
                X = [rate_list, data_list, samples_list]
                y = labels_list
                return X, y
            else:
                X = [rate_list, data_list]
                y = labels_list
                return X, y
            #return data, rate, labels


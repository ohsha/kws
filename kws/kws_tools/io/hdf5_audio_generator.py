from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py

class HDF5AudioGenerator:
    """
    This class allows to handle batch feeding (like DataGenerator of Keras, in order to handle large data sets).
    It also allows to process each batch with pre-processors, before saving the output:
    * Reads a raw data from the HDF5 dataset
    * Process each batch (on-the-fly) with the given pre-processors
    * Writes the processed data to the training / generator in batches.

    ** all the manipulations on the data that should happen during the flow are done here.
    """
    def __init__(self, db_path, batch_size, classes, preprocessors=None, aug=None, one_hot=True, n_seq=99, mode='aux'):

        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.one_hot = one_hot
        self.classes = classes
        self.db = h5py.File(db_path)
        self.n_instances = self.db['labels'].shape[0]
        self.mode = mode
        self.n_frames = n_seq

        possible_modes = ['aux', 'lstm']
        assert  self.mode in possible_modes, 'An unknown mode provided.'


    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in np.arange(0, self.n_instances - self.batch_size, self.batch_size):
                samples = self.db['samples'][i: i + self.batch_size]
                labels = self.db['labels'][i: i + self.batch_size]
                if self.mode == 'lstm':
                    # labeling each frame of the wav file.
                    # note, for partial labeling (end of word), replace between the comments and uncomments.
                    labels = np.repeat(labels, self.n_frames).reshape(labels.shape[0], self.n_frames)

                    # temp_list = labels
                    # labels = []
                    # for i, l in enumerate(temp_list):
                    #     labels.append(np.repeat(l, self.n_frames))

                if self.one_hot:
                    labels = to_categorical(labels, self.classes)

                # generates augmentation on-the-fly
                if self.aug is not None:
                    (samples, labels) = next(self.aug.flow(samples, labels, batch_size=self.batch_size))

                if self.preprocessors is not None:
                    processed = []
                    labeled = []
                    # run over all the processors and generate processed data.
                    for i, sample in enumerate(samples):
                        for p in self.preprocessors:
                            sample = p.preprocessor(sample)

                        processed.append(sample)

                    samples = np.array(processed, dtype=np.float16)

                if self.mode == 'lstm':
                    samples = samples.squeeze()
                    labels = labels.squeeze()

                yield (samples, labels)

            epochs += 1


    def close(self):

        self.db.close()
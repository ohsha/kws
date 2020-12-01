import numpy as np
import h5py


class HDF5AudioSimulator:
    """
    This class illustrate the real-world situation and streams the data frame by frame

    """
    def __init__(self, db_path, batch_size, classes, preprocessors=None,  n_seq=99):

        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.classes = classes
        self.db = h5py.File(db_path)
        self.n_instances = self.db['samples'].shape[0]
        self.n_steps = n_seq

    def generator(self, step=0, frm=0, passes=np.inf):

        epochs = 0
        while epochs < passes:
            for i in np.arange(step, self.n_instances, self.batch_size):
                current_step = step
                samples = self.db['samples'][i: i + self.batch_size]

                if self.preprocessors is not None:
                    processed = []
                    for i, sample in enumerate(samples):
                        for p in self.preprocessors:
                                sample = p.preprocessor(sample)

                        processed.append(sample)

                    samples = np.array(processed, dtype=np.float16)

                samples = samples.squeeze()

                while step == current_step:
                    sample = samples[frm]
                    sample = sample[np.newaxis, np.newaxis, :]
                    yield sample

            epochs += 1

    def close(self):

        self.db.close()

#
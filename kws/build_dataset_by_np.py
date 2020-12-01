from config import config_kws as config
from kws_tools.io import HDF5DatasetWriter

from sklearn.preprocessing import LabelEncoder
import numpy as np
import progressbar
import os

"""
This module creates a dataset for training in HDF5 format.
the raw data are required for this module should be stored on .npy format:
    -train.npy 
    -test.npy
    -val.npy

"""

train_path = os.path.join(config.DATASET_PATH, r'train.npy')
test_path = os.path.join(config.DATASET_PATH, r'test.npy')
val_path = os.path.join(config.DATASET_PATH, r'val.npy')

print('[INFO] loading data...')
train_data = np.load(train_path)
test_data = np.load(test_path)
val_data = np.load(val_path)

X_train, y_train = train_data[0], train_data[1]
y_train[y_train == 'background_noise'] = 'a_unknown'
X_test, y_test = test_data[0], test_data[1]
X_val, y_val = val_data[0], val_data[1]

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_val = le.fit_transform(y_val)

datasets = [
    ('train', X_train, y_train, config.TRAIN_HDF5),
    ('test', X_test, y_test, config.TEST_HDF5),
    ('val', X_val, y_val, config.VAL_HDF5)
]
# create the HDF5 datasets.
for (data_type, X, y, output) in datasets:
    print('[INFO] building {}...'.format(output))
    #writer = HDF5DatasetWriter(dims=(len(X),99, 256, 1), output_path=output, data_key='samples')
    writer = HDF5DatasetWriter(dims=(len(X), 16000), output_path=output, data_key='samples')
    # progress bar
    widgets = ["building Datasets: ", progressbar.Percentage(), " ",
               progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(X), widgets=widgets).start()

    for i, (x,y) in enumerate(zip(X, y)):
        writer.add([x], [y])
        pbar.update(i)

    writer.store_class_labels(le.classes_)
    pbar.finish()
    writer.close()

print('[INFO] the dataset has been created.')

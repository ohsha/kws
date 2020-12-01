
from config import config_kws as config
from kws_tools.preprocessing import WavLoader
from kws_tools.dev_tools import list_files, splitting_by_users
from kws_tools.io import HDF5DatasetWriter

from sklearn.preprocessing import  LabelEncoder
import progressbar
import os

#TODO comments and prints

"""
This module creates a dataset for training in HDF5 format.
The raw data required for this module are directories
that contain organized .wav files, where each label has
a folder with the name of the label.

for example:

├── recording
│   ├── down
│   │   └── rec_1.wav
│   │   :
│   │   └── rec_100.wav
:   :   :
:   :   :
│   ├── go
│   │   └── rec_301.wav
│   │   :
│   │   └── rec_400.wav


Note:
 - specifying the parent path at the config.DATASET_PATH is needed.
 - For the 'unknown' label - you can keep the folders that belongs
   to this label under the folder 'unknown'.
   for instance:
   
    ├── unknown
    │   ├── bird
    │   │   :
    │   │   └── rec_700.wav
    │   ├── happy
    │   │   └── rec_901.wav
   
 - make sure to validate the splitting index at the
       extraction_method function below (line 65)
       
"""
# loading data from the parent path
wav_paths = list(list_files(config.DATASET_PATH, file_types=('wav')))

split_size = config.NUM_TEST_DATA + config.NUM_VAL_DATA
train_paths, temp_paths = splitting_by_users(wav_paths, split_size)

val_paths = temp_paths[0 : config.NUM_VAL_DATA]
test_paths = temp_paths[config.NUM_VAL_DATA :]
temp_paths = []

wl = WavLoader()
# extract the label from the path by specifying the right pattern.
# note, the index of the splitting depends the local dataset path
# for instance, the 10th splitting in my path points on the name of the folder (the label).
extraction_method = lambda p: p.split(os.path.sep)[10]

# loading the paths into numpy array
print('[INFO] loading data...')
train, y_train = wl.load(train_paths, extraction_method=extraction_method, with_scale=True, verbose=200)
test, y_test = wl.load(test_paths, extraction_method=extraction_method, with_scale=True, verbose=200)
val, y_val = wl.load(val_paths, extraction_method=extraction_method, with_scale=True, verbose=200)


X_train = train[1]
X_test = test[1]
X_val = val[1]

train, test, val = [], [], []

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
    writer = HDF5DatasetWriter(dims=(len(X), 16000), output_path=output, data_key='samples')
    widgets = ["building Datasets: ", progressbar.Percentage(), " ",
               progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(X), widgets=widgets).start()

    for i, (x,y) in enumerate(zip(X, y)):
        writer.add([x], [y])
        pbar.update(i)

    writer.store_class_labels(le.classes_)
    pbar.finish()
    writer.close()

print('[INFO] dataset has been created.')


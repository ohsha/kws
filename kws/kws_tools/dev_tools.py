import matplotlib
matplotlib.use("tkagg")

from tensorflow.keras.layers import Conv2D
import simplejson as json
import numpy as np
import random
import os

file_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_files(base_path, contains=None, file_types=file_types):
    # return the set of valid files.
    return _get_files(base_path, valid_exts=file_types, contains=contains)


def _get_files(base_path, valid_exts=None, contains=None):
    for (root_dir, dir_names, file_names) in os.walk(base_path):
        for filename in file_names:
            if contains is not None and filename.find(contains) == -1:
                continue
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            # check to see if the file is valid and should be processed
            if valid_exts is None or ext.endswith(valid_exts):
                file_path = os.path.join(root_dir, filename)
                yield file_path


def splitting_by_users(data, flag):
    """
    The recording files are organized so each person's records got a unique ID to distinguish between different persons.
    This function is splitting the dataset to train and test, while avoiding a person's records will be in both train and test data sets.
    """
    print('[INFO] splitting data by user ID...')
    random.shuffle(data)
    data = np.array(data)
    user_extraction = np.array([u.split(os.path.sep)[-1].split('_')[0] for u in data])
    user_id, counts = np.unique(user_extraction, return_counts=True)

    divided = []
    deleted_idx = np.array([])
    sum_ = 0
    for i, id in enumerate(user_id):
        x = sum_ + counts[i]
        indices = np.nonzero(user_extraction == id)
        if x <= flag:
            sum_ = x
            divided.extend(data[indices])
            deleted_idx = np.append(deleted_idx, indices)
            if x == flag:
                data = np.delete(data, deleted_idx)
                return data.tolist(), divided
        else:
            continue


def export_model_details(model, path, name=None, cr=None):
    """
    export the final summary report
    """
    if name is not None:
        path = os.path.join(path, name)

    hash = '#'*30
    model_conf = model.get_config()
    opt_conf = model.optimizer.get_config()

    with open(path, 'w') as f:
        if cr is not None:
            f.write('\n  {}\t  CLASSIFICATION REPORT \t {}\n\n\n'.format(hash, hash))
            f.write('\n' + cr + '\n')

        f.write('\n  {}\t  SUMMARY \t {}\n\n\n'.format(hash, hash))
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write('\n  {}\t  OPTIMIZER CONFIGURATION \t {}\n\n\n'.format(hash, hash))
        f.write(json.dumps(str(opt_conf), indent=4, sort_keys=True))

        f.write('\n  {}\t  LAYERS CONFIGURATION \t {}\n\n\n'.format(hash, hash))
        f.write(json.dumps(model_conf, indent=4, sort_keys=True))

    print('[INFO] {} saved.'.format(path))


def set_weights_to_model(model, weights, ltype='conv'):
    i = 0
    for layer in model.layers:
        if type(layer) == Conv2D:
            if ltype == 'dense':
                layer.set_weights(weights[i][0, :, 0, :])
            elif ltype == 'conv':
                layer.set_weights(weights[i])
            else:
                print('[INFO] type is not supported.')
                break

            layer.trainable = False
            i +=1
    print('[INFO] new weights were set to the model...')













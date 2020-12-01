from config import config_kws as config
from kws_tools.preprocessing import FramePreprocessor
from kws_tools.io import HDF5AudioGenerator
from kws_tools.dev_tools import *

from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model

"""
evaluating the auxiliary model

"""

fp = FramePreprocessor(f_size=256, steps=160)

print('[INFO] loading model...')
model = load_model(config.SAVED_MODEL_PATH)

batch_size = 32
# creating the data generator object
gen_test = HDF5AudioGenerator(config.TEST_HDF5, batch_size=batch_size,
                              preprocessors=[fp],
                              mode='aux',
                              classes=config.NUM_CLASSES)

steps_per_test = gen_test.n_instances // batch_size

print('[INFO] evaluating model...')
predictions = model.predict_generator(gen_test.generator(), steps=steps_per_test,
                                     max_queue_size=batch_size*2)

y_pred = [p.argmax(axis=0) for p in predictions]

y_test = gen_test.db['labels'].value
if predictions.shape[0] < y_test.shape[0]:
    y_test = y_test[0 : predictions.shape[0]]

classes_name = 'unknown down go left no off on right stop up yes'.split()
cr = classification_report(y_test, y_pred, target_names=[str(x) for x in classes_name])
print(cr)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

export_model_details(model, path=config.MODEL_SUMMARY_PATH, cr=cr)

# conv_weights = [layer.get_weights() for layer in model.layers if type(layer) == Conv2D]
# np.save(config.WEIGHTS_PATH, arr=conv_weights)
gen_test.close()


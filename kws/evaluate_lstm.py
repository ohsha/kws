from config import config_kws as config
from kws_tools.preprocessing import FramePreprocessor, AGCPreprocessor, AudioGeneratorPreprocessor
from kws_tools.io import HDF5AudioGenerator
from kws_tools.dev_tools import export_model_details

from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd

aug = AudioGeneratorPreprocessor()
fp = FramePreprocessor(f_size=256, steps=160)

print('[INFO] loading model...')
model = load_model(config.SAVED_MODEL_PATH)
batch_size = 32

gen_test = HDF5AudioGenerator(config.TEST_HDF5, batch_size=batch_size,
                              preprocessors=[fp],
                              mode='lstm',
                              classes=config.NUM_CLASSES)

steps_per_test = gen_test.n_instances // batch_size

print('[INFO] evaluating model...')
predictions = model.predict_generator(gen_test.generator(), steps=steps_per_test,
                                     max_queue_size=batch_size*2)

pred_test = [p.argmax(axis=1) for p in predictions]
# check who's the label from the last 20 frames is the most frequently
# and choose it as a prediction
y_pred = [pd.Series(p[80:]).value_counts().index[0] for p in pred_test]

y_test = gen_test.db['labels'].value
if predictions.shape[0] < y_test.shape[0]:
    y_test = y_test[0 : predictions.shape[0]]


classes_name = 'unknown down go left no off on right stop up yes'.split()
cr = classification_report(y_test, y_pred, target_names=[str(x) for x in classes_name])
print(cr)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

export_model_details(model, path=config.MODEL_SUMMARY_PATH, cr=cr)
gen_test.close()




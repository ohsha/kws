
from config import config_kws as config
from kws_tools.io import HDF5AudioGenerator
from kws_tools.preprocessing import FramePreprocessor

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model

fp = FramePreprocessor(f_size=256, steps=160)

print('[INFO] loading data...')
batch_size = 32

gen_train = HDF5AudioGenerator(config.TRAIN_HDF5, batch_size=batch_size,
                               preprocessors=[fp], mode='lstm', classes=config.NUM_CLASSES)
gen_val = HDF5AudioGenerator(config.VAL_HDF5, batch_size=batch_size,
                             preprocessors=[fp], mode='lstm', classes=config.NUM_CLASSES)

print('[INFO] model compiling...')
model = load_model(config.SAVED_MODEL_PATH)

monitor = TensorBoard(log_dir=config.MONITORING_PATH, histogram_freq=1, write_images=False, embeddings_freq=1)
checkpoint = ModelCheckpoint(config.CHECKPOINT_PATH, monitor='loss', verbose=0, save_best_only=True)
callbacks = [monitor, checkpoint]

epochs = 20
# optimizer = Adam(0.0001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
steps_per_epoch = (gen_train.n_instances)// batch_size
steps_per_val = (gen_val.n_instances)// batch_size
print('[INFO] model training...')
H = model.fit_generator(gen_train.generator(), steps_per_epoch=steps_per_epoch,
                        validation_data=gen_val.generator(), validation_steps=steps_per_val,
                        epochs=epochs, max_queue_size=2*batch_size,
                        callbacks=callbacks,
                        verbose=1)

model.save(config.MODEL_PATH)

gen_train.close()
gen_val.close()
import os

current_path = os.path.dirname(__file__)
project_path = os.path.split(current_path)[0]

# fill here the name of the model after the aux training.
model_name = '19968_checkpoint'
model_dir = 'checkpoints'

PID = os.getpid()
print(PID)

USE_AUX = True

NUM_CLASSES = 11
NUM_VAL_DATA = 15170
NUM_TEST_DATA =3000

TRAIN_HDF5 = os.path.join(project_path, r'dataset\hdf5\train.hdf5')
TEST_HDF5 = os.path.join(project_path, r'dataset\hdf5\test.hdf5')
VAL_HDF5 = os.path.join(project_path, r'dataset\hdf5\val.hdf5')
INFER_HDF5 = os.path.join(project_path, r'dataset\hdf5\infer.hdf5')

DATASET_PATH = os.path.join(project_path, r'dataset\recordings\audio')
OUTPUT_PATH = os.path.join(project_path, r'output')
SAVED_MODEL_PATH = os.path.join(project_path, 'output\{}\{}.hdf5'.format(model_dir,model_name))
MODEL_SUMMARY_PATH = os.path.join(project_path, r'output\summary\{}_summary.txt'.format(model_name)) # report summary
MODEL_PATH = os.path.join(project_path, r'output\models\{}_model.hdf5'.format(PID))
CHECKPOINT_PATH = os.path.join(project_path, r'output\checkpoints\{}_checkpoint.hdf5'.format(PID)) # ModelCheckpoint
MONITORING_PATH = os.path.join(project_path, r'output\monitors', str(PID)) # Tensorboard
GRAPH_PATH = os.path.join(project_path, r'output\graphs\{}_graph.jpg'.format(PID))

WEIGHTS_PATH = os.path.join(project_path, r'output\weights\{}_weights.npy'.format(model_name.split('_')[0]))
SAVING_WEIGHTS_PATH = os.path.join(project_path, r'output\weights\{}_weights.npy'.format(PID))

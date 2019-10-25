'''
Fine-tune PredNet model trained for t+1 prediction for up to t+5 prediction.
'''
from data_utils import SequenceGenerator
from prednet import PredNet
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input
from keras.models import Model, model_from_json
from keras import backend as K
import os
import numpy as np
import pandas as pd
np.random.seed(123)


# Define loss as MAE of frame predictions after t=0
# It doesn't make sense to compute loss on error representation,
# since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    # 0.5 to match scale of loss when trained in error mode
    # (positive and negative errors split)
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)


EXP_ID = '20191023150622'
DATA_DIR = '../inputs/hkl/'
WEIGHTS_DIR = './logs/{}/'.format(EXP_ID)
RESULTS_SAVE_DIR = './logs/{}/'.format(EXP_ID)

nt = 48
nextrap = 24
# starting at this time step,
# the prediction from the previous time step will be treated as the actual input
extrap_start_time = nt - nextrap
orig_weights_file = os.path.join(WEIGHTS_DIR, 'weights.hdf5')  # original t+1 weights
orig_json_file = os.path.join(WEIGHTS_DIR, 'model.json')

save_model = True
# where new weights will be saved
extrap_weights_file = os.path.join(
    WEIGHTS_DIR, 'weights-extrapfinetuned-{}-{}.hdf5'.format(nt, nextrap))
extrap_json_file = os.path.join(
    WEIGHTS_DIR, 'model-extrapfinetuned-{}-{}.json'.format(nt, nextrap))

# Data files
train_file = os.path.join(DATA_DIR, 'X_2016+2017_168x128.hkl')
train_sources = os.path.join(DATA_DIR, 'source_2016+2017_168x128.hkl')
val_file = os.path.join(DATA_DIR, 'X_test_2018_168x128.hkl')
val_sources = os.path.join(DATA_DIR, 'source_test_2018_168x128.hkl')

# Training parameters
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Load t+1 model
f = open(orig_json_file, 'r')
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
orig_model.load_weights(orig_weights_file)

layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']  # noqa
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

input_shape = list(orig_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt

inputs = Input(input_shape)
predictions = prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=extrap_loss, optimizer='adam')

train_generator = SequenceGenerator(
    train_file, train_sources, nt, batch_size=batch_size, shuffle=True,
    output_mode='prediction')
val_generator = SequenceGenerator(
    val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val,
    output_mode='prediction')


# start with lr of 0.001 and then drop to 0.0001 after 75 epochs
def lr_schedule(epoch): return 0.001 if epoch < 75 else 0.0001


callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)
    callbacks.append(
        ModelCheckpoint(
            filepath=extrap_weights_file, monitor='val_loss', save_best_only=True))
history = model.fit_generator(
    train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
    validation_data=val_generator, validation_steps=N_seq_val / batch_size)

pd.DataFrame(history.history).to_csv(os.path.join(RESULTS_SAVE_DIR, 'history_ext.csv'))

if save_model:
    json_string = model.to_json()
    with open(extrap_json_file, "w") as f:
        f.write(json_string)

'''
Train PredNet on KITTI sequences.
(Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''
from data_utils import SequenceGenerator
from prednet import PredNet
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import TimeDistributed
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras import backend as K
import datetime as dt
import os
import numpy as np
np.random.seed(123)


EXP_ID = dt.datetime.now().strftime('%Y%m%d%H%M%S')
DATA_DIR = '../inputs/hkl'
WEIGHTS_DIR = './logs/{}'.format(EXP_ID)
RESULTS_SAVE_DIR = './logs/{}'.format(EXP_ID)
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)
    os.chmod(WEIGHTS_DIR, 0o0777)
if not os.path.exists(RESULTS_SAVE_DIR):
    os.mkdir(RESULTS_SAVE_DIR)
    os.chmod(RESULTS_SAVE_DIR, 0o0777)


def main():
    save_model = True  # if weights will be saved
    # where weights will be saved
    weights_file = os.path.join(WEIGHTS_DIR, 'weights.hdf5')
    json_file = os.path.join(WEIGHTS_DIR, 'model.json')

    # Data files
    train_file = os.path.join(DATA_DIR, 'X_2016+2017_168x128x1.hkl')
    train_sources = os.path.join(DATA_DIR, 'source_2016+2017_168x128x1.hkl')
    val_file = os.path.join(DATA_DIR, 'X_2018_168x128x1.hkl')
    val_sources = os.path.join(DATA_DIR, 'source_2018_168x128x1.hkl')

    # Training parameters
    nb_epoch = 150
    batch_size = 16
    samples_per_epoch = 500
    N_seq_val = 100  # number of sequences to use for validation

    # Model parameters
    n_channels, im_height, im_width = (1, 168, 128)
    input_shape = (n_channels, im_height, im_width) if K.image_data_format(
    ) == 'channels_first' else (im_height, im_width, n_channels)
    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    # weighting for each layer in final loss;
    # "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.array([1., 0., 0., 0.])
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    nt = 24  # number of timesteps used for sequences in training
    # equally weight all timesteps except the first
    time_loss_weights = 1. / (nt - 1) * np.ones((nt, 1))
    time_loss_weights[0] = 0

    prednet = PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='error', return_sequences=True)

    inputs = Input(shape=(nt,) + input_shape)
    errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
    errors_by_time = TimeDistributed(
        Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)],
        trainable=False)(errors)  # calculate weighted error by layer
    errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
    final_errors = Dense(1, weights=[time_loss_weights, np.zeros(
        1)], trainable=False)(errors_by_time)  # weight errors by time
    model = Model(inputs=inputs, outputs=final_errors)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    train_generator = SequenceGenerator(
        train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
    val_generator = SequenceGenerator(
        val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    def lr_schedule(epoch): return 0.001 if epoch < 75 else 0.0001

    callbacks = [LearningRateScheduler(lr_schedule)]
    if save_model:
        if not os.path.exists(WEIGHTS_DIR):
            os.mkdir(WEIGHTS_DIR)
            os.chmod(WEIGHTS_DIR, 0o0777)
        callbacks.append(ModelCheckpoint(filepath=weights_file,
                                         monitor='val_loss', save_best_only=True))

    history = model.fit_generator(
        train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
        validation_data=val_generator, validation_steps=N_seq_val / batch_size)

    pd.DataFrame(history.history).to_csv(os.path.join(RESULTS_SAVE_DIR, 'history.csv'))

    if save_model:
        json_string = model.to_json()
        with open(json_file, "w") as f:
            f.write(json_string)


if __name__ == '__main__':
    main()

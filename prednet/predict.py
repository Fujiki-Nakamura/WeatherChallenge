'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''
import os

import hickle as hkl
import numpy as np
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image
from keras.layers import Input
from keras.models import Model, model_from_json

from prednet import PredNet
from submit_utils import eval_h, eval_w, crop_eval_area


n_plot = 50
batch_size = 10
nt = 48
extrap_start_time = 24
last_n_timestep = 24
input_h, input_w, input_c = 168, 128, 1
TARGET_TS = 24
sample_submit = '../inputs/sample_submit.csv'

EXP_ID = '20191022064146'
DATA_DIR = '../inputs/hkl/'
WEIGHTS_DIR = './logs/{}/'.format(EXP_ID)
RESULTS_SAVE_DIR = './logs/{}/'.format(EXP_ID)
test = True
valid = not test


def resize(X):
    H, W = 672, 512
    bs, ts, h, w, c = X.shape
    resized = np.zeros((bs * ts, H, W, c))
    ims = X.reshape(bs * ts, h, w, c)
    for i in range(len(ims)):
        im = ims[i].squeeze(2)
        _resized = Image.fromarray(im).resize((W, H))
        resized[i] = np.asarray(_resized)[:, :, np.newaxis]
    return resized.reshape(bs, ts, H, W, c)


def main():
    weights_file = os.path.join(WEIGHTS_DIR, 'weights-extrapfinetuned-48-24.hdf5')
    json_file = os.path.join(WEIGHTS_DIR, 'model-extrapfinetuned-48-24.json')

    # Load trained model
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
    train_model.load_weights(weights_file)

    # Create testing model (to output predictions)
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'
    layer_config['extrap_start_time'] = extrap_start_time
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']  # noqa
    test_prednet = PredNet(
        weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(inputs=inputs, outputs=predictions)

    X_test = hkl.load(INPUT_PATH)
    X_test_inp = X_test / 255.
    if last_n_timestep > 0:
        X_test_inp = X_test_inp[:, -last_n_timestep:]
        zeros = np.zeros((len(X_test), nt - last_n_timestep, input_h, input_w, input_c))
        X_test_inp = np.concatenate([X_test_inp, zeros], axis=1)
    if data_format == 'channels_first':
        X_test_inp = X_test_inp.transpose((0, 1, 4, 2, 3))
    X_hat = test_model.predict(X_test_inp, batch_size)
    if data_format == 'channels_first':
        X_test_inp = np.transpose(X_test_inp, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

    # MAE
    X_hat = np.round(X_hat * 255).clip(0, 255)
    X_hat = resize(X_hat)
    if test:
        # make submission
        indices = [i for i in range(TARGET_TS) if i % 6 == 5]
        preds = crop_eval_area(X_hat.squeeze(4))
        preds_eval = preds[:, indices, :, :].reshape(-1, eval_h, eval_w)
        df = pd.read_csv(sample_submit, header=None)
        df.loc[:, 1:] = preds_eval.reshape(-1, eval_w)
        df = df.astype(int)
        path = os.path.join(RESULTS_SAVE_DIR, 'submission.csv')
        df.to_csv(path, index=False, header=False)
        print('Submission saved at {}'.format(path))

    if valid:
        y_test = hkl.load(TARGET_PATH)
        MAE = np.mean(np.abs(y_test - X_hat[:, -last_n_timestep:]))
        print('MAE {:.4f}'.format(MAE))

        X_test[:, -last_n_timestep:] = y_test
        # Plot some predictions
        aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
        plt.figure(figsize=(nt, 2*aspect_ratio))
        gs = gridspec.GridSpec(2, nt)
        gs.update(wspace=0., hspace=0.)
        plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots_ext/')
        if not os.path.exists(plot_save_dir):
            os.mkdir(plot_save_dir)
            os.chmod(plot_save_dir, 0o0777)
        plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
        for i in plot_idx:
            for t in range(nt):
                plt.subplot(gs[t])
                X_true = X_test[i, t]
                if X_true.shape[2] == 1:
                    plt.imshow(X_test[i, t, :, :, 0], interpolation='none', cmap='gray')
                else:
                    plt.imshow(X_test[i, t], interpolation='none')
                plt.tick_params(
                    axis='both', which='both', bottom='off', top='off',
                    left='off', right='off', labelbottom='off', labelleft='off')
                if t == 0:
                    plt.ylabel('Actual', fontsize=10)

                plt.subplot(gs[t + nt])
                X_pred = X_hat[i, t]
                if X_pred.shape[2] == 1:
                    plt.imshow(X_pred[:, :, 0], interpolation='none', cmap='gray')
                else:
                    plt.imshow(X_pred, interpolation='none')
                plt.tick_params(
                    axis='both', which='both', bottom='off', top='off',
                    left='off', right='off', labelbottom='off', labelleft='off')
                if t == 0:
                    plt.ylabel('Predicted', fontsize=10)

            plt.savefig(plot_save_dir + 'plot_{:02d}.png'.format(i + 1))
            plt.clf()


if __name__ == '__main__':
    main()

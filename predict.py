import argparse
from argparse import Namespace  # noqa
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import WCDataset
import models


TARGET_TS = 24
eval_i, eval_j = (130, 40)
eval_h, eval_w = 420, 340


def crop_eval_area(data):
    return data[:, :, eval_j:eval_j+eval_h, eval_i:eval_i+eval_w]


def main(args):
    print(args)
    print('Predict with the data {}'.format(args.csv))

    test_set = WCDataset(args.data_root, is_training=False, test=True, args=args)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print('Loaded checkpoint {}'.format(args.checkpoint))
    if (checkpoint.get('best/MAE') and checkpoint.get('valid/MAE')
            and checkpoint.get('best/L1') and checkpoint.get('valid/L1')):
        print('Epoch {} best/MAE {} valid/MAE {} best/L1 {} valid/L1 {}'.format(
            checkpoint['epoch'],
            checkpoint['best/MAE'], checkpoint['valid/MAE'],
            checkpoint['best/L1'], checkpoint['valid/L1'],)
        )
    else:
        print('best/loss {} valid/loss {} epoch {}'.format(
            checkpoint['best/loss'], checkpoint['valid/loss'], checkpoint['epoch']))

    new_state_dict = OrderedDict()
    for k, v in iter(checkpoint['state_dict'].items()):
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v
    model = models.__dict__[args.model](args)
    model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()

    preds = np.zeros((len(test_set), TARGET_TS, eval_h, eval_w))
    trues = np.zeros((len(test_set), TARGET_TS, eval_h, eval_w))
    pbar = tqdm(total=len(test_loader))
    for batch_i, (data, impath) in enumerate(test_loader):
        input_ = data[0][:, args.last_n_ts:]
        target = data[1]
        bs, ts, c, h, w = input_.size()
        input_, target = input_.to(args.device), target.to(args.device)

        with torch.no_grad():
            output = model((input_ / 255.).float(), (target / 255.).float())

        output = (output * 255).round().type(torch.uint8).squeeze(2).cpu().numpy()
        target = target.type(torch.uint8).squeeze(2).cpu().numpy()
        output_eval = crop_eval_area(output)
        target_eval = crop_eval_area(target)
        preds[batch_i * bs:(batch_i + 1) * bs] = output_eval.astype(np.uint8)
        trues[batch_i * bs:(batch_i + 1) * bs] = target_eval.astype(np.uint8)

        pbar.update(1)
    pbar.close()

    '''
    mae = 0.
    if args.split.lower().startswith('valid'):
        indices = [i for i in range(24) if i % 6 == 5]
        p = preds[:, indices, :, :]
        t = trues[:, indices, :, :]
        mae = np.mean(np.abs(p - t))
        print('MAE {:.4f}'.format(mae))

    if args.dump:
        save_dir = os.path.join(args.log_dir, args.split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, 'preds_MAE-{:.4f}.npy'.format(mae))
        preds.dump(path)
        print('Dumped {}'.format(path))
        path = os.path.join(save_dir, 'trues_MAE-{:.4f}.npy'.format(mae))
        trues.dump(path)
        print('Dumped {}'.format(path))
    '''

    # make submission
    indices = [i for i in range(TARGET_TS) if i % 6 == 5]
    preds_eval = preds[:, indices, :, :].reshape(-1, eval_h, eval_w)
    df = pd.read_csv(args.sample_submit, header=None)
    df.loc[:, 1:] = preds_eval.reshape(-1, eval_w)
    df = df.astype(int)
    path = os.path.join(args.log_dir, 'submission.csv')
    df.to_csv(path, index=False, header=False)
    print('Submission saved at {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_root', type=str, default='../inputs/')
    parser.add_argument('--csv', type=str, default='inference_terms.csv')
    parser.add_argument('--height', type=int, default=672)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--ts', type=int, default=96)
    factor = 8
    parser.add_argument('--input_h', type=int, default=int(672 / factor))
    parser.add_argument('--input_w', type=int, default=int(512 / factor))
    parser.add_argument('--input_ts', type=int, default=96)
    parser.add_argument('--last_n_ts', type=int, default=24)
    parser.add_argument('--interpolation_mode', type=str, default='nearest')
    parser.add_argument('--n_workers', type=int, default=8)
    # network
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(5, 5))
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[16, ])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--weight_init', type=str, default='')
    # training
    parser.add_argument('--batch_size', type=int, default=1)
    # misc
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dump', action='store_true', default=False)
    parser.add_argument(
        '--sample_submit', type=str, default='../inputs/sample_submit.csv')

    args, _ = parser.parse_known_args()

    _logdir = '../logs/20191026020439/'
    args.checkpoint = os.path.join(_logdir, 'bestMAE.pt')
    args.model = 'encdec_02'
    args.hidden_dims = [8, 8, 8, ]
    args.n_layers = len(args.hidden_dims)
    args.loss = 'L1'
    args.logit_output = False
    args.teacher_forcing_ratio = -1
    args.residual = False

    '''
    logdir = os.path.dirname(args.checkpoint)
    logpath = os.path.join(logdir, 'main.log')
    with open(logpath, 'r') as f:
        first_line = f.readline()
        saved_args = eval(first_line.split(' - ')[-1])
    saved_args.__dict__.update(args.__dict__)
    '''
    main(args)

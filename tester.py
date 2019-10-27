from collections import OrderedDict
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import WCDataset
import utils


TARGET_TS = 24
eval_i, eval_j = (130, 40)
eval_h, eval_w = 420, 340


def crop_eval_area(data):
    return data[:, :, eval_j:eval_j+eval_h, eval_i:eval_i+eval_w]


def predict(args):
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
    model = utils.get_model(args)
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

        output = (output * 255).round().clamp(0, 255)
        output = output.type(torch.uint8).squeeze(2).cpu().numpy()
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
        save_dir = os.path.join(args.logdir, args.split)
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
    path = os.path.join(args.logdir, 'submission.csv')
    df.to_csv(path, index=False, header=False)
    print('Submission saved at {}'.format(path))

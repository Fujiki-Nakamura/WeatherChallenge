import datetime as dt
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import WCDataset
import utils


ID = dt.datetime.now().strftime('%Y%m%d%H%M%S')
TARGET_TS = 24
eval_i, eval_j = (130, 40)
eval_h, eval_w = 420, 340


def crop_eval_area(data):
    return data[:, :, eval_j:eval_j+eval_h, eval_i:eval_i+eval_w]


def predict(args):
    global TARGET_TS

    print(args)
    print('Predict with the data {}'.format(args.csv))

    test_set = WCDataset(
        args.data_root, args.csv, is_training=False, test=True, args=args)
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

    if not args.is_making_submission:
        TARGET_TS = args.target_ts
    preds = np.zeros((len(test_set), TARGET_TS, eval_h, eval_w))
    trues = np.zeros((len(test_set), TARGET_TS, eval_h, eval_w))
    pbar = tqdm(total=len(test_loader))
    for batch_i, (data, impath) in enumerate(test_loader):
        input_ = data[0][:, -args.last_n_ts:]
        bs, ts, h, w, c = input_.size()
        # (bs, ts, h, w, c) -> (bs, ts, c, h, w)
        input_ = input_.permute(0, 1, 4, 2, 3)
        input_ = input_.to(args.device)
        if not args.is_making_submission:
            target = data[1]
            target = target.permute(0, 1, 4, 2, 3)
            target = target.to(args.device)

        with torch.no_grad():
            target_tmp = None if args.is_making_submission else (target / 255.).float()
            output = model((input_ / 255.).float(), target_tmp)
            if TARGET_TS // args.output_ts == 2:
                if args.do_repeat_last_pred:
                    pass
                else:
                    bs, ts, c, h, w = output.size()
                    output_tmp = output.contiguous().view(bs * ts, c, h, w)
                    output_tmp = F.interpolate(
                        output_tmp, size=(args.input_h, args.input_w),
                        mode=args.interpolation_mode)
                    output_tmp = output_tmp.view(bs, ts, c, args.input_h, args.input_w)
                    output2 = model(output_tmp, target_tmp)
                    output = torch.cat([output, output2], dim=1)

        output = (output * 255).round().clamp(0, 255)
        output = output.type(torch.uint8).squeeze(2).cpu().numpy()
        if args.do_repeat_last_pred:
            copied = np.stack([output[:, -1], ] * 12, axis=1)
            output = np.concatenate([output, copied], axis=1)
        output_eval = crop_eval_area(output)
        preds[batch_i * bs:(batch_i + 1) * bs] = output_eval.astype(np.uint8)
        if not args.is_making_submission:
            target = target.type(torch.uint8).squeeze(2).cpu().numpy()
            target_eval = crop_eval_area(target)
            trues[batch_i * bs:(batch_i + 1) * bs] = target_eval.astype(np.uint8)

        pbar.update(1)
    pbar.close()

    if args.is_making_submission:
        # make submission
        indices = [i for i in range(TARGET_TS) if i % 6 == 5]
        preds_eval = preds[:, indices, :, :].reshape(-1, eval_h, eval_w)
        df = pd.read_csv(args.sample_submit, header=None)
        df.loc[:, 1:] = preds_eval.reshape(-1, eval_w)
        df = df.astype(int)
        path = os.path.join(args.logdir, f'submission_{ID}.csv')
        df.to_csv(path, index=False, header=False)
        print('Saved at {}'.format(path))
    else:
        if args.csv == 'training.csv':
            split = '2016'
        elif args.csv == 'validation.csv':
            split = '2017'
        elif args.csv == 'inference_terms.csv':
            split = '2018'
        else:
            raise Exception('Invalid CSV {}'.format(args.csv))
        if TARGET_TS == 24:
            indices = [i for i in range(TARGET_TS) if i % 6 == 5]
            p = preds[:, indices, :, :]
            t = trues[:, indices, :, :]
            mae = np.mean(np.abs(p - t))
            print('MAE {:.4f}'.format(mae))
        else:
            mae = np.mean(np.abs(preds - trues))
            print('L1 {:.4f}'.format(mae))
        if args.dump:
            path = os.path.join(args.logdir, '{}_preds_MAE-{:.4f}.npy'.format(split, mae))
            preds.dump(path)
            print('Dumped {}'.format(path))
            path = os.path.join(args.logdir, '{}_trues_MAE-{:.4f}.npy'.format(split, mae))
            trues.dump(path)
            print('Dumped {}'.format(path))

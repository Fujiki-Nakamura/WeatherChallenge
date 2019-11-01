import datetime as dt
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(args):
    transform_input = transforms.Compose([
        transforms.Resize((args.input_h, args.input_w))
    ])
    return transform_input


class WCDataset(Dataset):
    def __init__(
        self, data_root, csv='inference_terms.csv',
        ts=48, input_ts=24, target_ts=24,
        is_training=True, test=False, is_training_with_2018=False,
        args=None
    ):
        random.seed(args.random_seed)
        self.args = args
        self.data_root = data_root
        self.csv = csv
        self.is_training = is_training
        self.test = test
        self.h, self.w, self.c = args.height, args.width, args.channels
        self.input_h, self.input_w = args.input_h, args.input_w
        self.ts = ts
        self.input_ts = input_ts
        self.target_ts = target_ts
        assert self.input_ts + self.target_ts == self.ts

        self.transform = get_transforms(args)
        if is_training:
            dt_format = '%Y-%m-%d %H:%M'
            start_dt = dt.datetime.strptime('2016-01-01 00:00', dt_format)
            end_dt = dt.datetime.strptime('2017-12-31 23:00', dt_format)

            self.start_dt_list = []
            current_dt = start_dt
            while current_dt <= end_dt - dt.timedelta(hours=self.ts - 1):
                self.start_dt_list.append(current_dt)
                current_dt += dt.timedelta(hours=1)

            if is_training_with_2018:
                df = pd.read_csv(os.path.join(self.data_root, 'inference_terms.csv'))
                start_list = list(df.loc[:, 'OpenData_96hr_Start'].values)
                for i in range(len(df)):
                    start_dt = dt.datetime.strptime(start_list[i], '%Y/%m/%d %H:%M')
                    current_dt = start_dt
                    while current_dt <= start_dt + dt.timedelta(hours=24):
                        self.start_dt_list.append(current_dt)
                        current_dt += dt.timedelta(hours=1)
                print('Added 2018 data to training. The last is from {} to {}'.format(
                    start_dt.strftime(dt_format), current_dt.strftime(dt_format)))
        else:
            self.start_dt_list = []
            df = pd.read_csv(os.path.join(self.data_root, 'inference_terms.csv'))
            start_list = list(df.loc[:, 'OpenData_96hr_Start'].values)
            end_list = list(df.loc[:, 'OpenData_96hr_End'].values)
            for i in range(len(df)):
                dt_list = []
                start_dt = dt.datetime.strptime(start_list[i], '%Y/%m/%d %H:%M')
                end_dt = dt.datetime.strptime(end_list[i], '%Y/%m/%d %H:%M')
                if is_training_with_2018:
                    current_dt = start_dt + dt.timedelta(hours=48)
                else:
                    current_dt = start_dt
                while current_dt <= end_dt - dt.timedelta(hours=self.ts - 1):
                    dt_list.append(current_dt)
                    current_dt += dt.timedelta(hours=1)
                self.start_dt_list.extend(dt_list)
            if is_training_with_2018:
                assert len(self.start_dt_list) == len(df)
                print('Added 2018 data to training. Validate with {}, {}, etc'.format(
                    self.start_dt_list[0], self.start_dt_list[1]))

        if self.test:
            self.start_dt_list = []
            df = pd.read_csv(os.path.join(self.data_root, self.csv))
            start_list = list(df.loc[:, 'OpenData_96hr_Start'].values)
            for i in range(len(df)):
                start_dt = dt.datetime.strptime(start_list[i], '%Y/%m/%d %H:%M')
                self.start_dt_list.append(start_dt)

    def __len__(self):
        return len(self.start_dt_list)

    def __getitem__(self, idx):
        input_ = np.zeros((self.input_ts, self.input_h, self.input_w, self.c))
        target = np.zeros((self.target_ts, self.h, self.w, self.c))
        impaths = []

        start_dt = self.start_dt_list[idx]
        for ti in range(self.ts):
            current_dt = start_dt + dt.timedelta(hours=ti)
            dname = current_dt.strftime('%Y-%m-%d')
            fname = current_dt.strftime('%Y-%m-%d-%H-%M')
            if current_dt.year in [2016, 2017]:
                split = 'train'
            elif current_dt.year == 2018:
                split = 'test'
            else:
                raise NotImplementedError('Invalid year: {}'.format(current_dt.year))
            impath = '{split}/sat/{dname}/{fname}.fv.png'
            impath = impath.format(split=split, dname=dname, fname=fname)
            impath = os.path.join(self.data_root, impath)
            if os.path.isfile(impath):
                im = Image.open(impath).convert('L')
            else:
                path = os.path.join(
                    self.args.logdir, 'notFoundFor{}{}.list'.format(
                        'Test' if self.test else 'Train', current_dt.year))
                with open(path, 'a') as f:
                    print('Not found {}'.format(impath), file=f)
                continue

            if ti < self.input_ts:
                if self.transform is not None:
                    im = self.transform(im)
                input_[ti] = np.asarray(im)[:, :, np.newaxis]
            else:
                target[ti - self.input_ts] = np.asarray(im)[:, :, np.newaxis]
            impaths.append(impath)

        input_ = torch.from_numpy(input_.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        assert len(target) == self.target_ts

        return (input_, target), impaths

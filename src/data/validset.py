import datetime as dt
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


hr_range = 96
hr_range_target = 24


def get_transforms(args):
    transform_input = transforms.Compose([
        transforms.Lambda(lambda im: F.crop(im, *args.crop_params)),
        transforms.Resize((args.input_h, args.input_w))
    ])
    transform_target = transforms.Compose([
        transforms.Lambda(lambda im: F.crop(im, *args.crop_params)),
    ])
    return transform_input, transform_target


class WCValidset(Dataset):
    def __init__(
        self, data_root, csv, test=False, input_reversing_ratio=0.,
        args=None
    ):
        random.seed(args.random_seed)
        self.data_root = data_root
        self.df = pd.read_csv(csv)
        self.h, self.w = args.height, args.width
        self.input_h, self.input_w = args.resize_to
        self.split = 'test' if test else 'train'
        self.args = args
        self.input_reversing_ratio = input_reversing_ratio

        self.input_transform, self.target_transform = get_transforms(args)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ = np.zeros((hr_range, self.input_h, self.input_w))
        target = np.zeros((hr_range_target, self.h, self.w))

        # input
        start_96hr = self.df.loc[idx, 'OpenData_96hr_Start']
        end_96hr = self.df.loc[idx, 'OpenData_96hr_End']
        input_, impaths = self.get_images(
            start_96hr, end_96hr, hr_range, input_, is_input=True,
            transform=self.input_transform)

        # target
        start_hr = self.df.loc[idx, 'Inference_24hr_Start']
        end_hr = self.df.loc[idx, 'Inference_24hr_End']
        target, impaths_target = self.get_images(
            start_hr, end_hr, hr_range_target, target,
            transform=self.target_transform)

        input_ = input_ / 255.
        input_ = torch.from_numpy(input_.astype(np.float32))
        target_uint8 = torch.from_numpy(target.astype(np.float32))
        target = torch.from_numpy((target / 255.).astype(np.float32))

        return input_, (target, target_uint8), impaths, impaths_target

    def get_images(
        self, start_hr, end_hr, hr_range, images, is_input=False, transform=None
    ):
        impaths = []

        start_hr_dt = dt.datetime.strptime(start_hr, '%Y/%m/%d %H:%M')
        end_hr_dt = dt.datetime.strptime(end_hr, '%Y/%m/%d %H:%M')
        for i in range(hr_range):
            tmp_dt = start_hr_dt + dt.timedelta(hours=i)
            dname = tmp_dt.strftime('%Y-%m-%d')
            fname = tmp_dt.strftime('%Y-%m-%d-%H-%M')
            impath = '{}/sat/{}/{}.fv.png'.format(self.split, dname, fname)
            impath = os.path.join(self.data_root, impath)
            if os.path.isfile(impath):
                im = Image.open(impath).convert('L')

                if transform is not None:
                    im = transform(im)

                images[i] = np.array(im)
            else:
                path = os.path.join(self.args.log_dir, 'notFound.list')
                with open(path, 'a') as f:
                    print('Not found {}'.format(impath), file=f)
            impaths.append(impath)

        if not tmp_dt == dt.datetime.strptime('2016-03-02 15:00:00', '%Y-%m-%d %H:%M:%S'):
            assert tmp_dt == end_hr_dt

        if is_input and (random.random() < self.input_reversing_ratio):
            images = images[::-1, :, :]
            impaths = list(reversed(impaths))

        return images, impaths

import datetime as dt
import os
import random
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


input_range = 96
target_range = 24
earliest_dt = dt.datetime.strptime('2016-01-01-01-00', '%Y-%m-%d-%H-%M')
latest_dt = dt.datetime.strptime('2016/12/28 16:00', '%Y/%m/%d %H:%M')


def get_transforms(args):
    transform_input = transforms.Compose([
        transforms.Lambda(lambda im: F.crop(im, *args.crop_params)),
        transforms.Resize((args.input_h, args.input_w))
    ])
    transform_target = transforms.Compose([
        transforms.Lambda(lambda im: F.crop(im, *args.crop_params)),
    ])
    return transform_input, transform_target


class WCTrainset(Dataset):
    def __init__(self, root, csvpath, args, transform=None):
        self.root = root
        self.df = pd.read_csv(csvpath)
        self.h, self.w = args.height, args.width
        self.resize_to = args.resize_to
        self.args = args
        self.input_transform, self.target_transform = get_transforms(args)
        self.random_crop_delta = args.random_crop_delta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        start_hr_str = self.df.loc[idx, 'hour']
        dx = random.randint(-self.random_crop_delta, self.random_crop_delta)
        dy = random.randint(-self.random_crop_delta, self.random_crop_delta)
        # input
        start_hr_dt = dt.datetime.strptime(start_hr_str, '%Y-%m-%d-%H-%M')
        input_, impath_list = self.get_consecutive_images(
            start_hr_dt, input_range, is_input=True, transform=self.input_transform,
            dx=dx, dy=dy
        )

        # target
        start_hr_dt += dt.timedelta(hours=input_range)
        target, impath_target_list = self.get_consecutive_images(
            start_hr_dt, target_range, transform=self.target_transform,
            dx=dx, dy=dy
        )

        input_ = input_ / 255.
        input_ = torch.from_numpy(input_.astype(np.float32))
        target_uint8 = torch.from_numpy(target.astype(np.float32))
        target = torch.from_numpy((target / 255.).astype(np.float32))

        return input_, (target, target_uint8), impath_list, impath_target_list

    def get_consecutive_images(
        self, start_hr_dt, hr_range, is_input=False, transform=None, dx=0, dy=0,
    ):
        h = self.resize_to[0] if is_input else self.h
        w = self.resize_to[1] if is_input else self.w
        images = np.zeros((hr_range, h, w))
        impath_list = []
        for i in range(hr_range):
            current_hr_dt = start_hr_dt + dt.timedelta(hours=i)
            if not(earliest_dt <= current_hr_dt and current_hr_dt <= latest_dt):
                path = os.path.join(self.args.log_dir, 'notInRange.list')
                with open(path, 'a') as f:
                    print(current_hr_dt, file=f)

            # get impath
            current_hr_str = current_hr_dt.strftime('%Y-%m-%d-%H-%M')
            date = current_hr_str[:10]
            impath = os.path.join(
                self.root, 'train/sat/{}/{}.fv.png'.format(date, current_hr_str))
            impath_list.append(impath)

            # get image
            if os.path.isfile(impath):
                im = Image.open(impath).convert('L')
                im = im.crop((0+dx, 0+dy, 512+dx, 672+dy))

                if transform is not None:
                    im = transform(im)

                images[i] = np.array(im)
            else:
                path = os.path.join(self.args.log_dir, 'notFound.list')
                with open(path, 'a') as f:
                    print('Not found {}'.format(impath), file=f)

        return images, impath_list

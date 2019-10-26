from glob import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(args):
    transform = transforms.Compose([
        transforms.Resize((args.input_h, args.input_w)),
        transforms.ToTensor(),
    ])
    return transform


class WCDataset(Dataset):
    def __init__(self, root, args, test=False):
        self.args = args
        self.transform = get_transforms(args)
        self.root = root
        self.h, self.w, self.c = args.h, args.w, args.c
        self.test = test
        self.impath = 'test/sat/*/*.fv.png' if self.test else 'train/sat/*/*.fv.png'
        self.impaths = sorted(glob(os.path.join(self.root, self.impath)))

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        # input
        impath = self.impaths[idx]
        im = Image.open(impath).convert('L')

        if self.transform is not None:
            input_ = self.transform(im)

        target = transforms.ToTensor()(im)

        return input_, target, impath

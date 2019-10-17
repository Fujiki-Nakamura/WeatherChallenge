from torchvision import transforms
import torchvision.transforms.functional as F


def get_transform(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda im: F.crop(im, args.crop_params)),
        transforms.Resize((args.input_h, args.input_w))
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda im: F.crop(im, args.crop_params)),
    ])

    return transform, val_transform

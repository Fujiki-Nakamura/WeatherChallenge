from torch.utils.data import DataLoader

from .dataset import WCDataset


def get_dataloader(args):
    train_set = WCDataset(
        args.data_root, args=args,
        is_training=True, is_training_with_2018=args.is_training_with_2018,)
    valid_set = WCDataset(
        args.data_root, args=args,
        is_training=False, is_training_with_2018=args.is_training_with_2018,)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.n_workers,
        pin_memory=True, shuffle=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, num_workers=args.n_workers,
        pin_memory=True, shuffle=False)
    return train_loader, valid_loader

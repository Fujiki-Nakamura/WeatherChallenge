import os
import shutil
import torch
from torch import nn, optim
import adabound
from RAdam import RAdam


def get_loss_fn(args):
    if args.loss.lower().startswith('bce'):
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    elif args.loss.lower().startswith('mse'):
        loss_fn = nn.MSELoss(reduction='mean')
    elif args.loss.lower().startswith('l1'):
        loss_fn = nn.L1Loss(reduction='mean')
    elif args.loss.lower() == 'SmoothL1'.lower():
        loss_fn = nn.SmoothL1Loss(reduction='mean')
    else:
        raise NotImplementedError
    valid_loss_fn = nn.L1Loss(reduction='mean')
    return loss_fn, valid_loss_fn


def get_optimizer(model, optim_str):
    torch_optim_list = ['SGD', 'Adam']
    possible_optim_list = torch_optim_list + ['RAdam', 'AdaBound']

    optim_args = optim_str.split('/')
    name = optim_args[0]
    assert name in possible_optim_list, '{} not implemented.'.format(name)

    args_dict = {e.split('=')[0]: eval(e.split('=')[1]) for e in optim_args[1:]}

    model_params = model.parameters()
    if name in torch_optim_list:
        optimizer = optim.__dict__[name](model_params, **args_dict)
    elif name == 'AdaBound':
        optimizer = adabound.AdaBound(model_params, **args_dict)
    elif name == 'RAdam':
        optimizer = RAdam(model_params, **args_dict)

    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler.lower() == 'multisteplr':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, args.milestones, args.gamma)
    else:
        return None
    return scheduler


def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler
    from logging import Formatter, DEBUG, INFO
    fh = FileHandler(log_file)
    fh.setLevel(INFO)
    sh = StreamHandler()
    sh.setLevel(DEBUG)
    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('log')
    logger.setLevel(DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_choices(choice_list):
    return choice_list + list(map(lambda s: s.lower(), choice_list))


def save_checkpoint(state, is_best, log_dir):
    filename = os.path.join(log_dir, 'checkpoint.pt')
    torch.save(state, filename)
    if is_best[0]:
        shutil.copyfile(filename, os.path.join(log_dir, 'bestMAE.pt'))
    if is_best[1]:
        shutil.copyfile(filename, os.path.join(log_dir, 'bestL1.pt'))


class AverageMeter(object):
    """Computes and stores the average and current value
        adopted from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L296
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

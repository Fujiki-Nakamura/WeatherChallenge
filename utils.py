import datetime as dt
import os
import shutil
import torch
from torch import optim
import adabound
import models
from RAdam import RAdam


def get_optimizer(model, optim_str):
    torch_optim_list = ['SGD', 'Adam']
    possible_optim_list = torch_optim_list + ['RAdam', 'AdaBound']

    optim_args = optim_str.strip('/').split('/')
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


def get_model(args):
    model = models.__dict__[args.model](args.latent_size, args=args)
    return model


def get_logdir(args):
    if args.expid == '':
        args.expid = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    logdir = os.path.join(args.logdir, args.expid)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    os.chmod(logdir, 0o0777)
    return logdir


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


def save_checkpoint(state, is_best, logdir):
    filename = os.path.join(logdir, 'checkpoint.pt')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(logdir, 'best.pt'))


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

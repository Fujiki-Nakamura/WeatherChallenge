import datetime as dt
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from data import WCDataset, WCTrainset, WCValidset
import models
from trainer import train, validate
from utils import get_logger, get_optimizer, save_checkpoint
from utils import get_scheduler, get_loss_fn


def run(args):
    start_epoch = 1
    best = {'L1': 1e+9, 'MAE': 1e+9}

    # logs
    if args.expid == '':
        args.expid = dt.datetime.now().strftime('%Y%m%d%H%M')
    args.log_dir = os.path.join(args.log_dir, args.expid)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    os.chmod(args.log_dir, 0o0777)
    logger = get_logger(os.path.join(args.log_dir, 'main.log'))
    logger.info(args)
    writer = SummaryWriter(args.log_dir)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    if args.trainset == 'trainset':
        train_set = WCTrainset(args.data_root, args.train_csv, args=args)
    else:
        train_set = WCDataset(args.data_root, args.train_csv, args=args)
    valid_set = WCValidset(args.data_root, args.valid_csv, args=args)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False)

    # network
    model = models.__dict__[args.model](args=args)
    if torch.cuda.device_count() > 1:
        logger.info('{} GPUs found.'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(args.device)
    # training
    criterion, valid_loss_fn = get_loss_fn(args)
    optimizer = get_optimizer(model, args.optim_str)
    scheduler = get_scheduler(optimizer, args)
    logger.debug(optimizer)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best['L1'] = checkpoint['best/L1']
            best['MAE'] = checkpoint['best/MAE']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Loaded checkpoint {} (epoch {})'.format(
                args.resume, start_epoch - 1))
        else:
            raise IOError('No such file {}'.format(args.resume))

    for epoch_i in range(start_epoch, args.epochs + 1):
        message = '[{}] Epoch {} Train/{} {:.2f} /MAE {:.4f} Valid/L1 {:.2f} /MAE {:.4f} (Best {:.4f}) '  # noqa
        for param_group in optimizer.param_groups:
            message += 'LR {:.4f} '.format(param_group['lr'])

        training = train(
            train_loader, model, criterion, optimizer, logger=logger, args=args)
        validation = validate(
            valid_loader, model, valid_loss_fn, logger=logger, args=args)

        writer.add_scalar('{}/Train'.format(args.loss), training['loss'], epoch_i)
        writer.add_scalar('{}/Valid'.format(args.loss), validation['loss'], epoch_i)
        writer.add_scalar('MAE/Train', training['mae'], epoch_i)
        writer.add_scalar('MAE/Valid', validation['mae'], epoch_i)
        writer.add_scalar(
            'Grad/L2/Mean/BeforeClipped/Train', training['grad/L2/BeforeClipped'], epoch_i)
        writer.add_scalar(
            'Grad/L2/Mean/Clipped/Train', training['grad/L2/Clipped'], epoch_i)
        writer.add_scalar('Grad/L2/Mean/Train', training['grad/L2/Mean'], epoch_i)
        if epoch_i % args.freq_to_log_image == 0:
            writer.add_image(
                'Train/Predict', _get_images(training['pred'], args), epoch_i)
            writer.add_image(
                'Train/Target', _get_images(training['true'], args), epoch_i)
            writer.add_image(
                'Valid/Predict', _get_images(validation['pred'], args), epoch_i)
            writer.add_image(
                'Valid/Target', _get_images(validation['true'], args), epoch_i)

        is_best = (validation['mae'] < best['MAE'], validation['loss'] < best['L1'])
        if is_best[0]:
            best['MAE'] = validation['mae']
        if is_best[1]:
            best['L1'] = validation['loss']
        save_checkpoint({
            'epoch': epoch_i,
            'state_dict': model.state_dict(),
            'valid/L1': validation['loss'],
            'valid/MAE': validation['mae'],
            'best/L1': best['L1'],
            'best/MAE': best['MAE'],
            'optimizer': optimizer.state_dict(),
        }, is_best, args.log_dir)

        if scheduler is not None:
            scheduler.step(epoch=epoch_i)

        message = message.format(
            args.expid, epoch_i,
            args.loss, training['loss'], training['mae'],
            validation['loss'], validation['mae'], best['MAE'])
        logger.info(message)


def _get_images(output, args):
    nrow = 6
    # (bs, ts, h, w) -> (bs, ts, c, h, w) -> (bs * ts, c, h, w)
    _ims = output.unsqueeze(2).contiguous().view(-1, args.channels, 420, 340)
    return vutils.make_grid(
        _ims, nrow=nrow, scale_each=True, normalize=True, range=(0, 255))

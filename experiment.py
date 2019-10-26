import os

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset import WCDataset
from trainer import train, validate
from utils import get_logdir, get_logger, save_checkpoint
from utils import get_model, get_optimizer


def run(args):
    start_epoch = 1
    best = {'L1': 1e+9}

    # logs
    logdir = get_logdir(args)
    logger = get_logger(os.path.join(logdir, 'main.log'))
    logger.info(args)
    writer = SummaryWriter(logdir)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    train_set = WCDataset(args.data_root, test=False, args=args)
    valid_set = WCDataset(args.data_root, test=True, args=args)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True,
        pin_memory=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
        pin_memory=True)

    # network
    model = get_model(args).to(args.device)
    # training
    optimizer = get_optimizer(model, args.optim_str)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best['L1'] = checkpoint['best/L1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Loaded checkpoint {} (epoch {})'.format(
                args.resume, start_epoch - 1))
        else:
            raise IOError('No such file {}'.format(args.resume))

    for epoch_i in range(start_epoch, args.epoch + 1):
        message = '[{}] Epoch {}/{} '
        message += 'Train/{} {:.4f} /L1 {:.4f} '
        message += 'Valid/{} {:.4f} /L1 {:.4f} (Best {:.4f}) '
        for param_group in optimizer.param_groups:
            message += 'LR {:.4f} '.format(param_group['lr'])

        training = train(train_loader, model, optimizer, logger=logger, args=args)  # noqa
        validation = validate(valid_loader, model, logger=logger, args=args)

        writer.add_scalar('{}/Train'.format(args.loss), training['loss'], epoch_i)
        writer.add_scalar('{}/Valid'.format(args.loss), validation['loss'], epoch_i)
        writer.add_scalar('L1/Train', training['L1'], epoch_i)
        writer.add_scalar('L1/Valid', validation['L1'], epoch_i)
        if epoch_i % args.log_image_freq == 0:
            writer.add_image(
                'Train/Predict', _get_images(training['pred'], args), epoch_i)
            writer.add_image(
                'Train/Target', _get_images(training['true'], args), epoch_i)
            writer.add_image(
                'Valid/Predict', _get_images(validation['pred'], args), epoch_i)
            writer.add_image(
                'Valid/Target', _get_images(validation['true'], args), epoch_i)

        is_best = validation['L1'] < best['L1']
        if is_best:
            best['L1'] = validation['L1']
        save_checkpoint({
            'epoch': epoch_i,
            'state_dict': model.state_dict(),
            'valid/L1': validation['L1'],
            'best/L1': best['L1'],
            'optimizer': optimizer.state_dict(),
        }, is_best, logdir)

        message = message.format(
            args.expid, epoch_i, args.epoch,
            args.loss, training['loss'], training['L1'],
            args.loss, validation['loss'], validation['L1'], best['L1'])
        logger.info(message)


def _get_images(output, args):
    return vutils.make_grid(output, nrow=4)

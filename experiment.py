import datetime as dt
import os

from tensorboardX import SummaryWriter

import data
from trainer import train
import utils


def run(args):
    start_epoch = 1
    best = {'L1': 1e+9, 'MAE': 1e+9}

    # logdir
    expid = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    args.logdir = os.path.join(args.logdir, expid)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
        os.chmod(args.logdir, 0o0777)
    # logger
    logger = utils.get_logger(os.path.join(args.logdir, 'main.log'))
    logger.info(args)
    with open(os.path.join(args.logdir, 'args.log'), 'a') as f:
        for k, v in sorted(args.__dict__.items()):
            f.write('{} {}\n'.format(k, v))

    writer = SummaryWriter(args.logdir)

    # data
    train_loader, valid_loader = data.get_dataloader(args)

    # network
    model = utils.get_model(args).to(args.device)
    # training
    criterion = utils.get_loss_fn(args)
    optimizer = utils.get_optimizer(model, args.optim_str)
    if args.resume:
        model, optimizer = utils.resume(model, optimizer)
        logger.info('Loaded {} (epoch {})'.format(args.resume, start_epoch - 1))

    for epoch_i in range(start_epoch, args.epochs + 1):
        message = '[{expid}] Epoch {epoch_i} '.format(expid=expid, epoch_i=epoch_i)
        message += '[{loss}] Train {train_loss:.2f} Valid {val_loss:.2f} '
        message += '[MAE] Train {train_MAE:.4f} Valid {val_MAE:.4f} (Best {best:.4f}) '

        training = train(
            train_loader, model, criterion, optimizer, is_training=True,
            logger=logger, args=args)
        validation = train(
            valid_loader, model, criterion, optimizer, is_training=False,
            logger=logger, args=args)

        writer.add_scalar('{}Loss/Train'.format(args.loss), training['loss'], epoch_i)
        writer.add_scalar('{}Loss/Valid'.format(args.loss), validation['loss'], epoch_i)
        writer.add_scalar('L1/Train', training['L1'], epoch_i)
        writer.add_scalar('L1/Valid', validation['L1'], epoch_i)
        writer.add_scalar('MAE/Train', training['mae'], epoch_i)
        writer.add_scalar('MAE/Valid', validation['mae'], epoch_i)
        if epoch_i % args.freq_to_log_image == 0:
            writer.add_image(
                'Train/Predict', utils.get_images(training['pred'], args), epoch_i)
            writer.add_image(
                'Train/Target', utils.get_images(training['true'], args), epoch_i)
            writer.add_image(
                'Valid/Predict', utils.get_images(validation['pred'], args), epoch_i)
            writer.add_image(
                'Valid/Target', utils.get_images(validation['true'], args), epoch_i)

        is_best = (validation['mae'] < best['MAE'], validation['loss'] < best['L1'])
        if is_best[0]:
            best['MAE'] = validation['mae']
        if is_best[1]:
            best['L1'] = validation['loss']
        utils.save_checkpoint({
            'epoch': epoch_i,
            'state_dict': model.state_dict(),
            'valid/L1': validation['loss'],
            'valid/MAE': validation['mae'],
            'best/L1': best['L1'],
            'best/MAE': best['MAE'],
            'optimizer': optimizer.state_dict(),
        }, is_best, args.logdir)

        message = message.format(
            loss=args.loss, train_loss=training['L1'], val_loss=validation['L1'],
            train_MAE=training['mae'], val_MAE=validation['mae'], best=best['MAE'])
        logger.info(message)

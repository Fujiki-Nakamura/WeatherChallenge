import numpy as np
import torch
from tqdm import tqdm
from utils import AverageMeter


hr_range_target = 24
mae_fn = torch.nn.L1Loss(reduction='mean')
indices = [i for i in range(hr_range_target) if i % 6 == 5]


def train(dataloader, model, criterion, optimizer, logger=None, args=None):
    model.train()
    losses = AverageMeter()
    losses_255 = AverageMeter()
    maes = AverageMeter()
    grads = grads_clipped = 0.
    grad_norm = 0.
    pbar = tqdm(total=len(dataloader))
    for i, (inputs, targets, _, _) in enumerate(dataloader):
        bs, ts, h, w = targets[0].size()
        outputs, loss, loss_255, mae, output_255_eval, target_255_eval = step(
            inputs, targets, model, criterion, args=args)
        n = bs * ts * h * w
        losses.update(loss.item(), n)
        losses_255.update(loss_255.item(), n)
        maes.update(mae.item(), bs * ts / 6 * h * w)

        optimizer.zero_grad()
        loss.backward()
        grads += np.sqrt(sum(
            [p.grad.data.pow(2).sum().item() for p in model.parameters()]))
        if args.max_norm > 0.:
            grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        grads_clipped += np.sqrt(sum(
            [p.grad.data.pow(2).sum().item() for p in model.parameters()]))
        optimizer.step()

        if args.debug: break  # noqa

        pbar.update(1)
    pbar.close()

    logger.debug('Train/Loss {:.4f}'.format(losses.avg))

    return {
        'loss': losses_255.avg, 'mae': maes.avg,
        'pred': output_255_eval, 'true': target_255_eval,
        'grad/L2/BeforeClipped': grads / len(dataloader),
        'grad/L2/Clipped': grads_clipped / len(dataloader),
        'grad/L2/Mean': grad_norm / len(dataloader),
    }


def validate(dataloader, model, criterion, logger=None, args=None):
    model.eval()
    losses = AverageMeter()
    losses_255 = AverageMeter()
    maes = AverageMeter()
    pbar = tqdm(total=(len(dataloader)))
    for i, (inputs, targets, _, _) in enumerate(dataloader):
        bs, ts, h, w = targets[0].size()
        with torch.no_grad():
            outputs, loss, loss_255, mae, output_255_eval, target_255_eval = step(
                inputs, targets, model, criterion, args=args, mode='valid')
        n = bs * ts * h * w
        losses.update(loss.item(), n)
        losses_255.update(loss_255.item(), n)
        maes.update(mae.item(), bs * ts / 6 * h * w)

        if args.debug: break  # noqa

        pbar.update(1)
    pbar.close()

    logger.debug('Valid/Loss {:.4f}'.format(losses.avg))

    return {
        'loss': losses_255.avg, 'mae': maes.avg,
        'pred': output_255_eval, 'true': target_255_eval,
    }


def step(inputs, targets, model, criterion, args, mode='train'):
    targets, target_255 = targets

    # (bs, ts, h, w) -> (bs, ts, c, h, w)
    inputs = inputs.unsqueeze(2)
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    target_255 = target_255.to(args.device)

    outputs = model(inputs, targets.unsqueeze(2))
    output_255 = (outputs * 255).round()

    # (bs, ts, c, h, w) -> (bs, ts, h, w) -> (ts, bs, h, w)
    outputs = outputs.squeeze(2).permute(1, 0, 2, 3)
    output_255 = output_255.squeeze(2).permute(1, 0, 2, 3)
    # (bs, ts, h, w) -> (ts, bs, h, w)
    targets = targets.permute(1, 0, 2, 3)
    target_255 = target_255.permute(1, 0, 2, 3)

    loss = 0.
    loss_255 = 0
    loss += criterion(outputs, targets)
    loss_255 += criterion(output_255, target_255)

    # loss as to 6/12/18/24hr
    assert len(outputs) == len(targets) == hr_range_target
    output_255 = output_255[:, :, 40:460, 130:470]
    target_255 = target_255[:, :, 40:460, 130:470]
    output_255_eval = output_255[indices]
    target_255_eval = target_255[indices]
    mae = mae_fn(output_255_eval, target_255_eval)

    return (
        outputs, loss, loss_255, mae,
        # batch_first
        output_255.permute(1, 0, 2, 3), target_255.permute(1, 0, 2, 3),
    )

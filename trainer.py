import numpy as np
import torch
from tqdm import tqdm
from utils import AverageMeter


target_ts = 24
mae_fn = torch.nn.L1Loss(reduction='mean')
eval_indices = [i for i in range(target_ts) if i % 6 == 5]


def train(
    dataloader, model, criterion, optimizer, is_training=True, logger=None, args=None
):
    if is_training:
        model.train()
    else:
        model.eval()
    losses = AverageMeter()
    MAEs = AverageMeter()

    pbar = tqdm(total=len(dataloader))
    for data, _ in dataloader:
        with torch.set_grad_enabled(is_training):
            bs, _, h, w, c = data[0].size()
            output, loss, MAE = step(data, model, criterion, args=args)
            n = bs * target_ts * h * w * c
            losses.update(loss.item(), n)
            MAEs.update(MAE.item(), bs * int(target_ts / 6) * h * w)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if args.debug: break  # noqa
        pbar.update(1)
    pbar.close()

    return {
        'loss': losses.avg, 'mae': MAEs.avg,
    }


def step(data, model, criterion, args):
    input_ = data[0].to(args.device)
    target = data[1].to(args.device)
    # (bs, ts, h, w, c) -> (bs, ts, c, h, w)
    input_ = input_.permute(0, 1, 4, 2, 3)
    target = target.permute(0, 1, 4, 2, 3)
    logit, pred = model((input_ / 255.).float(), (target / 255.).float())

    _target = target.permute(0, 2, 1, 3, 4)
    label = torch.zeros(_target.size(), dtype=torch.long, device=_target.device)
    ls = np.linspace(0, 255, args.output_c)
    for i in range(len(ls) - 1):
        lower = int(ls[i])
        upper = int(ls[i + 1])
        label[(_target >= lower) & (_target < upper)] = i
    label[_target >= upper] = 9
    loss = criterion(logit.permute(0, 2, 1, 3, 4), label.squeeze(1))

    # loss as to 6/12/18/24hr
    assert pred.size()[1] == target.size()[1] == target_ts
    output_eval = (pred * 255.).round()[:, :, :, 40:460, 130:470]
    target_eval = target[:, :, :, 40:460, 130:470]
    output_eval = output_eval[:, eval_indices, :, :, :]
    target_eval = target_eval[:, eval_indices, :, :, :]
    MAE = mae_fn(output_eval, target_eval.float())

    return pred, loss, MAE

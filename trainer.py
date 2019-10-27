import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import AverageMeter


mae_fn = torch.nn.L1Loss(reduction='mean')


def train(
    dataloader, model, criterion, optimizer, is_training=True, logger=None, args=None
):
    if is_training:
        model.train()
    else:
        model.eval()
    losses = AverageMeter()
    L1s = AverageMeter()
    MAEs = AverageMeter()

    pbar = tqdm(total=len(dataloader))
    for data, _ in dataloader:
        with torch.set_grad_enabled(is_training):
            bs, _, h, w, c = data[0].size()
            output, loss, L1, MAE = step(data, model, criterion, args=args)
            n = bs * args.target_ts * h * w * c
            losses.update(loss.item(), n)
            L1s.update(L1.item(), n)
            MAEs.update(MAE.item(), bs * int(args.target_ts / 6) * h * w)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if args.debug: break  # noqa
        pbar.update(1)
    pbar.close()

    return {
        'loss': losses.avg, 'L1': L1s.avg, 'mae': MAEs.avg,
    }


def step(data, model, criterion, args):
    input_ = data[0].to(args.device)
    target = data[1].to(args.device)
    # (bs, ts, h, w, c) -> (bs, ts, c, h, w)
    input_ = input_.permute(0, 1, 4, 2, 3)
    target = target.permute(0, 1, 4, 2, 3)

    assert args.target_ts % args.output_ts == 0
    if args.target_ts // args.output_ts > 1:
        # TODO
        '''
        input_tmp = (input_ / 255.).float()
        target_tmp = (target / 255.).float()[:, :args.output_ts]
        output_list = []
        for _ in range(int(args.target_ts / args.output_ts)):
            output = model(input_tmp, target_tmp)
        '''
        input_tmp = (input_ / 255.).float()
        target_tmp = (target / 255.).float()[:, :args.output_ts]
        output = model(input_tmp, target_tmp)
        loss = criterion(output, target_tmp)

        bs, ts, c, h, w = output.size()
        output_tmp = output.contiguous().view(bs * ts, c, h, w)
        output_tmp = F.interpolate(
            output_tmp, size=(args.input_h, args.input_w), mode=args.interpolation_mode)
        output_tmp = output_tmp.view(bs, ts, c, args.input_h, args.input_w)
        input_tmp = torch.cat([input_tmp[:, args.output_ts:], output_tmp], dim=1)
        assert input_tmp.size()[1] == args.input_ts
        target_tmp = (target / 255.).float()[:, args.output_ts:]
        output2 = model(input_tmp, target_tmp)
        loss += criterion(output2, target_tmp)

        output = torch.cat([output, output2], dim=1)
    else:
        output = model((input_ / 255.).float(), (target / 255.).float())
        loss = criterion(output, (target / 255.).float())
        L1 = criterion((output * 255.).round(), target.float())

    assert output.size()[1] == target.size()[1] == args.target_ts
    L1 = mae_fn((output * 255).round().clamp(0, 255).float(), target.float())

    # loss as to 6/12/18/24hr
    eval_indices = [i for i in range(args.target_ts) if i % 6 == 5]
    output_eval = (output * 255.).round()[:, :, :, 40:460, 130:470]
    target_eval = target[:, :, :, 40:460, 130:470]
    output_eval = output_eval[:, eval_indices, :, :, :]
    target_eval = target_eval[:, eval_indices, :, :, :]
    MAE = mae_fn(output_eval, target_eval.float())

    return output, loss, L1, MAE

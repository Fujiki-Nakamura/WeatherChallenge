import torch
import torch.nn as nn


l1loss_fn = nn.L1Loss(reduction='mean')


def L1_GDL(output, target, alpha=1.):
    l1 = l1loss_fn(output, target)
    gdl = GDL(output, target, alpha=alpha)
    return l1 + gdl


def GDL(output, target, alpha=1.):
    assert len(output.size()) == len(target.size()) == 5
    bs, ts, c, h, w = output.size()
    term1 = torch.pow(torch.abs(
        torch.abs(output[:, :, :, 1:, :] - output[:, :, :, :-1, :]) -
        torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        ), alpha)
    term2 = torch.pow(torch.abs(
        torch.abs(output[:, :, :, :, 1:] - output[:, :, :, :, :-1]) -
        torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
        ), alpha)
    gdl = torch.mean(term1 + term2)

    return gdl

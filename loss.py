import torch
import torch.nn as nn


class L1andGDL(nn.Module):
    def __init__(
        self, coef_l1=1., coef_gdl=1., alpha=1.,
        reduction='mean',
        **kwargs
    ):
        super(L1andGDL, self).__init__()
        self.coef_l1 = coef_l1
        self.coef_gdl = coef_gdl
        self.l1 = nn.L1Loss(reduction=reduction)
        self.alpha = alpha
        self.gdl = GDL(alpha=alpha)

    def forward(self, output, target):
        l1 = self.l1loss(output, target)
        gdl = self.gdl(output, target)
        return self.coef_l1 * l1 + self.coef_gdl * gdl


class GDL(nn.Module):
    def __init__(self, alpha=1., reduction='mean'):
        super(GDL, self).__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, output, target):
        term1 = torch.pow(torch.abs(
            torch.abs(output[:, :, :, 1:, :] - output[:, :, :, :-1, :]) -
            torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
            ), self.alpha)
        term2 = torch.pow(torch.abs(
            torch.abs(output[:, :, :, :, 1:] - output[:, :, :, :, :-1]) -
            torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
            ), self.alpha)
        if self.reduction == 'mean':
            gdl = torch.mean(term1 + term2)
        elif self.reduction == 'sum':
            gdl = torch.sum(term1 + term2)
        else:
            raise NotImplementedError('Invalid reduction {}'.format(self.reduction))

        return gdl

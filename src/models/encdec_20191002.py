import torch
from torch import nn
from torch.nn import functional as F
from .convlstm2 import ConvLSTM
from .convlstm_20191002 import ConvLSTM as ConvLSTMDecoder


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.hidden_dims = args.hidden_dims
        self.kernel_size = args.kernel_size
        self.n_layers = args.n_layers

        self.convlstm1 = ConvLSTM(
            input_size=(args.input_h, args.input_w), input_dim=args.channels,
            hidden_dim=self.hidden_dims,  kernel_size=self.kernel_size,
            num_layers=self.n_layers,
            batch_first=True, bias=True, return_all_layers=True,
            n_additional_convs=args.n_additional_convs,
            weight_init=args.weight_init)

    def forward(self, x):
        out, hidden_list = self.convlstm1(x)
        return out[-1], hidden_list


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.is_teacher_forcing = (
            args.teacher_forcing_ratio > 0 or args.teacher_forcing_ratio == -1)
        self.input_dim = (
            args.channels if self.is_teacher_forcing else args.hidden_dims[-1])
        self.hidden_dims = args.hidden_dims
        self.kernel_size = args.kernel_size
        self.n_layers = args.n_layers
        self.device = args.device

        _input_size = (args.input_h, args.input_w)
        self.convlstm1 = ConvLSTMDecoder(
            input_size=_input_size, input_dim=self.input_dim,
            hidden_dim=self.hidden_dims,  kernel_size=self.kernel_size,
            num_layers=self.n_layers,
            batch_first=True, bias=True, return_all_layers=True,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            n_additional_convs=args.n_additional_convs,
            weight_init=args.weight_init)

    def forward(self, x, hidden_list=None, target=None):
        out, hidden_list = self.convlstm1(x, hidden_list, target)

        return out, hidden_list


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.loss = args.loss
        self.h, self.w = args.height, args.width
        self.input_h, self.input_w = args.input_h, args.input_w
        self.mode = args.interpolation_mode
        self.logit_output = args.logit_output
        self.is_teacher_forcing = (
            args.teacher_forcing_ratio > 0 or args.teacher_forcing_ratio == -1)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, input_, target):
        out_e, hidden_e = self.encoder(input_)

        input_d = input_[:, -1, :, :, :] if self.is_teacher_forcing else out_e
        if self.is_teacher_forcing:
            bs, ts, c, h, w = target.size()
            target = F.interpolate(
                target.view(bs * ts, c, h, w),
                size=(self.input_h, self.input_w), mode=self.mode
            ).view(bs, ts, c, self.input_h, self.input_w)
        out_d, hidden_d = self.decoder(input_d, hidden_e, target)

        if self.is_teacher_forcing:
            bs, ts, c, h, w = out_d.size()
            out = out_d.contiguous().view(bs * ts, c, h, w)
        else:
            # out_d: list of tensor(bs, ts, c=hidden_dim, h, w)
            out = torch.cat(out_d, dim=2)
            bs, ts, c, h, w = out.size()
            out = self.conv1x1(out.view(bs * ts, c, h, w))
        if not self.logit_output:
            out = torch.sigmoid(out)
        out = F.interpolate(out, size=(self.h, self.w), mode=self.mode)
        out = out.view(bs, ts, -1, self.h, self.w)

        return out


def encdec_20191002(args):
    return Model(args)

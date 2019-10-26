import numpy as np
from torch import nn
from torch.nn import functional as F
from .convlstm_02 import ConvLSTM
from .convlstm_clf import ConvLSTM as ConvLSTMClassifier


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
            weight_init=args.weight_init)

    def forward(self, x):
        out, hidden_list = self.convlstm1(x)
        return out[-1], hidden_list


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.input_dim = args.channels
        self.output_c = args.output_c
        self.hidden_dims = args.hidden_dims
        self.kernel_size = args.kernel_size
        self.n_layers = args.n_layers
        self.device = args.device

        self.class2value_list = []
        ls = np.linspace(0, 255, self.output_c)
        for i in range(len(ls) - 1):
            lower = int(ls[i])
            upper = int(ls[i + 1])
            middle = int((lower + upper) / 2)
            self.class2value_list.append(middle / 255.)
        self.class2value_list.append(255. / 255.)
        assert len(self.class2value_list) == self.output_c

        _input_size = (args.input_h, args.input_w)
        self.convlstm1 = ConvLSTMClassifier(
            input_size=_input_size, input_dim=self.input_dim, output_c=self.output_c,
            class2value_list=self.class2value_list,
            hidden_dim=self.hidden_dims, kernel_size=self.kernel_size,
            num_layers=self.n_layers,
            batch_first=True, bias=True, return_all_layers=True,
            weight_init=args.weight_init,)

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

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, input_, target):
        out_e, hidden_e = self.encoder(input_)

        input_d = input_[:, -1, :, :, :]
        out_d, pred_d = self.decoder(input_d, hidden_e, target)

        bs, ts, c, h, w = out_d.size()
        out = out_d.contiguous().view(bs * ts, c, h, w)
        out = F.interpolate(out, size=(self.h, self.w), mode=self.mode)
        out = out.view(bs, ts, c, self.h, self.w)

        bs, ts, c, h, w = pred_d.size()
        pred = pred_d.contiguous().view(bs * ts, c, h, w)
        pred = F.interpolate(pred, size=(self.h, self.w), mode=self.mode)
        pred = pred.view(bs, ts, c, self.h, self.w)

        return out, pred


def encdec_clf(args):
    return Model(args)

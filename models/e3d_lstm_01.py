'''
from https://github.com/metrofun/E3D-LSTM
'''
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from .e3d_lstm import E3DLSTM


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        dtype = torch.float

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # TODO
        self.input_time_window = 96
        self.output_time_horizon = 24
        self.temporal_stride = 12
        self.temporal_frames = 24
        self.time_steps = (
            self.input_time_window - self.temporal_frames
        ) // self.temporal_stride + 1
        self.h, self.w = args.height, args.width
        self.mode = args.interpolation_mode
        self.logit_output = args.logit_output

        # CxTxHxW
        input_shape = (args.channels, self.temporal_frames, args.input_h, args.input_w)
        output_shape = (args.channels, self.output_time_horizon, args.input_h, args.input_w)  # noqa
        self.tau = 2
        hidden_size = 4
        padding = (1, 2, 2)
        kernel = (3, 5, 5)
        lstm_layers = 1

        self.encoder = E3DLSTM(
            input_shape, hidden_size, lstm_layers, kernel, self.tau
        ).type(dtype)
        self.decoder = nn.Conv3d(
                hidden_size * self.time_steps, output_shape[0], kernel, padding=padding
        ).type(dtype)
        self.apply(weights_init())

    def forward(self, input_, target):
        # (bs, ts, c, h, w) -> (bs, c, ts, h, w)
        input_ = input_.permute(0, 2, 1, 3, 4)
        frames_seq = []
        for indices in window(range(self.input_time_window), self.temporal_frames, self.temporal_stride,):  # noqa:
            # batch x channels x time x window x height
            frames_seq.append(input_[:, :, indices[0]:indices[-1] + 1])
        input_ = torch.stack(frames_seq, dim=0)
        input_ = input_.to(self.device)
        out_e = self.encoder(input_)
        out_d = self.decoder(out_e)
        # out_d.size() == (bs, c, ts, h, w) -> (bs, ts, c, h, w)
        out = out_d.permute(0, 2, 1, 3, 4)

        if not self.logit_output:
            out = torch.sigmoid(out)

        bs, ts, c, h, w = out.size()
        out = out.contiguous().view(bs * ts, c, h, w)
        out = F.interpolate(out, size=(self.h, self.w), mode=self.mode)
        out = out.view(bs, ts, c, self.h, self.w)
        return out


def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            # print m.__class__.__name__
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def window(seq, size=2, stride=1):
    """Returns a sliding window (of width n) over data from the iterable
       E.g., s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = []
    for elem in it:
        result.append(elem)
        if len(result) == size:
            yield result
            result = result[stride:]


def e3d_lstm_01(args):
    return Model(args)

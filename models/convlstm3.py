import random
import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(
        self, input_size, input_dim, hidden_dim, kernel_size, bias,
        weight_init='', **kwargs
    ):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.convCtm1 = kwargs.get('ConvCtm1', False)
        self.hadamard = kwargs.get('Hadamard', '').lower()

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size, padding=self.padding,
            bias=self.bias)
        if self.convCtm1:
            self.convWciCtm1 = nn.Conv2d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
            self.convWcfCtm1 = nn.Conv2d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
            self.convWcoCtm1 = nn.Conv2d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
        elif self.hadamard == 'channel':
            _size = (self.hidden_dim, 1, 1)
            self.Wci = nn.Parameter(torch.Tensor(*_size))
            self.Wcf = nn.Parameter(torch.Tensor(*_size))
            self.Wco = nn.Parameter(torch.Tensor(*_size))
        elif self.hadamard == 'normal':
            _size = (self.hidden_dim, self.height, self.width)
            self.Wci = nn.Parameter(torch.Tensor(*_size))
            self.Wcf = nn.Parameter(torch.Tensor(*_size))
            self.Wco = nn.Parameter(torch.Tensor(*_size))

        self._initialize_weights(weight_init)

    def _initialize_weights(self, weight_init):
        if weight_init == 'xavier_normal_':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        if self.convCtm1:
            WciCtm1 = self.convWciCtm1(c_cur)
            WcfCtm1 = self.convWcfCtm1(c_cur)
            WcoCtm1 = self.convWcoCtm1(c_cur)
        elif self.hadamard in ['normal', 'channel']:
            WciCtm1 = self.Wci * c_cur
            WcfCtm1 = self.Wcf * c_cur
            WcoCtm1 = self.Wco * c_cur

        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        if self.convCtm1 or self.hadamard == 'channel':
            i = torch.sigmoid(cc_i + WciCtm1)
            f = torch.sigmoid(cc_f + WcfCtm1)
            o = torch.sigmoid(cc_o + WcoCtm1)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        size = (batch_size, self.hidden_dim, self.height, self.width)
        return (
            Variable(torch.zeros(*size)).to(self.conv.weight.device),
            Variable(torch.zeros(*size)).to(self.conv.weight.device))


class ConvLSTM(nn.Module):

    def __init__(
        self, input_size, input_dim, output_c, output_ts,
        hidden_dim, kernel_size, num_layers,
        batch_first=False, bias=True, return_all_layers=False, teacher_forcing_ratio=0.,
        weight_init='', **kwargs
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim`
        # are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.output_c = output_c
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.weight_init = weight_init
        self.output_ts = output_ts
        self.do_batchnorm = kwargs.get('BatchNorm', False)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          weight_init=self.weight_init,
                                          **kwargs))

        self.cell_list = nn.ModuleList(cell_list)

        if self.do_batchnorm:
            batchnorm_list = []
            for i in range(0, self.num_layers):
                batchnorm_list.append(nn.BatchNorm2d(self.hidden_dim[i]))
            self.batchnorm_list = nn.ModuleList(batchnorm_list)

        self.conv1x1 = nn.Conv2d(sum(hidden_dim), output_c, (1, 1), stride=1, padding=0)

    def forward(self, input_tensor, hidden_state=None, target=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        input_ = input_tensor
        output_list = []
        for t_i in range(self.output_ts):
            h1_list = []
            for layer_i in range(self.num_layers):
                h0, c0 = hidden_state[layer_i]
                h1, c1 = self.cell_list[layer_i](input_tensor=input_, cur_state=[h0, c0])
                # update hidden_state with the new states
                hidden_state[layer_i] = [h1, c1]
                # h.size() = (bs, c, h, w)
                h1_list.append(h1)
                if self.do_batchnorm:
                    input_ = self.batchnorm_list[layer_i](h1)
                else:
                    input_ = h1
            stacked_h = torch.cat(h1_list, dim=1)
            logit = self.conv1x1(stacked_h)
            pred = torch.sigmoid(logit)
            output_list.append(logit)

            if self.teacher_forcing_ratio > random.random() and self.training:
                # target.size() = (bs, ts, c, h, w)
                input_ = target[:, t_i, :, :, :]
            else:
                input_ = pred

        # (ts, bs, c, h, w) -> (bs, ts, c, h, w)
        return torch.stack(output_list, dim=0).permute(1, 0, 2, 3, 4), None
        # TODO: return hiddens as second return value?

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple) or
            (isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

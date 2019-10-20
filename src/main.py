import argparse

from experiment import run
from utils import get_choices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--trainset', type=str, default='')
    parser.add_argument('--data_root', type=str, default='../inputs/')
    parser.add_argument('--train_csv', type=str, default='../inputs/training.csv')
    parser.add_argument('--valid_csv', type=str, default='../inputs/validation.csv')
    parser.add_argument('--height', type=int, default=672)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--input_reversing_ratio', type=float, default=0.)
    # TODO: refactor. use args.input_{h,w} instead
    parser.add_argument('--resize_to', type=int, nargs='+', default=(672, 512))
    parser.add_argument('--crop_params', type=int, nargs='+', default=(130, 40, 340, 420))
    parser.add_argument('--random_crop_delta', type=int, default=0)
    parser.add_argument('--smooth', type=str, default='step=25/minv=0/maxv=255')
    # network
    parser.add_argument('--model', type=str, default='convlstm_1_layer')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(5, 5))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[16, ])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=-1)
    parser.add_argument('--input_h', type=int, default=672)
    parser.add_argument('--input_w', type=int, default=512)
    parser.add_argument('--logit_output', action='store_true', default=False)
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--interpolation_mode', type=str, default='bilinear')
    choices = get_choices(['', 'xavier_normal_'])
    parser.add_argument('--weight_init', type=str, choices=choices, default='')
    parser.add_argument('--n_additional_convs', type=int, default=0)
    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    choices = ['L1', 'SmoothL1']
    choices = choices + list(map(lambda s: s.lower(), choices))
    parser.add_argument('--loss', type=str, choices=choices, default='L1')
    parser.add_argument('--clip_value', type=float, default=5.0)
    parser.add_argument('--max_norm', type=float, default=-1.)
    # optim
    parser.add_argument('--optim_str', type=str, default='Optim/lr=0.001/arg1=1/arg2=2')

    choices = ['', 'MultiStepLR']
    choices = choices + list(map(lambda s: s.lower(), choices))
    parser.add_argument('--scheduler', type=str, choices=choices, default='')
    parser.add_argument('--milestones', nargs='+', type=int)
    parser.add_argument('--gamma', nargs='+', type=float)
    # log
    parser.add_argument('--log_dir', type=str, default='../logs')
    parser.add_argument('--expid', type=str, default='')
    parser.add_argument('--freq_to_log_image', type=int, default=10)
    # misc
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', default=False)

    args, _ = parser.parse_known_args()

    # data
    args.trainset = 'trainset'
    args.train_csv = '../inputs/start_hr.list'
    args.valid_csv = '../inputs/validation.csv'
    args.height = 672
    args.width = 512
    args.crop_params = (0, 0, 672, 512)
    args.random_crop_delta = 0
    args.batch_size = 16
    args.n_workers = 8
    # model
    args.model = 'encdec_02'
    factor = 8
    args.input_h, args.input_w = int(args.height / factor), int(args.width / factor)
    args.resize_to = (args.input_h, args.input_w)
    args.hidden_dims = [8, 8, 8, ]
    args.n_layers = len(args.hidden_dims)
    args.kernel_size = (5, 5)
    args.residual = False
    args.logit_output = args.residual
    # training
    args.epochs = 100
    args.loss = 'L1'
    args.valid_loss = 'L1'
    # args.max_norm = 0.1
    # optim
    args.optim_str = 'RAdam/lr=0.001/betas=(0.9, 0.999)/weight_decay=0'

    args.freq_to_log_image = 1000

    run(args)

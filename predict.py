import argparse
import os

from tester import predict


def main(args):
    args.logdir = '../logs/20191029132611/'
    args.checkpoint = os.path.join(args.logdir, 'bestMAE.pt')
    args.is_making_submission = True

    factor = 4
    args.input_h = int(672 / factor)
    args.input_w = int(512 / factor)
    args.ts = 96
    args.input_ts = 96
    args.last_n_ts = 24
    args.target_ts = 0

    args.model = 'encdec_02/BatchNorm=False/ConvC0=True'
    args.hidden_dims = [64, 32, 16, ]
    args.n_layers = len(args.hidden_dims)
    args.loss = 'L1'
    args.logit_output = False
    args.teacher_forcing_ratio = -1
    args.residual = False
    predict(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_root', type=str, default='../inputs/')
    parser.add_argument('--csv', type=str, default='inference_terms.csv')
    parser.add_argument('--height', type=int, default=672)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--ts', type=int, default=96)
    parser.add_argument('--input_h', type=int, default=int(672 / 4))
    parser.add_argument('--input_w', type=int, default=int(512 / 4))
    parser.add_argument('--input_ts', type=int, default=96)
    parser.add_argument('--output_ts', type=int, default=24)
    parser.add_argument('--target_ts', type=int, default=24)
    parser.add_argument('--interpolation_mode', type=str, default='nearest')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--random_crop_delta', type=int, default=0)
    # network
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(5, 5))
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[16, ])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--weight_init', type=str, default='')
    # training
    parser.add_argument('--batch_size', type=int, default=1)
    # misc
    parser.add_argument('--logdir', type=str, default='../garbage')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    # prediction
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--last_n_ts', type=int, default=24)
    parser.add_argument('--dump', action='store_true', default=False)
    parser.add_argument(
        '--sample_submit', type=str, default='../inputs/sample_submit.csv')
    parser.add_argument('--is_making_submission', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    main(args)

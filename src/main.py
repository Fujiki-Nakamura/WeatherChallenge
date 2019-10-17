import argparse

from experiment import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_root', type=str, default='../../inputs/')
    parser.add_argument('--train_csv', type=str, default='training.csv')
    parser.add_argument('--valid_csv', type=str, default='validation.csv')
    parser.add_argument('--h', type=int, default=672)
    parser.add_argument('--w', type=int, default=512)
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=8)
    # network
    parser.add_argument('--model', type=str, default='ConvVAE')
    parser.add_argument('--input_h', type=int, default=672)
    parser.add_argument('--input_w', type=int, default=512)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--interpolation_mode', type=str, default='bilinear')
    parser.add_argument('--resume', type=str, default=None)
    # training
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--loss', type=str, default='MSE')
    # optim
    parser.add_argument('--optim_str', type=str, default='Adam/lr=1e-3/')
    # log
    parser.add_argument('--logdir', type=str, default='../../logs')
    parser.add_argument('--expid', type=str, default='')
    parser.add_argument('--log_image_freq', type=int, default=1)
    # misc
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    run(args)

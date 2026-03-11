import argparse

from experiment import train
from templates import *


TEMPLATE_REGISTRY = {
    'ffhq128_autoenc_130M': ffhq128_autoenc_130M,
    'ffhq128_autoenc_72M': ffhq128_autoenc_72M,
    'ffhq256_autoenc': ffhq256_autoenc,
    'horse128_autoenc': horse128_autoenc,
    'bedroom128_autoenc': bedroom128_autoenc,
    'celeba64d2c_autoenc': celeba64d2c_autoenc,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Diff-AE with optional z_d sparse conditioning.')
    parser.add_argument('--template',
                        default='ffhq128_autoenc_130M',
                        choices=sorted(TEMPLATE_REGISTRY.keys()))
    parser.add_argument('--gpus',
                        default='0',
                        help='Comma-separated GPU ids, e.g. "0,1,2,3".')
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--eval_program',
                        action='append',
                        default=None,
                        help='Optional eval program entries, can be repeated.')
    parser.add_argument('--name_suffix',
                        default='',
                        help='Append suffix to conf.name.')

    parser.add_argument('--use_zd_cond', action='store_true')
    parser.add_argument('--m', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lambda_l1', type=float, default=None)
    parser.add_argument('--ista_steps', type=int, default=None)
    parser.add_argument('--beta_align', type=float, default=None)
    parser.add_argument('--gamma_align', type=float, default=None)
    parser.add_argument('--lr_D', type=float, default=None)
    parser.add_argument('--lr_E', type=float, default=None)
    parser.add_argument('--lr_eps', type=float, default=None)
    parser.add_argument('--ddim_eta', type=float, default=None)
    parser.add_argument('--disable_zd_cond_only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    conf = TEMPLATE_REGISTRY[args.template]()

    conf.use_zd_cond = args.use_zd_cond
    if args.m is not None:
        conf.m = args.m
    if args.k is not None:
        conf.k = args.k
    if args.lambda_l1 is not None:
        conf.lambda_l1 = args.lambda_l1
    if args.ista_steps is not None:
        conf.ista_steps = args.ista_steps
    if args.beta_align is not None:
        conf.beta_align = args.beta_align
    if args.gamma_align is not None:
        conf.gamma_align = args.gamma_align
    if args.lr_D is not None:
        conf.lr_D = args.lr_D
    if args.lr_E is not None:
        conf.lr_E = args.lr_E
    if args.lr_eps is not None:
        conf.lr_eps = args.lr_eps
    if args.ddim_eta is not None:
        conf.ddim_eta = args.ddim_eta
    if args.disable_zd_cond_only:
        conf.zd_cond_only = False

    if conf.use_zd_cond:
        conf.name = f'{conf.name}_zd'
    if args.name_suffix:
        conf.name = f'{conf.name}{args.name_suffix}'
    if args.eval_program is not None:
        conf.eval_programs = args.eval_program

    # Rebuild model_conf after applying CLI overrides.
    conf.make_model_conf()

    gpus = [int(g) for g in args.gpus.split(',') if g != '']
    train(conf, gpus=gpus, nodes=args.nodes, mode=args.mode)


if __name__ == '__main__':
    main()

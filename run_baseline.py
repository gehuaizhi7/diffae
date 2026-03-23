#!/usr/bin/env python3

import argparse

from experiment import train
from templates import *


def parse_gpus(gpus_arg: str):
    gpus = []
    for part in gpus_arg.split(','):
        part = part.strip()
        if not part:
            continue
        gpus.append(int(part))
    if not gpus:
        raise ValueError('Expected at least one GPU id, e.g. --gpus 0,1,2,3')
    return gpus


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run stage-1 FFHQ128 baseline autoencoder training.')
    parser.add_argument('--gpus',
                        default='4,5,6,7',
                        help='Comma-separated GPU ids, e.g. 0,1,2,3')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project',
                        default='diffae',
                        help='W&B project name')
    parser.add_argument('--wandb-entity',
                        default=None,
                        help='W&B entity/team name')
    parser.add_argument('--wandb-name',
                        default='ffhq128_baseline',
                        help='W&B run name')
    parser.add_argument('--exp-name',
                        default='ffhq128_autoenc_130M_baseline',
                        help='Experiment/checkpoint directory name')
    parser.add_argument('--wandb-mode',
                        default='online',
                        choices=['online', 'offline'],
                        help='W&B logging mode')
    parser.add_argument('--batch-size',
                        type=int,
                        default=None,
                        help='Override total training batch size')
    return parser


def autoscale_batch_sizes(conf, num_gpus: int, template_num_gpus: int = 4):
    conf.batch_size = max(1, conf.batch_size * num_gpus // template_num_gpus)
    conf.batch_size_eval = max(1, conf.batch_size_eval * num_gpus //
                               template_num_gpus)


def main():
    args = build_parser().parse_args()
    gpus = parse_gpus(args.gpus)

    conf = ffhq128_autoenc_130M()
    conf.name = args.exp_name
    autoscale_batch_sizes(conf, num_gpus=len(gpus))
    if args.batch_size is not None:
        conf.batch_size = args.batch_size
        conf.batch_size_eval = args.batch_size
    conf.use_wandb = args.wandb
    conf.wandb_project = args.wandb_project
    conf.wandb_entity = args.wandb_entity
    conf.wandb_name = args.wandb_name
    conf.wandb_mode = args.wandb_mode

    train(conf, gpus=gpus)


if __name__ == '__main__':
    main()

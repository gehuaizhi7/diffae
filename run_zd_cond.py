import argparse
import os

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
    parser.add_argument('--ista_solver',
                        '--ista-solver',
                        dest='ista_solver',
                        choices=['ista', 'fista'],
                        default=None,
                        help=('Sparse-code inference algorithm used by the '
                              'unrolled solver.'))
    parser.add_argument('--lambda_ze_zd',
                        type=float,
                        default=None,
                        help=('Coefficient for the ||z_e - z_d||_2^2 loss used '
                              'in joint training and fixed-D stage 2.'))
    parser.add_argument('--beta_align', type=float, default=None)
    parser.add_argument('--gamma_align', type=float, default=None)
    parser.add_argument('--lr_D', type=float, default=None)
    parser.add_argument('--zd_d_only_samples',
                        '--d_only_samples',
                        dest='zd_d_only_samples',
                        type=int,
                        default=None,
                        help='Train only the dictionary D for this many samples before joint training.')
    parser.add_argument('--zd_train_mode',
                        '--zd-train-mode',
                        dest='zd_train_mode',
                        choices=['joint', 'dict_then_diffusion'],
                        default=None,
                        help=('z_d training schedule. '
                              '"joint" keeps the current path, while '
                              '"dict_then_diffusion" runs stage 1 '
                              '(dictionary only) then stage 2 '
                              '(configured by --zd-stage2-mode).'))
    parser.add_argument('--zd_stage1_samples',
                        '--zd-stage1-samples',
                        dest='zd_stage1_samples',
                        type=int,
                        default=None,
                        help=('For --zd_train_mode=dict_then_diffusion, '
                              'number of samples in stage 1 before switching '
                              'to stage 2.'))
    parser.add_argument('--zd_stage2_mode',
                        '--zd-stage2-mode',
                        dest='zd_stage2_mode',
                        choices=['alternating', 'fixed_d'],
                        default=None,
                        help=('Stage-2 behavior for '
                              '--zd_train_mode=dict_then_diffusion. '
                              '"alternating" keeps the current alternating '
                              'diffusion/dictionary updates, while '
                              '"fixed_d" freezes D and trains only encoder '
                              '+ decoder with L_diff + lambda_ze_zd * '
                              '||z_e - z_d||_2^2.'))
    parser.add_argument('--zd_stage2_use_zstar_cond',
                        '--zd-stage2-use-zstar-cond',
                        dest='zd_stage2_use_zstar_cond',
                        action='store_true',
                        help=('For --zd_train_mode=dict_then_diffusion, use '
                              'the sparse code z* as the diffusion condition '
                              'for stage-2 diffusion steps instead of '
                              'z_d = D z*.'))
    parser.add_argument('--lr_E', type=float, default=None)
    parser.add_argument('--lr_eps', type=float, default=None)
    parser.add_argument('--ddim_eta', type=float, default=None)
    parser.add_argument('--disable_zd_cond_only', action='store_true')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='Per-process train batch size override.')
    parser.add_argument('--total_samples',
                        type=int,
                        default=None,
                        help='Total number of training samples to consume.')
    parser.add_argument('--batch_size_eval',
                        type=int,
                        default=None,
                        help='Per-process eval batch size override.')
    parser.add_argument('--recon_every_samples',
                        type=int,
                        default=None,
                        help='How often to log reconstruction grids.')
    parser.add_argument('--accum_batches',
                        type=int,
                        default=None,
                        help='Gradient accumulation steps.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=None,
                        help='Dataloader workers override.')
    parser.add_argument('--grad_checkpoint',
                        action='store_true',
                        help='Enable UNet gradient checkpointing.')
    parser.add_argument('--enc_grad_checkpoint',
                        action='store_true',
                        help='Enable encoder gradient checkpointing.')
    parser.add_argument('--use_wandb',
                        action='store_true',
                        help='Enable W&B logging in addition to TensorBoard.')
    parser.add_argument('--wandb_project',
                        default=None,
                        help='W&B project name.')
    parser.add_argument('--wandb_entity',
                        default=None,
                        help='W&B entity (team/user).')
    parser.add_argument('--wandb_name',
                        default=None,
                        help='W&B run name.')
    parser.add_argument('--wandb_mode',
                        choices=['online', 'offline'],
                        default=None,
                        help='W&B mode.')
    parser.add_argument('--wandb_tags',
                        default=None,
                        help='Comma-separated W&B tags.')
    parser.add_argument('--pretrain_exp',
                        '--pretrain-exp',
                        dest='pretrain_exp',
                        default=None,
                        help=('Experiment name under checkpoints/ to load with '
                              'strict=False before training, e.g. '
                              'ffhq128_autoenc_130M_baseline_compare.'))
    parser.add_argument('--pretrain_ckpt',
                        '--pretrain-ckpt',
                        dest='pretrain_ckpt',
                        default=None,
                        help='Explicit checkpoint path to load with strict=False before training.')
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
    if args.ista_solver is not None:
        conf.ista_solver = args.ista_solver
    if args.lambda_ze_zd is not None:
        conf.lambda_ze_zd = args.lambda_ze_zd
    if args.beta_align is not None:
        conf.beta_align = args.beta_align
    if args.gamma_align is not None:
        conf.gamma_align = args.gamma_align
    if args.lr_D is not None:
        conf.lr_D = args.lr_D
    if args.zd_d_only_samples is not None:
        conf.zd_d_only_samples = args.zd_d_only_samples
    if args.zd_train_mode is not None:
        conf.zd_train_mode = args.zd_train_mode
    if args.zd_stage1_samples is not None:
        conf.zd_stage1_samples = args.zd_stage1_samples
    if args.zd_stage2_mode is not None:
        conf.zd_stage2_mode = args.zd_stage2_mode
    if args.zd_stage2_use_zstar_cond:
        conf.zd_stage2_use_zstar_cond = True
    if args.lr_E is not None:
        conf.lr_E = args.lr_E
    if args.lr_eps is not None:
        conf.lr_eps = args.lr_eps
    if args.ddim_eta is not None:
        conf.ddim_eta = args.ddim_eta
    if args.disable_zd_cond_only:
        conf.zd_cond_only = False
    if args.batch_size is not None:
        conf.batch_size = args.batch_size
    if args.total_samples is not None:
        conf.total_samples = args.total_samples
    if args.batch_size_eval is not None:
        conf.batch_size_eval = args.batch_size_eval
    if args.recon_every_samples is not None:
        conf.recon_every_samples = args.recon_every_samples
    if args.accum_batches is not None:
        conf.accum_batches = args.accum_batches
    if args.num_workers is not None:
        conf.num_workers = args.num_workers
    if args.grad_checkpoint:
        conf.net_beatgans_gradient_checkpoint = True
    if args.enc_grad_checkpoint:
        conf.net_enc_grad_checkpoint = True
    if args.use_wandb:
        conf.use_wandb = True
    if args.wandb_project is not None:
        conf.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        conf.wandb_entity = args.wandb_entity
    if args.wandb_name is not None:
        conf.wandb_name = args.wandb_name
    if args.wandb_mode is not None:
        conf.wandb_mode = args.wandb_mode
    if args.wandb_tags:
        conf.wandb_tags = tuple(
            t.strip() for t in args.wandb_tags.split(',') if t.strip())
    if args.pretrain_exp is not None and args.pretrain_ckpt is not None:
        raise ValueError('Specify only one of --pretrain_exp or --pretrain_ckpt.')
    if args.pretrain_exp is not None:
        conf.pretrain = PretrainConfig(
            name=args.pretrain_exp,
            path=os.path.join('checkpoints', args.pretrain_exp, 'last.ckpt'),
        )
    elif args.pretrain_ckpt is not None:
        pretrain_name = os.path.splitext(os.path.basename(args.pretrain_ckpt))[0]
        conf.pretrain = PretrainConfig(name=pretrain_name,
                                       path=args.pretrain_ckpt)

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

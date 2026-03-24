import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.dictionary import SparseDictionary
from model.ista import ista


def main():
    parser = argparse.ArgumentParser(
        description=('Run one-batch ISTA/FISTA sanity check '
                     '(objective should decrease).')
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--solver', choices=['ista', 'fista'], default='ista')
    parser.add_argument('--lambda_l1', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    z_e = torch.randn(args.batch_size, args.m)
    dictionary = SparseDictionary(args.m, args.k)

    out = ista(z_e=z_e,
               atoms=dictionary.atoms.detach(),
               lambda_l1=args.lambda_l1,
               steps=args.steps,
               solver=args.solver,
               return_history=True)
    obj_mean = out.objectives.mean(dim=0)
    deltas = obj_mean[1:] - obj_mean[:-1]
    monotonic = bool((deltas <= 1e-6).all().item())
    decreased = bool((obj_mean[-1] <= obj_mean[0] + 1e-6).item())

    print(f'{args.solver.upper()} objective (batch mean):', obj_mean.tolist())
    print('Final objective <= initial:', decreased)
    print('Monotonic non-increasing:', monotonic)
    if args.solver == 'ista' and not monotonic:
        raise SystemExit(1)
    if args.solver == 'fista' and not decreased:
        raise SystemExit(1)
    print('PASS')


if __name__ == '__main__':
    main()

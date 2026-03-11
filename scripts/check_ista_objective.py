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
        description='Run one-batch ISTA sanity check (objective should decrease).'
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--steps', type=int, default=8)
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
               return_history=True)
    obj_mean = out.objectives.mean(dim=0)
    deltas = obj_mean[1:] - obj_mean[:-1]
    monotonic = bool((deltas <= 1e-6).all().item())

    print('ISTA objective (batch mean):', obj_mean.tolist())
    print('Monotonic non-increasing:', monotonic)
    if not monotonic:
        raise SystemExit(1)
    print('PASS')


if __name__ == '__main__':
    main()

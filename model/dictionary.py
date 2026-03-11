import torch
import torch.nn.functional as F
from torch import Tensor, distributed, nn


class SparseDictionary(nn.Module):
    """
    Dictionary D with column-normalized atoms.

    D has shape (m, k), where m is feature dim and k is number of atoms.
    """
    def __init__(self, m: int, k: int, eps: float = 1e-8):
        super().__init__()
        self.m = m
        self.k = k
        self.eps = eps
        atoms = torch.randn(m, k)
        atoms = F.normalize(atoms, p=2, dim=0, eps=eps)
        # D can be optimized directly via the training objective.
        self.atoms = nn.Parameter(atoms, requires_grad=True)

    def decode(self, code: Tensor) -> Tensor:
        return code @ self.atoms.t()

    def reconstruction_loss(self, z_e: Tensor, code: Tensor) -> Tensor:
        recon = self.decode(code)
        return F.mse_loss(recon, z_e)

    @torch.no_grad()
    def normalize_columns(self):
        self.atoms.data = F.normalize(self.atoms.data,
                                      p=2,
                                      dim=0,
                                      eps=self.eps)

    @torch.no_grad()
    def bcd_update(self, z_e: Tensor, code: Tensor, lr: float):
        """
        Gradient step on D for ||z_e - D z*||^2 with detached z* (BCD style).
        """
        recon = self.decode(code)
        residual = recon - z_e
        grad = (2.0 / residual.numel()) * residual.t().matmul(code)
        if distributed.is_initialized():
            distributed.all_reduce(grad)
            grad /= distributed.get_world_size()
        self.atoms.data -= lr * grad
        self.normalize_columns()
        return residual.pow(2).mean()

    def max_offdiag(self) -> Tensor:
        gram = self.atoms.t().matmul(self.atoms)
        offdiag = gram - torch.diag_embed(torch.diagonal(gram))
        return offdiag.abs().max()

    @staticmethod
    def dead_atom_fraction(code: Tensor, eps: float = 1e-6) -> Tensor:
        if code.numel() == 0:
            return torch.tensor(0.0, device=code.device)
        used = (code.abs() > eps).any(dim=0).float()
        return 1.0 - used.mean()

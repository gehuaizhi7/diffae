import math
from dataclasses import dataclass

import torch
from torch import Tensor


def soft_threshold(x: Tensor, thresh: Tensor) -> Tensor:
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


def ista_objective(z_e: Tensor, code: Tensor, atoms: Tensor,
                   lambda_l1: float) -> Tensor:
    recon = code @ atoms.t()
    recon_term = 0.5 * ((z_e - recon)**2).sum(dim=1)
    sparsity_term = lambda_l1 * code.abs().sum(dim=1)
    return recon_term + sparsity_term


@dataclass
class ISTAResult:
    code: Tensor
    recon: Tensor
    objectives: Tensor


def ista(
    z_e: Tensor,
    atoms: Tensor,
    lambda_l1: float,
    steps: int,
    solver: str = 'ista',
    step_size: float = None,
    init_code: Tensor = None,
    return_history: bool = True,
) -> ISTAResult:
    """
    Solve min_z 0.5||z_e - D z||_2^2 + lambda_l1 ||z||_1 using
    unrolled ISTA or FISTA updates.

    Shapes:
        z_e:   (B, m)
        atoms: (m, k)  # dictionary D
        code:  (B, k)
    """
    assert z_e.dim() == 2, z_e.shape
    assert atoms.dim() == 2, atoms.shape
    assert z_e.shape[1] == atoms.shape[0], (z_e.shape, atoms.shape)

    solver = solver.lower()
    if solver not in ['ista', 'fista']:
        raise ValueError(
            f'Unsupported solver={solver!r}. Expected "ista" or "fista".')

    batch, _ = z_e.shape
    k = atoms.shape[1]
    if init_code is None:
        code = torch.zeros(batch, k, device=z_e.device, dtype=z_e.dtype)
    else:
        code = init_code

    if step_size is None:
        # Lipschitz constant of grad 0.5||z_e - D z||_2^2 is ||D||_2^2.
        spectral_norm = torch.norm(atoms, p=2)
        step_size = 1.0 / (spectral_norm * spectral_norm + 1e-8)

    if torch.is_tensor(step_size):
        step_tensor = step_size.to(device=z_e.device, dtype=z_e.dtype)
    else:
        step_tensor = torch.tensor(float(step_size),
                                   device=z_e.device,
                                   dtype=z_e.dtype)
    thresh = step_tensor * float(lambda_l1)

    history = [ista_objective(z_e, code, atoms, lambda_l1)]
    if solver == 'ista':
        for _ in range(steps):
            grad = (code @ atoms.t() - z_e) @ atoms
            code = soft_threshold(code - step_tensor * grad, thresh)
            if return_history:
                history.append(ista_objective(z_e, code, atoms, lambda_l1))
    else:
        momentum_code = code
        t = 1.0
        for _ in range(steps):
            grad = (momentum_code @ atoms.t() - z_e) @ atoms
            next_code = soft_threshold(momentum_code - step_tensor * grad,
                                       thresh)
            next_t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            momentum = (t - 1.0) / next_t
            momentum_code = next_code + momentum * (next_code - code)
            code = next_code
            t = next_t
            if return_history:
                history.append(ista_objective(z_e, code, atoms, lambda_l1))

    recon = code @ atoms.t()
    if return_history:
        objectives = torch.stack(history, dim=1)
    else:
        objectives = ista_objective(z_e, code, atoms, lambda_l1)[:, None]
    return ISTAResult(code=code, recon=recon, objectives=objectives)

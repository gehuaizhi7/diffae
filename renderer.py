import torch

from config import *
from model.ista import ista
from torch.cuda import amp


def _extract_cond_tensor(cond):
    if cond is None:
        return None
    if isinstance(cond, dict):
        if 'cond' not in cond:
            raise KeyError('Expected conditioning dict to contain a "cond" key.')
        return cond['cond']
    return cond


def build_condition_model_kwargs(conf: TrainConfig,
                                 model: BeatGANsAutoencModel,
                                 x_start=None,
                                 cond=None):
    cond = _extract_cond_tensor(cond)
    model_kwargs = {
        'x_start': x_start,
        'cond': cond,
    }

    if not conf.use_zd_cond:
        return model_kwargs

    z_e = cond
    if z_e is None:
        if x_start is None:
            raise ValueError(
                'z_d-conditioned rendering requires x_start or an explicit cond tensor.'
            )
        if hasattr(model, 'encoder'):
            z_e = model.encoder(x_start)
        elif hasattr(model, 'encode'):
            z_e = _extract_cond_tensor(model.encode(x_start))
        else:
            raise RuntimeError(
                'z_d-conditioned rendering requires a model with encoder/encode().'
            )

    zd_dictionary = getattr(model, '_external_zd_dictionary', None)
    if zd_dictionary is None:
        raise RuntimeError(
            'Missing external z_d dictionary on the model. '
            'Attach it before z_d-conditioned rendering/evaluation.'
        )

    ista_out = ista(z_e=z_e,
                    atoms=zd_dictionary.atoms,
                    lambda_l1=conf.lambda_l1,
                    steps=conf.ista_steps,
                    solver=conf.ista_solver,
                    return_history=False)
    z_star = ista_out.code
    z_d = zd_dictionary.decode(z_star)

    if (conf.zd_train_mode == 'dict_then_diffusion'
            and conf.zd_stage2_use_zstar_cond):
        zd_cond = z_star
    else:
        zd_cond = z_d

    model_kwargs['z_d'] = zd_cond
    if not conf.zd_cond_only:
        model_kwargs['cond'] = z_e
    else:
        model_kwargs['cond'] = None
    return model_kwargs


def render_uncondition(conf: TrainConfig,
                       model: BeatGANsAutoencModel,
                       x_T,
                       sampler: Sampler,
                       latent_sampler: Sampler,
                       conds_mean=None,
                       conds_std=None,
                       clip_latent_noise: bool = False):
    device = x_T.device
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.can_sample()
        return sampler.sample(model=model, noise=x_T, eta=conf.ddim_eta)
    elif conf.train_mode.is_latent_diffusion():
        model: BeatGANsAutoencModel
        if conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = torch.randn(len(x_T), conf.style_ch, device=device)
        else:
            raise NotImplementedError()

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        cond = latent_sampler.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=conf.latent_clip_sample,
            eta=conf.ddim_eta,
        )

        if conf.latent_znormalize:
            cond = cond * conds_std.to(device) + conds_mean.to(device)

        # the diffusion on the model
        return sampler.sample(model=model,
                              noise=x_T,
                              cond=cond,
                              eta=conf.ddim_eta)
    else:
        raise NotImplementedError()


def render_condition(
    conf: TrainConfig,
    model: BeatGANsAutoencModel,
    x_T,
    sampler: Sampler,
    x_start=None,
    cond=None,
    latent_sampler: Sampler = None,
):
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        model_kwargs = build_condition_model_kwargs(conf=conf,
                                                    model=model,
                                                    x_start=x_start,
                                                    cond=cond)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs=model_kwargs,
                              eta=conf.ddim_eta)
    else:
        raise NotImplementedError()

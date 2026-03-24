#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / "tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_ROOT))
os.environ.setdefault("TMP", str(TMP_ROOT))
os.environ.setdefault("TEMP", str(TMP_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.utils import save_image

from dataset import ImageDataset
from experiment import LitModel
from model.encoder import EncoderAdapter
from model.ista import ista
from renderer import build_condition_model_kwargs
from templates import (bedroom128_autoenc, celeba64d2c_autoenc,
                       ffhq128_autoenc_130M, ffhq128_autoenc_72M,
                       ffhq256_autoenc, horse128_autoenc)


TEMPLATE_REGISTRY = {
    "ffhq128_autoenc_130M": ffhq128_autoenc_130M,
    "ffhq128_autoenc_72M": ffhq128_autoenc_72M,
    "ffhq256_autoenc": ffhq256_autoenc,
    "horse128_autoenc": horse128_autoenc,
    "bedroom128_autoenc": bedroom128_autoenc,
    "celeba64d2c_autoenc": celeba64d2c_autoenc,
}

ENUM_FIELDS = [
    "train_mode",
    "model_name",
    "model_type",
    "diffusion_type",
    "optimizer",
    "beatgans_gen_type",
    "beatgans_loss_type",
    "beatgans_model_mean_type",
    "beatgans_model_var_type",
    "latent_gen_type",
    "latent_loss_type",
    "latent_model_mean_type",
    "latent_model_var_type",
    "net_latent_activation",
    "net_latent_net_last_act",
    "net_latent_net_type",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load one dataset image, compute z*, then either invert to x_T or "
            "use random noise, render from (x_T, z*D), and perturb each active "
            "z* coefficient to save the resulting images."
        ))
    parser.add_argument(
        "--ckpt",
        default="checkpoints/ffhq128_autoenc_130M_zd_3stages_alter_new/last.ckpt",
        help="Checkpoint to load.",
    )
    parser.add_argument(
        "--template",
        default="ffhq128_autoenc_130M",
        choices=sorted(TEMPLATE_REGISTRY.keys()),
        help="Base template used to restore enum-valued config fields.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cpu, cuda:0, cuda:7. Overrides --gpu.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Convenience alias for --device cuda:<gpu>.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of DDIM steps for inversion and sampling.",
    )
    parser.add_argument(
        "--noise_mode",
        choices=["invert", "random"],
        default="invert",
        help="How to obtain x_T: DDIM inversion from the input image or fresh random noise.",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=None,
        help="Seed used when --noise_mode=random. Defaults to --seed.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to load.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split for built-in datasets.",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Optional image folder. If set, images are loaded from here instead of the dataset.",
    )
    parser.add_argument(
        "--exts",
        default="jpg,jpeg,JPG,png,PNG",
        help="Comma-separated extensions for --input_dir.",
    )
    parser.add_argument(
        "--deltas",
        default="-4.00,-2.00,2.00,4.00",
        help=(
            "Comma-separated perturbation values. In add mode, each delta is "
            "added directly to the selected z* coefficient. In relative mode, "
            "each delta scales it as z_i * (1 + delta)."
        ),
    )
    parser.add_argument(
        "--perturb_mode",
        choices=["relative", "add"],
        default="add",
        help="How to perturb active z* entries. Default is add for stronger single-atom edits.",
    )
    parser.add_argument(
        "--nonzero_eps",
        type=float,
        default=1e-6,
        help="Threshold for treating a z* entry as non-zero.",
    )
    parser.add_argument(
        "--max_active",
        type=int,
        default=None,
        help="Optional cap on how many active coefficients to perturb, ranked by |z*|.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save outputs. Defaults to perturb_outputs/<exp>/<name>_T<steps>_<noise_mode>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch seed used for deterministic sampling when eta > 0.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Use the raw model instead of ema_model.",
    )
    return parser.parse_args()


def choose_device(args) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    if args.gpu is not None:
        return torch.device(f"cuda:{args.gpu}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def parse_float_list(text: str, flag_name: str) -> List[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError(f"{flag_name} must contain at least one float.")
    return values


def restore_conf(template_name: str, ckpt_path: Path):
    base_conf = TEMPLATE_REGISTRY[template_name]()
    conf = TEMPLATE_REGISTRY[template_name]()
    state = torch.load(str(ckpt_path), map_location="cpu")
    conf.from_dict(state["hyper_parameters"])
    for field in ENUM_FIELDS:
        setattr(conf, field, getattr(base_conf, field))
    conf.model_conf = None
    conf.pretrain = None
    conf.continue_from = None
    conf.eval_programs = None
    conf.batch_size_eval = conf.batch_size_eval or conf.batch_size
    conf.make_model_conf()
    return conf, state


def load_single_image(conf, args, index: int) -> Tuple[torch.Tensor, str]:
    if args.input_dir is not None:
        exts = [x.strip() for x in args.exts.split(",") if x.strip()]
        dataset = ImageDataset(args.input_dir,
                               image_size=conf.img_size,
                               exts=exts,
                               do_augment=False,
                               sort_names=True)
        item = dataset[index]
        name = Path(dataset.paths[index]).stem
        return item["img"][None], name

    dataset = conf.make_dataset(split=args.split, do_augment=False)
    item = dataset[index]
    return item["img"][None], f"idx{index:05d}"


def to_vis(imgs: torch.Tensor) -> torch.Tensor:
    return ((imgs.detach().clamp(-1, 1) + 1) / 2).cpu()


def format_delta(delta: float) -> str:
    sign = "p" if delta >= 0 else "m"
    magnitude = f"{abs(delta):.4f}".rstrip("0").rstrip(".")
    magnitude = magnitude.replace(".", "p")
    return f"{sign}{magnitude or '0'}"


def repeat_first_dim(x: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if repeats == 1:
        return x
    return x.repeat(repeats, *([1] * (x.dim() - 1)))


def build_manual_condition_model_kwargs(conf,
                                        x_start: Optional[torch.Tensor],
                                        z_e: Optional[torch.Tensor],
                                        z_star: Optional[torch.Tensor],
                                        z_d: Optional[torch.Tensor]):
    model_kwargs = {
        "x_start": x_start,
        "cond": z_e,
    }
    if conf.use_zd_cond:
        model_kwargs["z_d"] = z_star if conf.zd_stage2_use_zstar_cond else z_d
        if conf.zd_cond_only:
            model_kwargs["cond"] = None
    return model_kwargs


def apply_perturbation(base_z_star: torch.Tensor, delta: float,
                       perturb_mode: str) -> torch.Tensor:
    if perturb_mode == "relative":
        return base_z_star * (1.0 + delta)
    if perturb_mode == "add":
        return base_z_star + delta
    raise ValueError(f"Unsupported perturb_mode={perturb_mode!r}")


def get_display_order(deltas: List[float]) -> List[int]:
    negatives = sorted((i for i, delta in enumerate(deltas) if delta < 0),
                       key=lambda i: deltas[i])
    nonnegatives = sorted((i for i, delta in enumerate(deltas) if delta >= 0),
                          key=lambda i: deltas[i])
    return negatives + nonnegatives


def build_row_panel(orig_vis: torch.Tensor,
                    recon_vis: torch.Tensor,
                    perturbed_vis: torch.Tensor,
                    deltas: List[float]) -> torch.Tensor:
    display_order = get_display_order(deltas)
    ordered_vis = perturbed_vis[display_order]
    num_negative = sum(delta < 0 for delta in deltas)
    parts = [orig_vis]
    if num_negative > 0:
        parts.append(ordered_vis[:num_negative])
    parts.append(recon_vis)
    if num_negative < len(display_order):
        parts.append(ordered_vis[num_negative:])
    return torch.cat(parts, dim=0)


def main():
    args = parse_args()
    device = choose_device(args)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    deltas = parse_float_list(args.deltas, "--deltas")
    display_order = get_display_order(deltas)
    noise_seed = args.seed if args.noise_seed is None else args.noise_seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    conf, state = restore_conf(args.template, ckpt_path)
    lit = LitModel(conf)
    missing, unexpected = lit.load_state_dict(state["state_dict"], strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    if not conf.use_zd_cond or lit.zd_dictionary is None:
        raise RuntimeError(
            "This script expects a checkpoint trained with use_zd_cond=True."
        )

    model = lit.model if args.no_ema else lit.ema_model
    lit.to(device)
    lit.eval()
    model.eval()

    img, name = load_single_image(conf, args, args.index)
    img = img.to(device)
    sampler = conf._make_diffusion_conf(T=args.steps).make_sampler()
    eta = conf.ddim_eta
    encoder = EncoderAdapter(model)

    exp_name = conf.name or ckpt_path.parent.name
    default_out = Path("perturb_outputs") / exp_name / f"{name}_T{args.steps}_{args.noise_mode}"
    out_root = Path(args.output_dir) if args.output_dir else default_out
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if args.noise_mode == "invert":
            invert_kwargs = build_condition_model_kwargs(conf, model, x_start=img)
            x_T = sampler.ddim_reverse_sample_loop(model=model,
                                                   x=img,
                                                   clip_denoised=True,
                                                   model_kwargs=invert_kwargs,
                                                   eta=eta)["sample"]
            x_T_source = "ddim_inversion"
        elif args.noise_mode == "random":
            torch.manual_seed(noise_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(noise_seed)
            x_T = torch.randn_like(img)
            x_T_source = f"random(seed={noise_seed})"
        else:
            raise ValueError(f"Unsupported noise_mode={args.noise_mode!r}")

        z_e = encoder(img)
        ista_out = ista(z_e=z_e,
                        atoms=lit.zd_dictionary.atoms,
                        lambda_l1=conf.lambda_l1,
                        steps=conf.ista_steps,
                        solver=conf.ista_solver,
                        return_history=True)
        z_star = ista_out.code
        z_d = lit.zd_dictionary.decode(z_star)

        base_kwargs = build_manual_condition_model_kwargs(conf=conf,
                                                          x_start=img,
                                                          z_e=z_e,
                                                          z_star=z_star,
                                                          z_d=z_d)
        recon = sampler.sample(model=model,
                               noise=x_T,
                               model_kwargs=base_kwargs,
                               eta=eta)

    active_indices = torch.nonzero(z_star[0].abs() > args.nonzero_eps,
                                   as_tuple=False).flatten()
    all_active = active_indices.tolist()
    if len(all_active) == 0:
        raise RuntimeError(
            f"No active z* entries found above eps={args.nonzero_eps}."
        )

    active_values = z_star[0, active_indices].detach().cpu()
    order = torch.argsort(active_values.abs(), descending=True)
    active_indices = active_indices[order]
    if args.max_active is not None:
        active_indices = active_indices[:args.max_active]

    orig_vis = to_vis(img)
    recon_vis = to_vis(recon)
    diff_vis = (recon_vis - orig_vis).abs()

    save_image(orig_vis, str(out_root / "original.png"))
    save_image(recon_vis, str(out_root / "reconstruction.png"))
    save_image(diff_vis, str(out_root / "reconstruction_abs_diff.png"))
    save_image(torch.cat([orig_vis, recon_vis, diff_vis], dim=0),
               str(out_root / "original_recon_diff.png"),
               nrow=3)

    torch.save({
        "x_T": x_T.detach().cpu(),
        "z_e": z_e.detach().cpu(),
        "z_star": z_star.detach().cpu(),
        "z_d": z_d.detach().cpu(),
        "reconstruction": recon.detach().cpu(),
        "ista_objectives": ista_out.objectives.detach().cpu(),
    }, out_root / "latents.pt")

    perturb_records = []
    combined_row_panels = []
    with torch.no_grad():
        for atom_idx in active_indices.tolist():
            base_value = float(z_star[0, atom_idx].item())
            perturbed_codes = []
            value_records = []
            for delta in deltas:
                new_code = z_star.clone()
                new_code[:, atom_idx] = apply_perturbation(new_code[:, atom_idx],
                                                           delta=delta,
                                                           perturb_mode=args.perturb_mode)
                perturbed_codes.append(new_code)
                value_records.append({
                    "delta": delta,
                    "value": float(new_code[0, atom_idx].item()),
                })

            batch_z_star = torch.cat(perturbed_codes, dim=0)
            batch_z_e = repeat_first_dim(z_e, len(deltas))
            batch_x_start = repeat_first_dim(img, len(deltas))
            batch_x_T = repeat_first_dim(x_T, len(deltas))
            batch_z_d = lit.zd_dictionary.decode(batch_z_star)
            batch_kwargs = build_manual_condition_model_kwargs(conf=conf,
                                                               x_start=batch_x_start,
                                                               z_e=batch_z_e,
                                                               z_star=batch_z_star,
                                                               z_d=batch_z_d)
            perturbed_imgs = sampler.sample(model=model,
                                            noise=batch_x_T,
                                            model_kwargs=batch_kwargs,
                                            eta=eta)
            perturbed_vis = to_vis(perturbed_imgs)

            panel = build_row_panel(orig_vis=orig_vis,
                                    recon_vis=recon_vis,
                                    perturbed_vis=perturbed_vis,
                                    deltas=deltas)
            panel_path = out_root / f"atom_{atom_idx:04d}_grid.png"
            save_image(panel, str(panel_path), nrow=2 + len(deltas))
            combined_row_panels.append(panel)

            variant_files = []
            for idx in display_order:
                delta = deltas[idx]
                value_record = value_records[idx]
                variant_vis = perturbed_vis[idx]
                delta_label = format_delta(delta)
                variant_path = out_root / f"atom_{atom_idx:04d}_{delta_label}.png"
                save_image(variant_vis, str(variant_path))
                variant_files.append({
                    "delta": delta,
                    "value": value_record["value"],
                    "path": str(variant_path),
                })

            perturb_records.append({
                "index": atom_idx,
                "base_value": base_value,
                "grid_path": str(panel_path),
                "variants": variant_files,
            })

    all_active_rows_path = None
    if combined_row_panels:
        all_active_rows = torch.cat(combined_row_panels, dim=0)
        all_active_rows_path = out_root / "all_active_rows.png"
        save_image(all_active_rows,
                   str(all_active_rows_path),
                   nrow=2 + len(deltas))

    all_active_summary = [{
        "index": int(idx),
        "value": float(z_star[0, idx].item()),
        "abs_value": float(abs(z_star[0, idx].item())),
    } for idx in torch.argsort(z_star[0].abs(), descending=True).tolist()
        if abs(float(z_star[0, idx].item())) > args.nonzero_eps]
    selected_active_summary = [{
        "index": int(idx),
        "value": float(z_star[0, idx].item()),
        "abs_value": float(abs(z_star[0, idx].item())),
    } for idx in active_indices.tolist()]
    with open(out_root / "active_indices.json", "w") as f:
        json.dump({
            "nonzero_eps": args.nonzero_eps,
            "all_active_count": len(all_active),
            "selected_active_count": len(active_indices),
            "all_active": all_active_summary,
            "selected_active": selected_active_summary,
        }, f, indent=2)

    metadata = {
        "ckpt": str(ckpt_path),
        "template": args.template,
        "device": str(device),
        "use_ema": not args.no_ema,
        "steps": args.steps,
        "noise_mode": args.noise_mode,
        "noise_seed": noise_seed,
        "x_T_source": x_T_source,
        "split": args.split,
        "input_dir": args.input_dir,
        "index": args.index,
        "sample_name": name,
        "eta": eta,
        "seed": args.seed,
        "perturb_mode": args.perturb_mode,
        "deltas": deltas,
        "display_order": [deltas[idx] for idx in display_order],
        "nonzero_eps": args.nonzero_eps,
        "all_active_count": len(all_active),
        "selected_active_count": len(active_indices),
        "lambda_l1": conf.lambda_l1,
        "ista_steps": conf.ista_steps,
        "ista_solver": conf.ista_solver,
        "zd_stage2_use_zstar_cond": conf.zd_stage2_use_zstar_cond,
        "zd_cond_only": conf.zd_cond_only,
        "all_active_rows_path": (str(all_active_rows_path)
                                  if all_active_rows_path is not None else None),
        "perturbations": perturb_records,
    }
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved outputs to {out_root}")
    print(f"Noise source: {x_T_source}")
    print(f"Active z* count: {len(all_active)} (selected: {len(active_indices)})")
    print("Selected active indices:")
    print("  " + ", ".join(str(idx) for idx in active_indices.tolist()))


if __name__ == "__main__":
    main()

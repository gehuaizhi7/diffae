#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.utils import make_grid, save_image

from dataset import ImageDataset
from experiment import LitModel
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
            "Run full-step reconstructions for a Diff-AE checkpoint, with optional "
            "DDIM inversion, multi-noise comparisons, and denoising-progress dumps."
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
        help="Number of DDIM steps for reconstruction.",
    )
    parser.add_argument(
        "--indices",
        default="0,1,2,3",
        help="Comma-separated sample indices to reconstruct.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split for built-in datasets. For FFHQ, test is the first 10k images.",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Optional image folder. If set, images are loaded from here instead of the LMDB dataset.",
    )
    parser.add_argument(
        "--exts",
        default="jpg,jpeg,JPG,png,PNG",
        help="Comma-separated extensions for --input_dir.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for reconstruction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling x_T without inversion.",
    )
    parser.add_argument(
        "--invert-noise",
        action="store_true",
        help="Use DDIM inversion to recover x_T before reconstruction.",
    )
    parser.add_argument(
        "--random-noise",
        action="store_true",
        help=(
            "When not using --invert-noise, sample a fresh random x_T. "
            "By default the script reuses the checkpoint's saved x_T buffer to better match "
            "training-time logged reconstructions."
        ),
    )
    parser.add_argument(
        "--noise-seeds",
        default=None,
        help=(
            "Comma-separated random seeds. Reuses the same conditioning tensors for each input "
            "image, but regenerates x_T for every seed so you can compare different-noise reconstructions."
        ),
    )
    parser.add_argument(
        "--match-training-log",
        action="store_true",
        help=(
            "Mimic training-time logged reconstructions by forcing steps=T_eval and using the "
            "checkpoint's saved x_T buffer."
        ),
    )
    parser.add_argument(
        "--save-progress",
        action="store_true",
        help="Save intermediate denoising frames from DDIM sampling.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="When --save-progress is set, save one frame every N reverse steps.",
    )
    parser.add_argument(
        "--progress-kind",
        choices=["pred_xstart", "sample"],
        default="pred_xstart",
        help="Which DDIM progressive output to visualize.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save results. Defaults to recon_outputs/<exp>/T<steps>[_inv].",
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


def parse_int_list(text: str, flag_name: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError(f"{flag_name} must contain at least one integer.")
    return out


def parse_optional_int_list(text: Optional[str], flag_name: str) -> Optional[List[int]]:
    if text is None:
        return None
    return parse_int_list(text, flag_name)


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


def load_batch(conf, args, indices: List[int]) -> Tuple[torch.Tensor, List[str]]:
    if args.input_dir is not None:
        exts = [x.strip() for x in args.exts.split(",") if x.strip()]
        dataset = ImageDataset(args.input_dir,
                               image_size=conf.img_size,
                               exts=exts,
                               do_augment=False,
                               sort_names=True)
        imgs = []
        names = []
        for idx in indices:
            item = dataset[idx]
            imgs.append(item["img"])
            rel = dataset.paths[idx]
            names.append(Path(rel).stem)
        return torch.stack(imgs, dim=0), names

    dataset = conf.make_dataset(split=args.split, do_augment=False)
    imgs = []
    names = []
    for idx in indices:
        item = dataset[idx]
        imgs.append(item["img"])
        names.append(f"idx{idx:05d}")
    return torch.stack(imgs, dim=0), names


def run_progressive_dump(sampler,
                         model,
                         noise,
                         model_kwargs,
                         eta: float,
                         out_dir: Path,
                         nrow: int,
                         every: int,
                         kind: str):
    frames = []
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sampler.num_timesteps
    for step_idx, out in enumerate(
            sampler.ddim_sample_loop_progressive(model=model,
                                                 noise=noise,
                                                 model_kwargs=model_kwargs,
                                                 eta=eta,
                                                 progress=True)):
        reverse_t = total - step_idx - 1
        if step_idx % every != 0 and reverse_t != 0:
            continue
        frame = out[kind].detach().clamp(-1, 1)
        frame = ((frame + 1) / 2).cpu()
        save_path = out_dir / f"step_{step_idx:04d}_t{reverse_t:04d}_{kind}.png"
        save_image(frame, str(save_path), nrow=nrow)
        frames.append(frame)

    if frames:
        contact = make_grid(torch.cat(frames, dim=0), nrow=nrow)
        save_image(contact, str(out_dir / f"progress_contact_{kind}.png"))


def prepare_forward_noise(lit: LitModel,
                          imgs: torch.Tensor,
                          device: torch.device,
                          seed: int,
                          random_noise: bool) -> Tuple[torch.Tensor, str]:
    if random_noise:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return torch.randn_like(imgs), f"random(seed={seed})"

    base_noise = getattr(lit, "x_T", None)
    if base_noise is None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return torch.randn_like(imgs), (
            f"random(seed={seed}) because checkpoint does not contain a saved x_T buffer"
        )

    if base_noise.ndim != 4 or tuple(base_noise.shape[1:]) != tuple(imgs.shape[1:]):
        raise ValueError(
            f"Checkpoint x_T buffer has shape {tuple(base_noise.shape)}, "
            f"but recon batch expects (*, {tuple(imgs.shape[1:])})."
        )

    if len(imgs) <= len(base_noise):
        noise = base_noise[:len(imgs)]
        return noise.to(device), "checkpoint_x_T"

    repeats = (len(imgs) + len(base_noise) - 1) // len(base_noise)
    noise = base_noise.repeat(repeats, 1, 1, 1)[:len(imgs)]
    return noise.to(device), f"checkpoint_x_T_repeated({len(base_noise)}->{len(imgs)})"


def sample_random_noise_like(imgs: torch.Tensor,
                             device: torch.device,
                             seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return torch.randn_like(imgs)


def slice_model_kwargs(model_kwargs, start: int, end: int):
    out = {}
    for key, val in model_kwargs.items():
        if torch.is_tensor(val):
            out[key] = val[start:end]
        else:
            out[key] = val
    return out


def slugify(text: str) -> str:
    slug = []
    for ch in text:
        if ch.isalnum() or ch in ('-', '_'):
            slug.append(ch)
        else:
            slug.append('_')
    return ''.join(slug)


def save_single_run_outputs(out_root: Path,
                            names: List[str],
                            orig_vis: torch.Tensor,
                            recon_vis: torch.Tensor,
                            diff_vis: torch.Tensor,
                            saved_files: List[Path]):
    for i, name in enumerate(names):
        panel = torch.stack([orig_vis[i], recon_vis[i], diff_vis[i]], dim=0)
        panel_path = out_root / f"{name}_orig_recon_diff.png"
        save_image(panel, str(panel_path), nrow=3)
        saved_files.append(panel_path)

    nrow = len(names)
    grid = make_grid(torch.cat([orig_vis, recon_vis, diff_vis], dim=0), nrow=nrow)
    grid_path = out_root / "grid.png"
    originals_path = out_root / "originals.png"
    recons_path = out_root / "recons.png"
    abs_diff_path = out_root / "abs_diff.png"
    save_image(grid, str(grid_path))
    save_image(orig_vis, str(originals_path), nrow=nrow)
    save_image(recon_vis, str(recons_path), nrow=nrow)
    save_image(diff_vis, str(abs_diff_path), nrow=nrow)
    saved_files.extend([grid_path, originals_path, recons_path, abs_diff_path])


def save_multi_noise_outputs(out_root: Path,
                             names: List[str],
                             orig_vis: torch.Tensor,
                             noise_runs,
                             saved_files: List[Path]):
    originals_path = out_root / "originals.png"
    save_image(orig_vis, str(originals_path), nrow=len(names))
    saved_files.append(originals_path)

    compare_rows = []
    recon_rows = []
    diff_rows = []
    compare_cols = 1 + 2 * len(noise_runs)
    recon_cols = 1 + len(noise_runs)

    for i, name in enumerate(names):
        row = [orig_vis[i]]
        recon_row = [orig_vis[i]]
        diff_row = [orig_vis[i]]
        for run in noise_runs:
            row.extend([run["recon_vis"][i], run["diff_vis"][i]])
            recon_row.append(run["recon_vis"][i])
            diff_row.append(run["diff_vis"][i])
        panel = torch.stack(row, dim=0)
        panel_path = out_root / f"{name}_noise_compare.png"
        save_image(panel, str(panel_path), nrow=compare_cols)
        saved_files.append(panel_path)
        compare_rows.extend(row)
        recon_rows.extend(recon_row)
        diff_rows.extend(diff_row)

    compare_grid_path = out_root / "noise_compare_grid.png"
    recon_compare_path = out_root / "noise_compare_recons.png"
    diff_compare_path = out_root / "noise_compare_diffs.png"
    save_image(make_grid(torch.stack(compare_rows, dim=0), nrow=compare_cols),
               str(compare_grid_path))
    save_image(make_grid(torch.stack(recon_rows, dim=0), nrow=recon_cols),
               str(recon_compare_path))
    save_image(make_grid(torch.stack(diff_rows, dim=0), nrow=recon_cols),
               str(diff_compare_path))
    saved_files.extend([compare_grid_path, recon_compare_path, diff_compare_path])

    for run in noise_runs:
        label = slugify(run["label"])
        grid = make_grid(torch.cat([orig_vis, run["recon_vis"], run["diff_vis"]], dim=0),
                         nrow=len(names))
        grid_path = out_root / f"grid_{label}.png"
        recons_path = out_root / f"recons_{label}.png"
        diff_path = out_root / f"abs_diff_{label}.png"
        save_image(grid, str(grid_path))
        save_image(run["recon_vis"], str(recons_path), nrow=len(names))
        save_image(run["diff_vis"], str(diff_path), nrow=len(names))
        saved_files.extend([grid_path, recons_path, diff_path])


def main():
    args = parse_args()
    device = choose_device(args)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    noise_seeds = parse_optional_int_list(args.noise_seeds, "--noise-seeds")
    if noise_seeds is not None:
        if args.invert_noise:
            raise ValueError("--noise-seeds cannot be used together with --invert-noise.")
        if args.match_training_log:
            raise ValueError("--noise-seeds cannot be used together with --match-training-log.")
        args.random_noise = True

    conf, state = restore_conf(args.template, ckpt_path)
    if args.match_training_log:
        args.steps = conf.T_eval
        args.random_noise = False
    lit = LitModel(conf)
    missing, unexpected = lit.load_state_dict(state["state_dict"], strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model = lit.model if args.no_ema else lit.ema_model
    lit.to(device)
    model.eval()

    indices = parse_int_list(args.indices, "--indices")
    imgs, names = load_batch(conf, args, indices)
    imgs = imgs.to(device)
    sampler = conf._make_diffusion_conf(T=args.steps).make_sampler()
    eta = conf.ddim_eta

    exp_name = conf.name or ckpt_path.parent.name
    suffix = f"T{args.steps}"
    if args.invert_noise:
        suffix += "_inv"
    if noise_seeds is not None:
        suffix += f"_multinoise{len(noise_seeds)}"
    out_root = Path(args.output_dir) if args.output_dir else Path("recon_outputs") / exp_name / suffix
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model_kwargs = build_condition_model_kwargs(conf, model, x_start=imgs)
        noise_runs = []
        if args.invert_noise:
            print("Running DDIM inversion ...")
            x_T = sampler.ddim_reverse_sample_loop(model=model,
                                                   x=imgs,
                                                   clip_denoised=True,
                                                   model_kwargs=model_kwargs,
                                                   eta=eta)["sample"]
            noise_runs.append({
                "label": "inverted_noise",
                "source": "ddim_inversion",
                "x_T": x_T,
            })
        elif noise_seeds is not None:
            print(f"Using fixed conditioning and random x_T from seeds: {noise_seeds}")
            for seed in noise_seeds:
                noise_runs.append({
                    "label": f"seed{seed}",
                    "source": f"random(seed={seed})",
                    "x_T": sample_random_noise_like(imgs, device, seed),
                })
        else:
            x_T, noise_source = prepare_forward_noise(lit=lit,
                                                     imgs=imgs,
                                                     device=device,
                                                     seed=args.seed,
                                                     random_noise=args.random_noise)
            noise_runs.append({
                "label": "default",
                "source": noise_source,
                "x_T": x_T,
            })

        if args.match_training_log:
            print(
                f"Matching training log settings: steps={args.steps} (T_eval), noise={noise_runs[0]['source']}."
            )
        else:
            if args.steps != conf.T_eval:
                print(
                    f"Note: training-time logged recon uses steps={conf.T_eval}; this run uses steps={args.steps}."
                )
            if noise_seeds is None and not args.invert_noise and noise_runs[0]["source"] != "checkpoint_x_T":
                print("Note: training-time logged recon uses the checkpoint's saved x_T buffer.")

        for run in noise_runs:
            print(f"Running reconstruction [{run['label']}] ...")
            recon = []
            x_T = run["x_T"]
            for start in range(0, len(imgs), args.batch_size):
                end = start + args.batch_size
                batch_x_T = x_T[start:end]
                batch_model_kwargs = slice_model_kwargs(model_kwargs, start, end)
                batch_recon = sampler.sample(model=model,
                                             noise=batch_x_T,
                                             model_kwargs=batch_model_kwargs,
                                             eta=eta)
                recon.append(batch_recon)
            run["recon"] = torch.cat(recon, dim=0)

    orig_vis = ((imgs.detach().clamp(-1, 1) + 1) / 2).cpu()
    for run in noise_runs:
        run["recon_vis"] = ((run["recon"].detach().clamp(-1, 1) + 1) / 2).cpu()
        run["diff_vis"] = (run["recon_vis"] - orig_vis).abs()

    saved_files: List[Path] = []
    if len(noise_runs) == 1:
        save_single_run_outputs(out_root=out_root,
                                names=names,
                                orig_vis=orig_vis,
                                recon_vis=noise_runs[0]["recon_vis"],
                                diff_vis=noise_runs[0]["diff_vis"],
                                saved_files=saved_files)
    else:
        save_multi_noise_outputs(out_root=out_root,
                                 names=names,
                                 orig_vis=orig_vis,
                                 noise_runs=noise_runs,
                                 saved_files=saved_files)

    metadata = {
        "ckpt": str(ckpt_path),
        "template": args.template,
        "device": str(device),
        "use_ema": not args.no_ema,
        "steps": args.steps,
        "indices": indices,
        "split": args.split,
        "input_dir": args.input_dir,
        "invert_noise": args.invert_noise,
        "save_progress": args.save_progress,
        "save_every": args.save_every,
        "progress_kind": args.progress_kind,
        "seed": args.seed,
        "noise_seeds": noise_seeds,
        "eta": eta,
        "exp_name": exp_name,
        "training_log_steps": conf.T_eval,
        "noise_runs": [
            {
                "label": run["label"],
                "source": run["source"],
            }
            for run in noise_runs
        ],
        "same_condition_across_runs": len(noise_runs) > 1,
    }
    if len(noise_runs) == 1:
        metadata["noise_source"] = noise_runs[0]["source"]
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if args.save_progress:
        max_progress_batch = min(args.batch_size, len(imgs))
        progress_kwargs = slice_model_kwargs(model_kwargs, 0, max_progress_batch)
        progress_dirs = []
        for run in noise_runs:
            progress_dir = out_root / "progress"
            if len(noise_runs) > 1:
                progress_dir = progress_dir / slugify(run["label"])
            progress_noise = run["x_T"][:max_progress_batch]
            run_progressive_dump(sampler=sampler,
                                 model=model,
                                 noise=progress_noise,
                                 model_kwargs=progress_kwargs,
                                 eta=eta,
                                 out_dir=progress_dir,
                                 nrow=max_progress_batch,
                                 every=max(1, args.save_every),
                                 kind=args.progress_kind)
            progress_dirs.append(progress_dir)
        for progress_dir in progress_dirs:
            print(f"Saved DDIM progress frames to {progress_dir}")
    else:
        print("Saved only final recon images. Add --save-progress to dump intermediate DDIM steps.")

    print(f"Saved reconstructions to {out_root}")
    print("Files:")
    for path in saved_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()

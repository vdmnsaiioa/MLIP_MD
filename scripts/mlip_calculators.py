#!/usr/bin/env python
"""
mlip_calculators.py
===================
Machine-learning potentials for the LCO | LPO scripts, chosen by name with --model. Used by
relax_lco_lpo_surfaces.py, relax_interface_scan.py, md_interface.py and mp_to_interf_relax.py;
keep it in the same directory as them.

Models (MODELS below)
  mace-mh    MACE-MH-1 (mace-torch), omat_pbe head: PBE/PBE+U, the level of the MP cells
  uma-omat   UMA-S 1.2.1 (fairchem-core, uma-s-1p2p1), omat task: OMat24, PBE/PBE+U
To add a model, add an entry to MODELS: backend (mace or uma), checkpoint (MACE file or URL, UMA
model name or file), head (MACE head or UMA task), tag (prefix of the default output folders) and
a one-line description. --checkpoint / --head override an entry for a single run.

mace-torch pins e3nn 0.4.4 while fairchem-core needs e3nn >= 0.5 and torch 2.13, so the two
backends need separate environments; a script imports only the package of the chosen model.

UMA notes
  - The checkpoints are gated on Hugging Face: request access at huggingface.co/facebook/UMA, then
    log in once with a read token (`hf auth login`, or set HF_TOKEN). The first run downloads the
    checkpoint to ~/.cache/fairchem (FAIRCHEM_CACHE_DIR to change it); download it once on a login
    node and set HF_HUB_OFFLINE=1 in jobs on nodes without internet.
  - --uma-settings is the fairchem inference preset:
      batch            full FP32 for any mix of structures; default for relaxations and energies
      high-precision   full FP32, merged and compiled model for one fixed composition
      default (nitro)  fastest, reduced precision (TF32); default for MD. UMA Nitro runs with
                       uma-s-1p2p1 on NVIDIA Ampere or newer, otherwise fairchem falls back to multigpu
      multigpu         merged and compiled model, TF32
  - --dtype only applies to MACE; UMA runs in FP32. D3 (--dispersion) is only wired up for MACE.

In the json files of the scripts, "model" is the checkpoint (as in older versions) and
"model_name" the entry of MODELS.
"""
from __future__ import annotations

import argparse

MACE_MH_1 = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"
MODELS = {
    "mace-mh": dict(backend="mace", checkpoint=MACE_MH_1, head="omat_pbe", tag="mh1",
                    description="MACE-MH-1, omat_pbe head (PBE/PBE+U)"),
    "uma-omat": dict(backend="uma", checkpoint="uma-s-1p2p1", head="omat", tag="uma",
                     description="UMA-S 1.2.1, omat task (PBE/PBE+U)"),
    # more models, e.g.
    # "uma-m-omat": dict(backend="uma", checkpoint="uma-m-1p1", head="omat", tag="umam",
    #                    description="UMA-M 1.1, omat task (PBE/PBE+U)"),
}
LEGACY_BACKEND = {"mace": "mace-mh", "uma": "uma-omat"}     # --backend of the previous version
UMA_SETTINGS = ("batch", "high-precision", "default", "nitro", "multigpu")
UMA_FULL_PRECISION = ("batch", "high-precision")
HF_HINT = ("The UMA checkpoints are gated: request access at https://huggingface.co/facebook/UMA, log in once "
           "with a read token (hf auth login, or export HF_TOKEN), and download the model on a node with internet "
           "before running with HF_HUB_OFFLINE=1.")


def add_model_args(p: argparse.ArgumentParser, dtype_default: str = "float64",
                   uma_settings_default: str = "batch", cueq: bool = False):
    """Model options shared by the scripts; call resolve_model_args() after parsing."""
    g = p.add_argument_group("model")
    g.add_argument("--model", choices=list(MODELS), default="mace-mh",
                   help="potential: " + "; ".join(f"{k} = {v['description']}" for k, v in MODELS.items()))
    g.add_argument("--checkpoint", default=None,
                   help="override the checkpoint of --model (MACE file or URL, UMA model name or file)")
    g.add_argument("--head", default=None, help="override the head (MACE) or task (UMA) of --model")
    g.add_argument("--backend", choices=list(LEGACY_BACKEND), default=None,
                   help=argparse.SUPPRESS)         # previous option: mace = mace-mh, uma = uma-omat
    g.add_argument("--dispersion", action="store_true", help="add D3 dispersion (MACE only; off = consistent with MP)")
    g.add_argument("--device", default="auto", help="cuda, cpu or auto")
    g.add_argument("--dtype", choices=["float32", "float64"], default=dtype_default,
                   help="MACE precision (UMA always runs in FP32)")
    g.add_argument("--uma-settings", choices=UMA_SETTINGS, default=uma_settings_default,
                   help="UMA inference preset (see mlip_calculators.py)")
    if cueq:
        g.add_argument("--cueq", action="store_true", help="MACE: use cuEquivariance kernels (must be installed)")
    return g


def resolve_model_args(args: argparse.Namespace) -> argparse.Namespace:
    """Set args.backend, args.checkpoint, args.head and args.tag from the --model entry."""
    if args.backend is not None:
        args.model = LEGACY_BACKEND[args.backend]
    entry = MODELS[args.model]
    args.backend = entry["backend"]
    args.checkpoint = args.checkpoint or entry["checkpoint"]
    args.head = args.head or entry["head"]
    args.tag = entry["tag"]
    if args.backend == "uma" and args.dispersion:
        raise SystemExit("--dispersion is only implemented for MACE models")
    return args


def model_label(args: argparse.Namespace) -> str:
    """Short name used in file names and table columns: MACE or UMA."""
    return "MACE" if args.backend == "mace" else "UMA"


def precision(args: argparse.Namespace, precise: bool = False) -> str:
    """MACE dtype or UMA inference preset; precise=True asks for full precision (float64 for MACE,
    an FP32 preset for UMA)."""
    if args.backend == "mace":
        return "float64" if precise else args.dtype
    return "batch" if precise and args.uma_settings not in UMA_FULL_PRECISION else args.uma_settings


def needs_precise_copy(args: argparse.Namespace) -> bool:
    """True if the main calculator runs at reduced precision, so a relaxation needs a second one."""
    return precision(args, True) != precision(args, False)


def describe(args: argparse.Namespace, precise: bool = False) -> str:
    if args.backend == "mace":
        return (f"{args.model}: MACE {args.checkpoint.split('/')[-1]}, head {args.head}, {precision(args, precise)}, "
                f"D3 {'on' if args.dispersion else 'off'}" + (", cuEquivariance" if getattr(args, "cueq", False) else ""))
    return f"{args.model}: UMA {args.checkpoint}, task {args.head}, inference settings {precision(args, precise)}"


def model_info(args: argparse.Namespace) -> dict:
    """Model settings for the json files of the scripts ("model" = checkpoint)."""
    info = dict(model_name=args.model, backend=args.backend, model=args.checkpoint, head=args.head)
    if args.backend == "mace":
        info.update(dispersion=bool(args.dispersion), dtype=args.dtype)
    else:
        info.update(uma_settings=args.uma_settings)
    return info


def args_record(args: argparse.Namespace) -> dict:
    """vars(args) for run_args.json, with "model" = checkpoint as in the other json files."""
    return dict(vars(args), model=args.checkpoint, model_name=args.model)


def resolve_device(args: argparse.Namespace) -> str:
    if args.device != "auto":
        return args.device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    print("  ! no CUDA device visible - running on the CPU (slow); check the GPU request and the torch build")
    return "cpu"


def get_calculator(args: argparse.Namespace, precise: bool = False):
    """ASE calculator of the chosen model (see precision() for `precise`)."""
    device = resolve_device(args)
    print(f"loading {describe(args, precise)} on {device}")
    if args.backend == "mace":
        from mace.calculators import mace_mp

        extra = {"enable_cueq": True} if getattr(args, "cueq", False) else {}
        return mace_mp(model=args.checkpoint, head=args.head, dispersion=args.dispersion,
                       default_dtype=precision(args, precise), device=device, **extra)

    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    try:
        predictor = pretrained_mlip.get_predict_unit(args.checkpoint, inference_settings=precision(args, precise),
                                                     device=device)
    except Exception as exc:                    # most likely Hugging Face access; other errors pass through
        name = type(exc).__name__
        if any(k in name for k in ("Gated", "RepositoryNotFound", "EntryNotFound", "HfHub", "Offline", "HTTP")):
            raise SystemExit(f"could not get the UMA checkpoint '{args.checkpoint}' ({name}: {exc})\n{HF_HINT}") from exc
        raise
    return FAIRChemCalculator(predictor, task_name=args.head)

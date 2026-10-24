#!/usr/bin/env python
"""
mlip_calculators.py
===================
ASE calculators for the LCO | LPO scripts: MACE-MH-1 (mace-torch) or UMA (fairchem-core), chosen
with --backend. Imported by relax_lco_lpo_surfaces.py, relax_interface_scan.py and md_interface.py;
keep it in the same directory as them.

  --backend mace   MACE-MH-1, head omat_pbe (PBE/PBE+U, the level of the MP cells) by default
  --backend uma    UMA (Meta FAIR), model uma-s-1p2p1 and task omat by default. The omat task is
                   the OMat24 inorganic-materials head (PBE/PBE+U).

mace-torch pins e3nn 0.4.4 while fairchem-core needs e3nn >= 0.5 and torch 2.13, so the two packages
need separate environments; each script imports only the package of the chosen backend.

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
"""
from __future__ import annotations

import argparse

MACE_MH_1 = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"
UMA_DEFAULT = "uma-s-1p2p1"
DEFAULTS = {
    "mace": dict(model=MACE_MH_1, head="omat_pbe"),
    "uma": dict(model=UMA_DEFAULT, head="omat"),
}
UMA_SETTINGS = ("batch", "high-precision", "default", "nitro", "multigpu")
UMA_FULL_PRECISION = ("batch", "high-precision")
HF_HINT = ("The UMA checkpoints are gated: request access at https://huggingface.co/facebook/UMA, log in once "
           "with a read token (hf auth login, or export HF_TOKEN), and download the model on a node with internet "
           "before running with HF_HUB_OFFLINE=1.")


def add_model_args(p: argparse.ArgumentParser, dtype_default: str = "float64",
                   uma_settings_default: str = "batch", cueq: bool = False):
    """Model options shared by the scripts; call resolve_model_args() after parsing."""
    g = p.add_argument_group("model")
    g.add_argument("--backend", choices=list(DEFAULTS), default="mace",
                   help="MACE-MH-1 (mace-torch) or UMA (fairchem-core)")
    g.add_argument("--model", default=None,
                   help=f"MACE: checkpoint URL or path (default MACE-MH-1). UMA: model name or checkpoint path "
                        f"(default {UMA_DEFAULT}; uma-m-1p1 is larger and slower)")
    g.add_argument("--head", default=None,
                   help="MACE head (default omat_pbe; matpes_r2scan for r2SCAN) or UMA task (default omat)")
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
    """Fill --model / --head with the defaults of the chosen backend."""
    for key, value in DEFAULTS[args.backend].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.backend == "uma" and args.dispersion:
        raise SystemExit("--dispersion is only implemented for --backend mace")
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
        return (f"MACE {args.model.split('/')[-1]}, head {args.head}, {precision(args, precise)}, "
                f"D3 {'on' if args.dispersion else 'off'}" + (", cuEquivariance" if getattr(args, "cueq", False) else ""))
    return f"UMA {args.model}, task {args.head}, inference settings {precision(args, precise)}"


def model_info(args: argparse.Namespace) -> dict:
    """Model settings for the json files of the scripts."""
    info = dict(backend=args.backend, model=args.model, head=args.head)
    if args.backend == "mace":
        info.update(dispersion=bool(args.dispersion), dtype=args.dtype)
    else:
        info.update(uma_settings=args.uma_settings)
    return info


def resolve_device(args: argparse.Namespace) -> str:
    if args.device != "auto":
        return args.device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    print("  ! no CUDA device visible - running on the CPU (slow); check the GPU request and the torch build")
    return "cpu"


def get_calculator(args: argparse.Namespace, precise: bool = False):
    """ASE calculator of the chosen backend (see precision() for `precise`)."""
    device = resolve_device(args)
    print(f"loading {describe(args, precise)} on {device}")
    if args.backend == "mace":
        from mace.calculators import mace_mp

        extra = {"enable_cueq": True} if getattr(args, "cueq", False) else {}
        return mace_mp(model=args.model, head=args.head, dispersion=args.dispersion,
                       default_dtype=precision(args, precise), device=device, **extra)

    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    try:
        predictor = pretrained_mlip.get_predict_unit(args.model, inference_settings=precision(args, precise),
                                                     device=device)
    except Exception as exc:                    # most likely Hugging Face access; other errors pass through
        name = type(exc).__name__
        if any(k in name for k in ("Gated", "RepositoryNotFound", "EntryNotFound", "HfHub", "Offline", "HTTP")):
            raise SystemExit(f"could not get the UMA checkpoint '{args.model}' ({name}: {exc})\n{HF_HINT}") from exc
        raise
    return FAIRChemCalculator(predictor, task_name=args.head)

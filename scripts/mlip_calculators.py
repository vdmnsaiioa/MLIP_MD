#!/usr/bin/env python
"""
mlip_calculators.py
===================
Machine-learning potentials for the LCO | LPO scripts, chosen with --model (MODEL in the batch
scripts). Used by relax_lco_lpo_surfaces.py, relax_interface_scan.py, md_interface.py and
mp_to_interf_relax.py; keep it in the same directory as them.

--model is the pretrained model, either one of the short names (MODELS below)
  short name     model                           package (backend)               folders  conda env*
  mace-mh        MACE-MH-1, omat_pbe head        mace-torch (mace)               mh1_*    vasp_stuff
  uma-omat       UMA-S 1.2.1, omat task          fairchem-core 2.x (uma)         uma_*    uma
  esen-omat      eSEN-30M-OMAT24                 fairchem-core 1.10 (fairchem1)  esen_*   fairchem1
  eqv2-omat      EquiformerV2-86M-OMAT24         fairchem-core 1.10 (fairchem1)  eqv2_*   fairchem1
  sevennet-ompa  SevenNet-MF-ompa, omat24 modal  sevenn (sevenn)                 7net_*   sevenn
  grace-omat     GRACE-2L-OMAT                   tensorpotential (grace)         grace_*  grace
or the name under which one of these packages publishes a pretrained model (pretrained_model()):
  UMA            uma-s-1p2p1, uma-s-1p1, uma-m-1p1, ... (task omat)
  fairchem v1    eSEN-30M-OMAT24, eSEN-30M-OAM, EquiformerV2-31M-OMAT24, EquiformerV2-86M-OMAT24,
                 EquiformerV2-153M-OMAT24 (and the -MP-sAlex fine-tunes of the EquiformerV2 models)
  SevenNet       7net-mf-ompa, 7net-omni, 7net-omni-i8, 7net-omni-i12 (modal omat24), 7net-omat, 7net-0, ...
  GRACE          GRACE-2L-OMAT, GRACE-2L-OMAT-medium-base, GRACE-2L-SMAX-OMAT-medium, ... (grace_models list)
  MACE           mace-mh-1
A published name that is the model of a short name runs as that short name (uma-s-1p2p1 = uma-omat,
same folders); any other gets folders named after it (EquiformerV2-31M-OMAT24_surfaces, ...).
*The batch scripts pick the conda environment from MODEL; the packages pin incompatible versions of
torch, e3nn, numpy or TensorFlow, so each backend needs its own environment, and a script imports only
the package of the chosen model.

The short-name models are trained on OMat24-level data (PBE/PBE+U); check what any other model is
trained on (e.g. -OAM, -MP-sAlex and the SevenNet mpa modal add MPtrj/sAlex, GRACE SMAX models a
generated dataset, 7net-0 is MPtrj only). Absolute energies differ between models, so only compare
energy differences computed with one model. The EquiformerV2 models predict forces and stress
directly (not as energy derivatives), so their MD does not conserve energy (the thermostat
compensates).

--checkpoint (a local file, or another name the package knows) and --head (MACE head, UMA task or
SevenNet modal) override the model for a single run, e.g. --model sevennet-ompa --head mpa. To add a
short name, add an entry to MODELS: backend, checkpoint, head (None if the model has none), tag (prefix
of the default output folders), label (file names and table columns) and a one-line description.

Weights: MACE-MH-1 (GitHub), SevenNet (GitHub) and GRACE (Hugging Face, public) download on first
use. UMA (facebook/UMA) and the fairchem v1 models (facebook/OMAT24, manual approval) are gated on
Hugging Face: request access, `hf auth login`, download once on a login node and run jobs with
HF_HUB_OFFLINE=1. For UMA either fairchem's own download or `hf download facebook/UMA
checkpoints/<name>.pt` works (uma_checkpoint()); the fairchem v1 models are found where
`hf download facebook/OMAT24 <file>.pt` puts them.

Precision: --dtype (MACE) and --uma-settings (UMA: batch / high-precision = full FP32, default =
nitro, fastest, reduced precision) choose it; the other models run at their own precision
(fairchem v1 and SevenNet float32; GRACE as published, GRACE-2L-OMAT float64). D3 (--dispersion) is
only wired up for MACE; --cueq (cuEquivariance) for MACE and SevenNet.

Memory: SevenNet-MF-ompa (about 4 GB for 96 atoms on a CPU in a test) and EquiformerV2-86M/153M are
the heaviest; for interfaces of several hundred atoms use --cueq with SevenNet, EquiformerV2-31M-OMAT24
or a smaller MD supercell if the GPU runs out of memory.

In the json files of the scripts, "model" is the checkpoint (as in older versions) and
"model_name" the --model value (short name or published name).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

MACE_MH_1 = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"
MODELS = {
    "mace-mh": dict(backend="mace", checkpoint=MACE_MH_1, head="omat_pbe", tag="mh1", label="MACE",
                    description="MACE-MH-1, omat_pbe head (PBE/PBE+U)"),
    "uma-omat": dict(backend="uma", checkpoint="uma-s-1p2p1", head="omat", tag="uma", label="UMA",
                     description="UMA-S 1.2.1, omat task (PBE/PBE+U)"),
    "esen-omat": dict(backend="fairchem1", checkpoint="eSEN-30M-OMAT24", head=None, tag="esen", label="eSEN",
                      description="eSEN-30M-OMAT24 (fairchem v1)"),
    "eqv2-omat": dict(backend="fairchem1", checkpoint="EquiformerV2-86M-OMAT24", head=None, tag="eqv2", label="EqV2",
                      description="EquiformerV2-86M-OMAT24 (fairchem v1, direct forces)"),
    "sevennet-ompa": dict(backend="sevenn", checkpoint="7net-mf-ompa", head="omat24", tag="7net", label="7net",
                          description="SevenNet-MF-ompa, omat24 modal"),
    "grace-omat": dict(backend="grace", checkpoint="GRACE-2L-OMAT", head=None, tag="grace", label="GRACE",
                       description="GRACE-2L-OMAT (tensorpotential)"),
}
# published names of the models above (lower case): --model uma-s-1p2p1 runs as uma-omat
PRESET_ALIASES = {"mace-mh-1": "mace-mh", "uma-s-1p2p1": "uma-omat", "esen-30m-omat24": "esen-omat",
                  "equiformerv2-86m-omat24": "eqv2-omat", "7net-mf-ompa": "sevennet-ompa",
                  "sevennet-mf-ompa": "sevennet-ompa", "grace-2l-omat": "grace-omat"}
# fairchem v1 (fairchem-core 1.10) models for inorganic materials (facebook/OMAT24 on Hugging Face)
FAIRCHEM1_MODELS = ("eSEN-30M-OMAT24", "eSEN-30M-OAM", "EquiformerV2-31M-OMAT24", "EquiformerV2-86M-OMAT24",
                    "EquiformerV2-153M-OMAT24", "EquiformerV2-31M-OMAT24-MP-sAlex",
                    "EquiformerV2-86M-OMAT24-MP-sAlex", "EquiformerV2-153M-OMAT24-MP-sAlex")
SEVENN_MULTITASK = ("mf-ompa", "omni", "omni-i8", "omni-i12")    # SevenNet models with modals: omat24 used
PACKAGE = {"mace": "MACE", "uma": "fairchem 2", "fairchem1": "fairchem v1", "sevenn": "SevenNet", "grace": "GRACE"}
LEGACY_BACKEND = {"mace": "mace-mh", "uma": "uma-omat"}     # --backend of an earlier version
FIXED_PRECISION = {"fairchem1": "float32", "sevenn": "float32", "grace": "model precision"}
UMA_SETTINGS = ("batch", "high-precision", "default", "nitro", "multigpu")
UMA_FULL_PRECISION = ("batch", "high-precision")
HF_HINT = ("The checkpoint is gated on Hugging Face (UMA: facebook/UMA; eSEN and EquiformerV2 OMat24: "
           "facebook/OMAT24, approved by hand): request access there, log in once with a read token (hf auth "
           "login, or export HF_TOKEN), and download the model on a node with internet before running with "
           "HF_HUB_OFFLINE=1, e.g. hf download facebook/UMA checkpoints/uma-s-1p2p1.pt, or "
           "hf download facebook/OMAT24 esen_30m_omat.pt.")


def pretrained_model(name: str) -> tuple[str, str, str | None] | None:
    """For the published name of a pretrained model: (short name of MODELS with the same package, the name
    as the package spells it, head / task / modal or None); None if no package here knows such a name.
    Whether the model exists is checked by the package when it loads it."""
    low = name.lower()
    fairchem1 = {n.lower(): n for n in FAIRCHEM1_MODELS}
    if low in fairchem1:
        return ("esen-omat" if low.startswith("esen") else "eqv2-omat"), fairchem1[low], None
    if low.startswith("uma-"):
        return "uma-omat", low, "omat"
    m = re.fullmatch(r"(?:7net|sevennet)-(.+)", low)
    if m:
        return "sevennet-ompa", name, "omat24" if m.group(1) in SEVENN_MULTITASK else None
    if name.startswith(("GRACE-", "GRACE_")):
        return "grace-omat", name, None
    return None


def model_entry(name: str) -> dict:
    """Settings of --model `name`, a short name of MODELS or the published name of a pretrained model, with
    "model" = the name the scripts use for it (the short name if the model is one of MODELS)."""
    key = PRESET_ALIASES.get(name.lower(), name.lower())
    if key in MODELS:
        return dict(MODELS[key], model=key)
    found = pretrained_model(name)
    if found is None:
        raise SystemExit(f"unknown model '{name}': use a short name ({', '.join(MODELS)}) or the published name of "
                         f"a pretrained model, e.g. uma-m-1p1, EquiformerV2-31M-OMAT24, 7net-omni, "
                         f"GRACE-2L-OMAT-medium-base (see mlip_calculators.py)")
    short, name, head = found
    base = MODELS[short]
    detail = f", {'task' if base['backend'] == 'uma' else 'modal'} {head}" if head else ""
    return dict(base, model=name, checkpoint=name, head=head, tag=name,
                description=f"{name} ({PACKAGE[base['backend']]}{detail})")


def add_model_args(p: argparse.ArgumentParser, dtype_default: str = "float64",
                   uma_settings_default: str = "batch", cueq: bool = False):
    """Model options shared by the scripts; call resolve_model_args() after parsing."""
    g = p.add_argument_group("model")
    g.add_argument("--model", default="mace-mh",
                   help="pretrained model: a short name (" + "; ".join(f"{k} = {v['description']}"
                                                                     for k, v in MODELS.items())
                        + ") or a published name, e.g. EquiformerV2-31M-OMAT24 (see mlip_calculators.py)")
    g.add_argument("--checkpoint", default=None,
                   help="override the checkpoint of --model (file, or a name the package knows)")
    g.add_argument("--head", default=None, help="override the head (MACE), task (UMA) or modal (SevenNet)")
    g.add_argument("--backend", choices=list(LEGACY_BACKEND), default=None,
                   help=argparse.SUPPRESS)         # earlier option: mace = mace-mh, uma = uma-omat
    g.add_argument("--dispersion", action="store_true", help="add D3 dispersion (MACE only; off = consistent with MP)")
    g.add_argument("--device", default="auto", help="cuda, cpu or auto")
    g.add_argument("--dtype", choices=["float32", "float64"], default=dtype_default,
                   help="MACE precision (the other models use their own)")
    g.add_argument("--uma-settings", choices=UMA_SETTINGS, default=uma_settings_default,
                   help="UMA inference preset (see mlip_calculators.py)")
    if cueq:
        g.add_argument("--cueq", action="store_true",
                       help="MACE / SevenNet: use cuEquivariance kernels (must be installed)")
    return g


def resolve_model_args(args: argparse.Namespace) -> argparse.Namespace:
    """Set args.model (short name if there is one), args.backend, args.checkpoint, args.head, args.tag and
    args.label from --model."""
    if args.backend is not None:
        args.model = LEGACY_BACKEND[args.backend]
    entry = model_entry(args.model)
    args.model = entry["model"]
    args.backend = entry["backend"]
    args.checkpoint = args.checkpoint or entry["checkpoint"]
    args.head = args.head or entry["head"]
    args.tag, args.label = entry["tag"], entry["label"]
    if args.dispersion and args.backend != "mace":
        raise SystemExit("--dispersion is only implemented for MACE models")
    if getattr(args, "cueq", False) and args.backend not in ("mace", "sevenn"):
        raise SystemExit("--cueq is only available for MACE and SevenNet models")
    return args


def model_label(args: argparse.Namespace) -> str:
    """Short name used in file names and table columns (MACE, UMA, eSEN, EqV2, 7net, GRACE)."""
    return args.label


def precision(args: argparse.Namespace, precise: bool = False) -> str:
    """MACE dtype or UMA inference preset; precise=True asks for full precision (float64 for MACE,
    an FP32 preset for UMA). The other models have a fixed precision."""
    if args.backend == "mace":
        return "float64" if precise else args.dtype
    if args.backend == "uma":
        return "batch" if precise and args.uma_settings not in UMA_FULL_PRECISION else args.uma_settings
    return FIXED_PRECISION[args.backend]


def needs_precise_copy(args: argparse.Namespace) -> bool:
    """True if the main calculator runs at reduced precision, so a relaxation needs a second one."""
    return precision(args, True) != precision(args, False)


def describe(args: argparse.Namespace, precise: bool = False) -> str:
    cueq = ", cuEquivariance" if getattr(args, "cueq", False) else ""
    name = "" if args.model == args.checkpoint else f"{args.model}: "
    if args.backend == "mace":
        return (f"{name}MACE {args.checkpoint.split('/')[-1]}, head {args.head}, {precision(args, precise)}, "
                f"D3 {'on' if args.dispersion else 'off'}{cueq}")
    if args.backend == "uma":
        return f"{name}UMA {args.checkpoint}, task {args.head}, inference settings {precision(args, precise)}"
    modal = f", modal {args.head}" if args.head else ""
    return f"{name}{PACKAGE[args.backend]} {args.checkpoint}{modal}, {precision(args, precise)}{cueq}"


def model_info(args: argparse.Namespace) -> dict:
    """Model settings for the json files of the scripts ("model" = checkpoint)."""
    info = dict(model_name=args.model, backend=args.backend, model=args.checkpoint, head=args.head)
    if args.backend == "mace":
        info.update(dispersion=bool(args.dispersion), dtype=args.dtype)
    elif args.backend == "uma":
        info.update(uma_settings=args.uma_settings)
    return info


def args_record(args: argparse.Namespace) -> dict:
    """vars(args) for run_args.json, with "model" = checkpoint as in the other json files."""
    return dict(vars(args), model=args.checkpoint, model_name=args.model)


def resolve_device(args: argparse.Namespace) -> str:
    if args.backend == "grace":                     # TensorFlow: no torch in this environment
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            return "cpu"
        import tensorflow as tf

        if tf.config.list_physical_devices("GPU"):
            return "cuda"
    else:
        if args.device != "auto":
            return args.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
    print("  ! no GPU visible - running on the CPU (slow); check the GPU request and the CUDA build of the package")
    return "cpu"


def uma_checkpoint(name: str) -> str:
    """What fairchem 2 should load for the UMA model `name`: the name itself if fairchem can get the model
    (its cache, ~/.cache/fairchem or $FAIRCHEM_CACHE_DIR, or a download when online), otherwise a copy of
    the checkpoint file already on disk, e.g. from `hf download facebook/UMA checkpoints/<name>.pt`, which
    goes to the Hugging Face cache, where fairchem does not look (jobs run with HF_HUB_OFFLINE=1). A file
    loads without fairchem's reference energies, which only matter for isolated single atoms and
    formation energies, so the energies, forces and stresses of the cells here are the same."""
    from fairchem.core.calculate import pretrained_mlip as pm

    resolve = getattr(pm, "resolve_checkpoint", None)        # in recent fairchem-core (2.24 has it)
    if resolve is None or Path(name).is_file() or name not in getattr(pm, "available_models", ()):
        return name
    try:
        resolve(name)
        return name
    except Exception as exc:
        from huggingface_hub import try_to_load_from_cache

        entry = getattr(pm, "_MODEL_CKPTS", None)
        entry = entry.checkpoints.get(name) if entry is not None else None
        if entry is not None:
            repo, revision = entry.repo_id, entry.revision
            filename = f"{entry.subfolder}/{entry.filename}" if entry.subfolder else entry.filename
        else:
            repo, revision, filename = "facebook/UMA", None, f"checkpoints/{name}.pt"
        for cache in (None, getattr(pm, "CACHE_DIR", None)):  # Hugging Face cache, then fairchem's
            path = try_to_load_from_cache(repo, filename, cache_dir=cache, revision=revision)
            if isinstance(path, str):
                print(f"  {name}: fairchem cannot get it ({type(exc).__name__}); using the copy {path}")
                return path
        raise


def get_calculator(args: argparse.Namespace, precise: bool = False):
    """ASE calculator of the chosen model (see precision() for `precise`)."""
    device = resolve_device(args)
    print(f"loading {describe(args, precise)} on {device}")
    try:
        return _load(args, device, precise)
    except Exception as exc:                    # gated Hugging Face downloads; other errors pass through
        name = type(exc).__name__
        if any(k in name for k in ("Gated", "RepositoryNotFound", "EntryNotFound", "HfHub", "Offline", "HTTP")):
            raise SystemExit(f"could not get the checkpoint '{args.checkpoint}' ({name}: {exc})\n{HF_HINT}") from exc
        raise


def _load(args: argparse.Namespace, device: str, precise: bool):
    if args.backend == "mace":
        from mace.calculators import mace_mp

        extra = {"enable_cueq": True} if getattr(args, "cueq", False) else {}
        return mace_mp(model=args.checkpoint, head=args.head, dispersion=args.dispersion,
                       default_dtype=precision(args, precise), device=device, **extra)

    if args.backend == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit(uma_checkpoint(args.checkpoint),
                                                     inference_settings=precision(args, precise), device=device)
        return FAIRChemCalculator(predictor, task_name=args.head)

    if args.backend == "fairchem1":
        from fairchem.core import OCPCalculator

        if Path(args.checkpoint).is_file():
            return OCPCalculator(checkpoint_path=args.checkpoint, cpu=device == "cpu", seed=0)
        return OCPCalculator(model_name=args.checkpoint, local_cache=str(Path.home() / ".cache" / "fairchem_v1"),
                             cpu=device == "cpu", seed=0)

    if args.backend == "sevenn":
        from sevenn.calculator import SevenNetCalculator

        extra = {"enable_cueq": True} if getattr(args, "cueq", False) else {}
        return SevenNetCalculator(args.checkpoint, modal=args.head, device=device, **extra)

    if args.backend == "grace":
        from tensorpotential.calculator import TPCalculator, grace_fm

        if Path(args.checkpoint).is_dir():          # a saved model folder
            return TPCalculator(model=args.checkpoint)
        return grace_fm(args.checkpoint)

    raise ValueError(f"unknown backend {args.backend}")

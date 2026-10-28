#!/usr/bin/env python
"""
mp_to_interf_relax.py
=====================
From Materials Project bulk cells to relaxed LCO | LPO interfaces with one machine-learning
potential (--model), as four steps run one after the other:

  1  relax_lco_lpo_surfaces.py   bulk LCO and LPO from MP, atoms and cell relaxed; all slabs of the
                                 chosen facets, atoms and in-plane lattice relaxed  -> <run>_surfaces
  2  match_relaxed_surfaces.py   lattice matching of the relaxed slabs (no model)   -> <run>_interfaces
  3  scan_interface_registry.py  registry (lateral shift) x gap structures of one interface (no model)
                                 -> <run>_interfaces/<interface>/registry_scan
  4  relax_interface_scan.py     single points and relaxations of these structures, work of adhesion
                                 -> <run>_interfaces/<interface>/registry_scan/results.csv
Molecular dynamics is separate: md_interface.py (batch script md_interface.sh).

--early-stop None (default) runs all four steps; 1, 2 or 3 stops after that step. Every step
resumes: work finished with the same model and settings is skipped, so a run can be repeated with a
larger --early-stop, or resubmitted after the wall-time limit. The folders of a run belong to one
model and one set of slab and interface settings: the steps refuse to mix them, so give a run with
other settings its own --run-name.

Model (--model): the pretrained model, as a short name, mace-mh (MACE-MH-1), uma-omat (UMA-S 1.2.1,
omat task), esen-omat (eSEN-30M-OMAT24), eqv2-omat (EquiformerV2-86M-OMAT24), sevennet-ompa
(SevenNet-MF-ompa, omat24 modal) or grace-omat (GRACE-2L-OMAT), or as the name its package publishes it
under, e.g. uma-m-1p1, EquiformerV2-31M-OMAT24, 7net-omni or GRACE-2L-OMAT-medium-base (see
mlip_calculators.py). --checkpoint (a local file) / --head (head, task or modal) change the model of a
run; give such a run its own --run-name.

Interface for steps 3 and 4 (--interface)
  Default: the interface with the smallest lattice mismatch (largest principal strain of LPO in
  interfaces.csv), the fewest atoms on ties. Or a folder name such as
  001_LCO_113_s1-bot__LPO_110_s2-bot: it is matched on the slabs and faces, so the number and the atom
  count (_432at), which can differ between models and runs, may be left out or differ.

Output names: <run> = --run-name, by default the folder tag of the model (mh1, uma, esen, eqv2,
7net, grace for the short names, mh1 and uma continuing the earlier folders; the name itself for other
models, e.g. EquiformerV2-31M-OMAT24). <run>_workflow.json records the parameters of the last run, the
interface it used and the time of each step.

Every parameter of the four scripts that matters is an option here (see --help); the batch script
mp_to_interf_relax.sh sets them all. The defaults are those of the MACE-MH-1 run.

Requirements: as the four scripts (ase, pymatgen, mp-api for MP queries, and mace-torch or
fairchem-core for the model). Run from scripts/ next to them. Untested on real data.

Example
  python mp_to_interf_relax.py --model uma-omat --early-stop 2
  python mp_to_interf_relax.py --model uma-omat --interface 001_LCO_113_s1-bot__LPO_110_s2-bot
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from mlip_calculators import MODELS, UMA_SETTINGS, model_entry

STEPS = {1: "bulk and slabs (relax_lco_lpo_surfaces.py)",
         2: "lattice matching (match_relaxed_surfaces.py)",
         3: "registry and gap scan (scan_interface_registry.py)",
         4: "interface relaxations (relax_interface_scan.py)"}


# =============================================================================
# Options
# =============================================================================
def optional(value: str) -> str | None:
    """'' / None / auto -> None, so a batch script can pass every option."""
    return None if value.strip().lower() in ("", "none", "auto") else value


def early_stop(value: str) -> int | None:
    v = value.strip().lower()
    if v in ("", "none", "0"):
        return None
    if v not in ("1", "2", "3"):
        raise argparse.ArgumentTypeError("None, 1, 2 or 3")
    return int(v)


def yes_no(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "yes", "1", "on"):
        return True
    if v in ("false", "no", "0", "off", "", "none"):
        return False
    raise argparse.ArgumentTypeError("true or false")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Steps 1-4 of the LCO | LPO workflow: MP bulk -> relaxed slabs -> interfaces -> "
                    "registry/gap scan -> relaxed interfaces (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("run and model")
    g.add_argument("--model", required=True,
                   help="pretrained model for all calculations: a short name (" + "; ".join(
                       f"{k} = {v['description']}" for k, v in MODELS.items()) + ") or a published name, e.g. "
                        "EquiformerV2-31M-OMAT24, uma-m-1p1, 7net-omni, GRACE-2L-OMAT-medium-base")
    g.add_argument("--run-name", type=optional, default=None,
                   help="prefix of the output folders (<run>_surfaces, <run>_interfaces); default: the model's tag")
    g.add_argument("--checkpoint", type=optional, default=None,
                   help="other checkpoint for the model's package, e.g. a local file (default: the model's own)")
    g.add_argument("--head", type=optional, default=None,
                   help="head / task / modal of the model, e.g. mpa for SevenNet (default: the model's own)")
    g.add_argument("--early-stop", type=early_stop, default=None, help="None = all steps; 1, 2 or 3 = stop after it")
    g.add_argument("--device", default="auto", help="cuda, cpu or auto")
    g.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="MACE precision")
    g.add_argument("--uma-settings", choices=UMA_SETTINGS, default="batch", help="UMA inference preset")
    g.add_argument("--cueq", type=yes_no, default=False,
                   help="cuEquivariance kernels for MACE / SevenNet (faster, less GPU memory; must be installed)")

    g = p.add_argument_group("step 1: bulk and slabs (relax_lco_lpo_surfaces.py)")
    g.add_argument("--lco-id", default="mp-22526", help="Materials Project id of LCO")
    g.add_argument("--lpo-id", default="mp-13725", help="Materials Project id of LPO")
    g.add_argument("--lco-file", type=optional, default=None,
                   help="local LCO bulk file instead of MP, e.g. mh1_surfaces/bulk/LCO_mp-22526_input_conv.vasp")
    g.add_argument("--lpo-file", type=optional, default=None, help="local LPO bulk file instead of MP")
    g.add_argument("--symprec", type=float, default=0.1, help="symmetry tolerance (A)")
    g.add_argument("--n-facets", type=int, default=8, help="facets per material (best Donnay-Harker ranked)")
    g.add_argument("--max-index", type=int, default=4, help="largest |h|,|k|,|l| of the facets")
    g.add_argument("--lco-millers", nargs="*", default=["auto"], help="LCO facets, e.g. 001 104 012; auto = ranked")
    g.add_argument("--lpo-millers", nargs="*", default=["auto"], help="LPO facets; auto = ranked")
    g.add_argument("--extra-lco-millers", nargs="*", default=[], help="LCO facets added to the list")
    g.add_argument("--extra-lpo-millers", nargs="*", default=[], help="LPO facets added to the list")
    g.add_argument("--lco-thickness", type=float, default=15.0, help="minimum LCO slab thickness (A)")
    g.add_argument("--lpo-thickness", type=float, default=15.0, help="minimum LPO slab thickness (A)")
    g.add_argument("--slab-vacuum", type=float, default=20.0, help="vacuum in the slab cells (A)")
    g.add_argument("--ftol", type=float, default=0.1, help="atoms closer than this along the normal form a layer (A)")
    g.add_argument("--bonds", nargs="+", default=["P-O:2.0"], help="bonds a slab must not cut (A-B:cutoff), or none")
    g.add_argument("--keep-broken", type=yes_no, default=False, help="also relax slabs that cut --bonds")
    g.add_argument("--bulk-fmax", type=float, default=0.005, help="bulk force/stress convergence (eV/A)")
    g.add_argument("--slab-fmax", type=float, default=0.02, help="slab force/stress convergence (eV/A)")
    g.add_argument("--slab-max-steps", type=int, default=2000, help="max optimizer steps per bulk/slab relaxation")
    g.add_argument("--slab-fix-bottom", type=float, default=0.0, help="freeze slab atoms this close to the bottom (A)")

    g = p.add_argument_group("step 2: lattice matching (match_relaxed_surfaces.py)")
    g.add_argument("--max-strain", type=float, default=0.01, help="largest principal strain of LPO (0.01 = 1 %%)")
    g.add_argument("--max-atoms", type=int, default=1000, help="largest interface (atoms)")
    g.add_argument("--matches-per-pair", type=int, default=2, help="cells per slab pair and orientation")
    g.add_argument("--separation", type=float, default=2.0, help="LCO-LPO gap of POSCAR_interface (A)")
    g.add_argument("--lco-vacuum", type=float, default=10.0, help="vacuum below LCO in the interface cell (A)")
    g.add_argument("--lpo-vacuum", type=float, default=10.0, help="vacuum above LPO in the interface cell (A)")
    g.add_argument("--include-unconverged", type=yes_no, default=False, help="also use unconverged slabs")

    g = p.add_argument_group("step 3: registry and gap scan (scan_interface_registry.py)")
    g.add_argument("--interface", type=optional, default=None,
                   help="interface to scan, e.g. 001_LCO_113_s1-bot__LPO_110_s2-bot; default: smallest mismatch")
    g.add_argument("--grid", type=int, nargs=2, default=[6, 6], metavar=("N1", "N2"), help="lateral shift grid")
    g.add_argument("--gaps", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0], help="LCO-LPO gaps (A)")
    g.add_argument("--min-contact", type=float, default=1.6, help="skip structures with a closer LCO-LPO contact (A)")
    g.add_argument("--shift-cell", choices=["minimal", "lpo", "interface"], default="minimal",
                   help="cell spanned by the shift grid")

    g = p.add_argument_group("step 4: interface relaxations (relax_interface_scan.py)")
    g.add_argument("--interface-fmax", type=float, default=0.02, help="force convergence (eV/A)")
    g.add_argument("--interface-max-steps", type=int, default=2000, help="max optimizer steps per structure")
    g.add_argument("--interface-fix-bottom", type=float, default=0.0, help="freeze LCO atoms this close to its bottom (A)")
    g.add_argument("--top", type=int, default=0, help="relax only the N lowest single points (0 = all)")

    args = p.parse_args(argv)
    for key in ("lco_millers", "lpo_millers"):
        tokens = [t for t in getattr(args, key) if t.strip()]
        setattr(args, key, None if not tokens or [t.lower() for t in tokens] == ["auto"] else tokens)
    for key in ("extra_lco_millers", "extra_lpo_millers"):
        setattr(args, key, [t for t in getattr(args, key) if t.strip() and t.lower() != "none"])
    entry = model_entry(args.model)                 # unknown names stop here
    args.model = entry["model"]                     # e.g. uma-s-1p2p1 -> uma-omat
    args.run_name = args.run_name or entry["tag"]
    return args


# =============================================================================
# Steps
# =============================================================================
def model_argv(args) -> list[str]:
    argv = ["--model", args.model, "--device", args.device, "--dtype", args.dtype, "--uma-settings", args.uma_settings]
    for flag, value in (("--checkpoint", args.checkpoint), ("--head", args.head)):
        if value:
            argv += [flag, value]
    if args.cueq:
        argv.append("--cueq")
    return argv


def step1(args, surf: Path) -> None:
    from relax_lco_lpo_surfaces import main as run

    argv = model_argv(args) + [
        "--outdir", str(surf), "--symprec", str(args.symprec),
        "--n-facets", str(args.n_facets), "--max-index", str(args.max_index),
        "--lco-thickness", str(args.lco_thickness), "--lpo-thickness", str(args.lpo_thickness),
        "--vacuum", str(args.slab_vacuum), "--ftol", str(args.ftol), "--bonds", *args.bonds,
        "--fmax-bulk", str(args.bulk_fmax), "--fmax", str(args.slab_fmax),
        "--max-steps", str(args.slab_max_steps), "--fix-bottom", str(args.slab_fix_bottom)]
    argv += ["--lco-file", args.lco_file] if args.lco_file else ["--lco-id", args.lco_id]
    argv += ["--lpo-file", args.lpo_file] if args.lpo_file else ["--lpo-id", args.lpo_id]
    for flag, value in (("--lco-millers", args.lco_millers), ("--lpo-millers", args.lpo_millers),
                        ("--extra-lco-millers", args.extra_lco_millers),
                        ("--extra-lpo-millers", args.extra_lpo_millers)):
        if value:
            argv += [flag, *value]
    if args.keep_broken:
        argv.append("--keep-broken")
    run(argv)


def step2(args, surf: Path, ifaces: Path) -> None:
    from match_relaxed_surfaces import main as run

    argv = [str(surf), "--outdir", str(ifaces), "--max-strain", str(args.max_strain),
            "--max-atoms", str(args.max_atoms), "--matches-per-pair", str(args.matches_per_pair),
            "--separation", str(args.separation), "--lco-vacuum", str(args.lco_vacuum),
            "--lpo-vacuum", str(args.lpo_vacuum), "--symprec", str(args.symprec)]
    if args.include_unconverged:
        argv.append("--include-unconverged")
    run(argv)


def step3(args, ifaces: Path, record: dict) -> Path:
    from match_relaxed_surfaces import find_interface
    from scan_interface_registry import main as run

    iface, why = find_interface(ifaces, args.interface)
    print(f"interface: {iface.name} ({why})")
    record["interface"] = dict(folder=str(iface), requested=args.interface, reason=why)
    run([str(iface), "--grid", *(str(n) for n in args.grid), "--gaps", *(str(x) for x in args.gaps),
         "--min-contact", str(args.min_contact), "--shift-cell", args.shift_cell])
    return iface


def step4(args, iface: Path) -> None:
    from relax_interface_scan import main as run

    run([str(iface / "registry_scan"), *model_argv(args), "--fmax", str(args.interface_fmax),
         "--max-steps", str(args.interface_max_steps), "--fix-bottom", str(args.interface_fix_bottom),
         "--top", str(args.top)])


def free_memory() -> None:
    """Drop the calculator of the previous step before the next one loads its own."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> None:
    args = parse_args(argv)
    surf, ifaces = Path(f"{args.run_name}_surfaces"), Path(f"{args.run_name}_interfaces")
    last = args.early_stop or 4
    record_path = Path(f"{args.run_name}_workflow.json")
    record = dict(parameters=vars(args), started=time.strftime("%Y-%m-%d %H:%M:%S"), steps={})
    variant = "".join(f", {k} {v}" for k, v in (("checkpoint", args.checkpoint), ("head", args.head)) if v)
    desc = model_entry(args.model)["description"]
    name = desc if desc.startswith(args.model) else f"{args.model} ({desc})"
    print(f"model {name}{variant}; run '{args.run_name}': "
          f"{surf}/, {ifaces}/; steps 1-{last}" + (f" (--early-stop {args.early_stop})" if args.early_stop else ""))

    iface = None
    for n in range(1, last + 1):
        print(f"\n{'=' * 100}\nstep {n}: {STEPS[n]}\n{'=' * 100}", flush=True)
        t0 = time.time()
        if n == 1:
            step1(args, surf)
        elif n == 2:
            step2(args, surf, ifaces)
        elif n == 3:
            iface = step3(args, ifaces, record)
        else:
            step4(args, iface)
        minutes = round((time.time() - t0) / 60.0, 2)
        record["steps"][str(n)] = dict(step=STEPS[n], minutes=minutes, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        record_path.write_text(json.dumps(record, indent=2))
        print(f"step {n} done in {minutes:.1f} min", flush=True)
        free_memory()

    print(f"\nfinished steps 1-{last}; record in {record_path}")
    if iface is not None and last == 4:
        print(f"relaxed interface: {iface}/registry_scan/results.csv\n"
              f"MD: md_interface.py --interfaces {ifaces} --interface {iface.name} --model ...")


if __name__ == "__main__":
    main()

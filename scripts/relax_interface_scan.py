#!/usr/bin/env python
"""
relax_interface_scan.py
=======================
Single points and relaxations of every structure written by scan_interface_registry.py, with the
work of adhesion of each, using the potential chosen with --model (mace-mh = MACE-MH-1, default;
uma-omat = UMA-S, omat task; see mlip_calculators.py).

For every structure in <scan dir>/scan.csv
  1. single point of the structure as generated (the rigid registry/gap scan),
  2. relaxation of the atomic positions; the cell stays fixed (in-plane = LCO lattice with LPO
     strained onto it, c fixed so the vacuum is kept). --fix-bottom freezes the lowest LCO layers;
  3. work of adhesion  W_adh = (E_LCO + E_LPO - E_interface) / A  in J/m^2, with LCO alone and
     strained LPO alone (reference/), computed once in the same cell with the same settings.
     Larger W_adh = stronger binding. The rigid W_adh uses the unrelaxed references and the
     single point, the relaxed one the relaxed references and the relaxed interface. For one pair
     of slabs, ranking by W_adh is the same as ranking by energy.

Also reported for every relaxed structure: final gap, shortest LCO-LPO distance, how far LPO slid
laterally relative to LCO during the relaxation and the shift (in the scan's shift-cell fractions)
it ended at, and the largest displacement of any atom. Structures that end at the same shift with
the same energy have relaxed into the same minimum.

Output (inside the scan directory)
  reference/ref.json, reference/LCO/, reference/LPO/   reference energies and relaxations
  <structure>/POSCAR_relaxed, relaxed.extxyz, relax.log, result.json  (relax.traj with --traj)
  results.csv                                          all structures, lowest relaxed energy first

--top N relaxes only the N structures with the lowest single-point energy, --no-relax stops after the
single points, --only NAME ... restricts the run to some structures. Re-running skips finished work
done on the same POSCAR with the same --model, --head and --dispersion (and --fix-bottom and --fmax
for relaxations), so an interrupted job can simply be restarted; --overwrite redoes everything.
A scan directory holds the results of one model: a run with another --model / --head / --dispersion
stops unless --overwrite is given. To compare two models on the same structures, write a second
copy of the scan with scan_interface_registry.py --outdir and run the other model there.

Requirements: ase >= 3.23, numpy, and mace-torch >= 0.3.14 or fairchem-core >= 2.24. Run next to
mlip_calculators.py. Untested on real data.

Examples
  python relax_interface_scan.py mh1_interfaces/001_LCO_113_s1-bot__LPO_110_s2-bot_432at/registry_scan
  python relax_interface_scan.py uma_interfaces/<interface>/registry_scan --model uma-omat
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS

from mlip_calculators import add_model_args, get_calculator, model_info, resolve_model_args

EV_PER_A2_TO_J_PER_M2 = 16.021766
OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}
RELAX_KEYS = ("relax_tag", "E_relaxed", "converged", "steps", "error", "gap_final_A", "min_contact_final_A",
              "contact_pair_final", "slide_A", "slide_u1", "slide_u2", "max_disp_A")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MACE-MH-1 or UMA single points and relaxations of a registry/gap scan (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("scan_dir", help="output directory of scan_interface_registry.py")
    add_model_args(p, dtype_default="float64", uma_settings_default="batch")
    g = p.add_argument_group("relaxation")
    g.add_argument("--optimizer", choices=list(OPTIMIZERS), default="FIRE", help="ASE optimizer")
    g.add_argument("--fmax", type=float, default=0.02, help="force convergence (eV/A)")
    g.add_argument("--max-steps", type=int, default=2000, help="max optimizer steps per relaxation")
    g.add_argument("--fix-bottom", type=float, default=0.0,
                   help="freeze LCO atoms within this distance (A) of the LCO bottom; 0 = all atoms move")
    g = p.add_argument_group("selection and output")
    g.add_argument("--top", type=int, default=0, help="relax only the N lowest single-point structures (0 = all)")
    g.add_argument("--no-relax", action="store_true", help="single points only (rigid scan)")
    g.add_argument("--only", nargs="+", help="structure names to run (default: all in scan.csv)")
    g.add_argument("--traj", action="store_true", help="also write relax.traj for every relaxation")
    g.add_argument("--overwrite", action="store_true",
                   help="redo finished single points and relaxations, also those of another model")
    return resolve_model_args(p.parse_args(argv))


# =============================================================================
# Helpers
# =============================================================================
def model_tag(args) -> dict:
    """What makes two energies comparable (precision settings are not part of it)."""
    return dict(model=args.checkpoint, head=args.head, dispersion=bool(args.dispersion))


def check_model(scan: Path, args) -> None:
    """Stop if the scan directory already holds results of another model (unless --overwrite)."""
    tag = load_json(scan / "reference" / "ref.json").get("tag")
    if tag and not args.overwrite and not same(tag, model_tag(args)):
        raise SystemExit(f"{scan} holds results of {tag.get('model')} (head {tag.get('head')}, D3 {tag.get('dispersion')}); "
                         f"--overwrite replaces them, or write a copy of the scan with scan_interface_registry.py "
                         f"--outdir and run this model there")


def relax_tag(args) -> dict:
    return dict(model_tag(args), fix_bottom=args.fix_bottom, fmax=args.fmax)


def same(stored: dict, tag: dict) -> bool:
    return all(stored.get(k) == v for k, v in tag.items())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path: Path) -> str:
    """Checksum of an input POSCAR, so results of an older scan with the same name are not reused."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def bottom_indices(atoms, candidates, thickness: float) -> np.ndarray:
    """Indices (among `candidates`) of atoms within `thickness` of the lowest candidate atom."""
    if thickness <= 0:
        return np.array([], dtype=int)
    z = atoms.positions[candidates, 2]
    return np.asarray(candidates)[z < z.min() + thickness]


def relax(atoms, calc, args, d: Path, fixed) -> tuple[bool, int, float]:
    """Positions only (fixed cell). Returns (converged, steps, energy)."""
    atoms.calc = calc
    if len(fixed):
        atoms.set_constraint(FixAtoms(indices=fixed))
    opt = OPTIMIZERS[args.optimizer](atoms, logfile=str(d / "relax.log"))
    traj = Trajectory(str(d / "relax.traj"), "w", atoms) if args.traj else None
    if traj is not None:
        opt.attach(traj.write, interval=1)
    try:
        conv = opt.run(fmax=args.fmax, steps=args.max_steps)
    finally:
        if traj is not None:
            traj.close()
    if conv is None:                                   # older ase: run() returns nothing
        conv = float(np.linalg.norm(atoms.get_forces(), axis=1).max()) < args.fmax
    return bool(conv), int(opt.nsteps), float(atoms.get_potential_energy())


def save_relaxed(atoms, d: Path, energy: float) -> None:
    write(str(d / "POSCAR_relaxed"), atoms, format="vasp", direct=True)
    out = atoms.copy()                                 # plain copy: energy and forces, no calculator
    out.constraints = []
    out.info["energy"] = energy
    out.new_array("forces", np.asarray(atoms.get_forces(apply_constraint=False), dtype=float))
    write(str(d / "relaxed.extxyz"), out, format="extxyz")


def shortest_contact(pa, sa, pb, sb, A2, n_img: int = 2) -> tuple[float, str]:
    """Shortest distance between a part-A atom and a part-B atom (periodic in-plane) and the pair."""
    best, pair = math.inf, ""
    for i in range(-n_img, n_img + 1):
        for j in range(-n_img, n_img + 1):
            t = i * A2[0] + j * A2[1]
            dd = pa[:, None, :] - (pb[None, :, :] + np.array([t[0], t[1], 0.0]))
            r = np.sqrt((dd ** 2).sum(-1))
            k = np.unravel_index(np.argmin(r), r.shape)
            if r[k] < best:
                best, pair = float(r[k]), f"{sa[k[0]]}-{sb[k[1]]}"
    return best, pair


def geometry(initial, final, is_lpo, info) -> dict:
    """Gap, contact, lateral slide of LPO relative to LCO, final shift, largest displacement."""
    A2 = np.array(final.cell)[:2, :2]
    p0, p1 = initial.positions, final.positions        # optimizers do not wrap, so p1 - p0 is the displacement
    sym = np.array(final.get_chemical_symbols())
    lco, lpo = ~is_lpo, is_lpo
    gap = float(p1[lpo, 2].min() - p1[lco, 2].max())
    near_lco = lco & (p1[:, 2] > p1[lco, 2].max() - 4.0)
    near_lpo = lpo & (p1[:, 2] < p1[lpo, 2].min() + 4.0)
    dmin, pair = shortest_contact(p1[near_lco], sym[near_lco], p1[near_lpo], sym[near_lpo], A2)
    disp = p1 - p0
    slide = disp[lpo, :2].mean(axis=0) - disp[lco, :2].mean(axis=0)
    U = np.array([info["shift_cell"]["u1"], info["shift_cell"]["u2"]])
    ds = slide @ np.linalg.inv(U)                      # slide in shift-cell fractions
    return dict(gap_final_A=round(gap, 4), min_contact_final_A=round(dmin, 4), contact_pair_final=pair,
                slide_A=round(float(np.linalg.norm(slide)), 4), slide_u1=round(float(ds[0]), 4),
                slide_u2=round(float(ds[1]), 4), max_disp_A=round(float(np.linalg.norm(disp, axis=1).max()), 4))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# References: LCO alone and strained LPO alone in the interface cell
# =============================================================================
def references(scan: Path, calc, args) -> dict:
    rd = scan / "reference"
    path = rd / "ref.json"
    refs = load_json(path)
    if args.overwrite or not same(refs.get("tag", {}), model_tag(args)):
        refs = {"tag": model_tag(args)}
    refs.setdefault("model_info", model_info(args))       # settings of the run that made these energies
    for key in ("LCO", "LPO"):
        r = refs.get(key, {})
        src = rd / f"POSCAR_{key}"
        atoms = read(str(src), format="vasp")
        if "E_unrelaxed" not in r or r.get("poscar") != digest(src):
            atoms.calc = calc
            r = dict(n_atoms=len(atoms), poscar=digest(src), E_unrelaxed=float(atoms.get_potential_energy()))
        if not args.no_relax and not same(r.get("relax_tag", {}), relax_tag(args)):
            d = rd / key
            d.mkdir(exist_ok=True)
            fixed = bottom_indices(atoms, np.arange(len(atoms)), args.fix_bottom) if key == "LCO" else []
            t0 = time.time()
            conv, steps, e = relax(atoms, calc, args, d, fixed)
            save_relaxed(atoms, d, e)
            r.update(E_relaxed=e, converged=conv, steps=steps, relax_tag=relax_tag(args))
            print(f"  reference {key}: {len(atoms)} atoms, E {r['E_unrelaxed']:.4f} -> {e:.4f} eV, "
                  f"converged {conv} ({steps} steps, {time.time() - t0:.0f} s)")
        refs[key] = r
        path.write_text(json.dumps(refs, indent=2))
    return refs


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> None:
    args = parse_args(argv)
    scan = Path(args.scan_dir)
    info = json.loads((scan / "scan_info.json").read_text())
    with open(scan / "scan.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["written"] == "True"]
    if args.only:
        rows = [r for r in rows if r["name"] in set(args.only)]
    if not rows:
        raise SystemExit("no structures to run")
    is_lpo = np.zeros(info["n_atoms"], dtype=bool)
    is_lpo[info["lpo_indices"]] = True
    area = float(info["area_A2"])
    w_adh = lambda e_lco, e_lpo, e_int: round((e_lco + e_lpo - e_int) / area * EV_PER_A2_TO_J_PER_M2, 5)
    print(f"{scan}: {len(rows)} structures of {info['n_atoms']} atoms ({info['n_lco']} LCO + {info['n_lpo']} LPO), "
          f"area {area:.2f} A^2")

    check_model(scan, args)
    calc = get_calculator(args)
    refs = references(scan, calc, args)

    # ---- 1. single points ------------------------------------------------------------------------
    results = {}
    t0 = time.time()
    for r in rows:
        d = scan / r["name"]
        res = load_json(d / "result.json")
        if (args.overwrite or not same(res.get("tag", {}), model_tag(args))
                or res.get("poscar") != digest(d / "POSCAR")):
            atoms = read(str(d / "POSCAR"), format="vasp")
            if len(atoms) != info["n_atoms"]:
                raise RuntimeError(f"{d}/POSCAR has {len(atoms)} atoms, scan_info.json says {info['n_atoms']}")
            atoms.calc = calc
            res = dict(name=r["name"], tag=model_tag(args), poscar=digest(d / "POSCAR"),
                       E_sp=float(atoms.get_potential_energy()))
            (d / "result.json").write_text(json.dumps(res, indent=2))
        results[r["name"]] = res
    print(f"single points done ({time.time() - t0:.0f} s)")

    # ---- 2. relaxations -------------------------------------------------------------------------
    if not args.no_relax:
        order = sorted(rows, key=lambda r: results[r["name"]]["E_sp"])
        chosen = order[: args.top] if args.top else order
        for k, r in enumerate(chosen, 1):
            d = scan / r["name"]
            res = results[r["name"]]
            if not args.overwrite and same(res.get("relax_tag", {}), relax_tag(args)) and "E_relaxed" in res:
                continue
            for key in RELAX_KEYS:                     # drop results of earlier settings or failed attempts
                res.pop(key, None)
            initial = read(str(d / "POSCAR"), format="vasp")
            atoms = initial.copy()
            fixed = bottom_indices(atoms, np.nonzero(~is_lpo)[0], args.fix_bottom)
            t1 = time.time()
            try:
                conv, steps, e = relax(atoms, calc, args, d, fixed)
            except Exception as exc:                   # keep going; the error is recorded
                res.update(relax_tag=relax_tag(args), error=repr(exc))
                (d / "result.json").write_text(json.dumps(res, indent=2))
                print(f"  [{k}/{len(chosen)}] {r['name']}: relaxation failed ({exc})")
                continue
            save_relaxed(atoms, d, e)
            res.update(relax_tag=relax_tag(args), E_relaxed=e, converged=conv, steps=steps,
                       **geometry(initial, atoms, is_lpo, info))
            (d / "result.json").write_text(json.dumps(res, indent=2))
            print(f"  [{k}/{len(chosen)}] {r['name']}: E {res['E_sp']:.4f} -> {e:.4f} eV, converged {conv} "
                  f"({steps} steps, {time.time() - t1:.0f} s), gap {res['gap_final_A']:.2f} A, slide {res['slide_A']:.2f} A")

    # ---- 3. table -------------------------------------------------------------------------------
    lco, lpo = refs["LCO"], refs["LPO"]
    refs_relaxed = all("E_relaxed" in x and same(x.get("relax_tag", {}), relax_tag(args)) for x in (lco, lpo))
    table = []
    for r in rows:
        res = results[r["name"]]
        row = dict(name=r["name"], i=int(r["i"]), j=int(r["j"]), gap_initial_A=float(r["gap_A"]),
                   shift_u1=float(r["shift_u1"]), shift_u2=float(r["shift_u2"]),
                   shift_x_A=float(r["shift_x_A"]), shift_y_A=float(r["shift_y_A"]),
                   min_contact_initial_A=float(r["min_contact_A"]),
                   E_sp_eV=round(res["E_sp"], 5),
                   W_adh_rigid_J_m2=w_adh(lco["E_unrelaxed"], lpo["E_unrelaxed"], res["E_sp"]))
        if "E_relaxed" in res and same(res.get("relax_tag", {}), relax_tag(args)):
            row.update(E_relaxed_eV=round(res["E_relaxed"], 5), converged=res["converged"], steps=res["steps"],
                       W_adh_J_m2=(w_adh(lco["E_relaxed"], lpo["E_relaxed"], res["E_relaxed"])
                                   if refs_relaxed else None),
                       gap_final_A=res["gap_final_A"], min_contact_final_A=res["min_contact_final_A"],
                       contact_pair_final=res["contact_pair_final"], slide_A=res["slide_A"],
                       final_shift_u1=round((row["shift_u1"] + res["slide_u1"]) % 1.0, 4) % 1.0,
                       final_shift_u2=round((row["shift_u2"] + res["slide_u2"]) % 1.0, 4) % 1.0,
                       max_disp_A=res["max_disp_A"])
        elif "error" in res:
            row.update(error=res["error"])
        table.append(row)
    relaxed = [t for t in table if "E_relaxed_eV" in t]
    if relaxed:
        e_min = min(t["E_relaxed_eV"] for t in relaxed)
        for t in relaxed:
            t["dE_relaxed_eV"] = round(t["E_relaxed_eV"] - e_min, 5)
    table.sort(key=lambda t: (0, t["E_relaxed_eV"]) if "E_relaxed_eV" in t else (1, t["E_sp_eV"]))
    write_csv(table, scan / "results.csv")

    print(f"\nreferences: LCO {lco['E_unrelaxed']:.4f} eV" + (f" (relaxed {lco['E_relaxed']:.4f})" if "E_relaxed" in lco else "")
          + f", LPO {lpo['E_unrelaxed']:.4f} eV" + (f" (relaxed {lpo['E_relaxed']:.4f})" if "E_relaxed" in lpo else ""))
    show = [t for t in table if "E_relaxed_eV" in t][:5] or table[:5]     # table is sorted
    print("lowest " + ("relaxed" if relaxed else "single-point") + " energies:")
    for t in show:
        if "E_relaxed_eV" in t:
            print(f"  {t['name']}: E {t['E_relaxed_eV']:.4f} eV (+{t['dE_relaxed_eV']:.4f}), W_adh {t['W_adh_J_m2']} J/m^2, "
                  f"gap {t['gap_final_A']:.2f} A, ends at shift ({t['final_shift_u1']:.2f}, {t['final_shift_u2']:.2f})")
        else:
            print(f"  {t['name']}: E_sp {t['E_sp_eV']:.4f} eV, W_adh (rigid) {t['W_adh_rigid_J_m2']} J/m^2")
    print(f"results.csv written to {scan}/")


if __name__ == "__main__":
    main()

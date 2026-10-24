#!/usr/bin/env python
"""
relax_lco_lpo_surfaces.py
=========================
Relax bulk LiCoO2 (LCO) and Li3PO4 (LPO) and all their candidate surfaces - atomic positions
AND in-plane lattice - with a foundation model, MACE-MH-1 or UMA, before any lattice matching.

Model (--backend, see mlip_calculators.py)
  mace  MACE-MH-1 (mace-foundations), head omat_pbe by default: trained on OMat (periodic
        materials, PBE/PBE+U), the level of the MP cells used in the screening; matpes_r2scan is
        the r2SCAN alternative.
  uma   UMA (fairchem-core), uma-s-1p2p1 with the omat task by default (OMat24, PBE/PBE+U).
  Both are short-range models without explicit electrostatics, so treat polar terminations with
  care. Slabs are fully periodic with --vacuum, which must be wider than the model's cutoff
  (6 A for both) so that the two faces do not interact through it.

Pipeline
  1. Bulk cells from the Materials Project (as in lco_lpo_lattice_match.py), --lco-id / --lpo-id,
     or local files.
  2. Full bulk relaxation (positions + cell, space group kept). bulk.csv compares MP and model
     lattice parameters; --bulk-only stops here so you can check them first.
  3. Facets: the screening's "reasonable" Miller indices (Donnay-Harker ranking of the MP cell,
     distinct under the crystal's own Laue group) or explicit --lco-millers / --lpo-millers;
     --extra-lco-millers / --extra-lpo-millers add facets to either list. pairs.csv lists every
     (LCO hkl, LPO hkl) pair; each facet is built and relaxed only once.
  4. All symmetry-distinct slabs per facet, cut from the relaxed bulk: whole repeat units,
     >= --lco-thickness / --lpo-thickness, Gauss-reduced primitive surface cell, c perpendicular
     to the surface. Cuts related by a bulk operation (including turning the slab over) give the
     same slab. Slabs that cut a bond in --bonds (default P-O, i.e. a PO4 tetrahedron) are
     skipped and listed in skipped.csv, unless --keep-broken.
  5. Single point of the unrelaxed slab (unrelaxed surface energy), then relaxation of all atoms
     and the in-plane lattice (a, b, gamma); c is fixed so the vacuum cannot collapse.

Repeating a run with another model: --like DIR
  Takes from an earlier output directory the input bulk cells (DIR/bulk/*_input_conv.vasp, i.e.
  the same MP structures), the facets (DIR/pairs.csv) and the slab and relaxation settings
  (DIR/run_args.json: thicknesses, vacuum, bonds, fmax, ...). Options given on the command line
  win. Only the model changes, so the slabs get the same names in both directories. With the
  same model and --outdir DIR, it extends DIR instead, e.g. by --extra-lco-millers (finished slabs
  are skipped).

Output (--outdir; default mh1_surfaces for MACE, uma_surfaces for UMA)
  bulk/, bulk.csv          input and relaxed bulk cells (<side>_<mpid>_MACE_conv.vasp or ..._UMA_..);
                           MP vs model lattice parameters
  surfaces/<name>/         POSCAR_initial, POSCAR_relaxed, relaxed.extxyz, relax.traj, relax.log,
                           info.json
  surfaces.csv             per slab: initial and relaxed in-plane lattice, relaxation strain,
                           formal-charge dipole, unrelaxed and relaxed surface energy, convergence
  skipped.csv, pairs.csv   slabs not relaxed and why; (LCO hkl, LPO hkl) -> relaxed slab names
Re-running skips slabs whose info.json was made with the same --model and --head
(--overwrite to redo them). A directory holding the results of another model is not touched
unless --overwrite is given.

Surface energy = (E_slab - N e_bulk) / (2 A), the mean of the two faces of a stoichiometric
slab; for polar (Tasker type 3) slabs it is not a clean surface property.

Requirements: ase >= 3.23, pymatgen, mp-api (only for MP queries), and mace-torch >= 0.3.14 or
fairchem-core >= 2.24 (separate environments). Run next to mlip_calculators.py,
lco_lpo_lattice_match.py and make_lco_lpo_terminations.py (helpers are imported). Untested.

Examples
  python relax_lco_lpo_surfaces.py --bulk-only            # check the bulk lattices first
  python relax_lco_lpo_surfaces.py --lco-thickness 15 --lpo-thickness 15 --vacuum 20
  python relax_lco_lpo_surfaces.py --backend uma --like mh1_surfaces             # -> uma_surfaces
  python relax_lco_lpo_surfaces.py --backend uma --like mh1_surfaces --extra-lco-millers 012 021
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixSymmetry
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from ase.filters import FrechetCellFilter as CellFilter      # ase >= 3.23
except ImportError:                                             # older ase
    from ase.constraints import ExpCellFilter as CellFilter

from lco_lpo_lattice_match import (angle_deg, fmt_hkl, make_phase, parse_millers, phases_from_files,
                                   phases_from_mp, reduce_2d, select_facets, strain_metrics)
from make_lco_lpo_terminations import (bulk_coordination, cartesian_ops, coordination, layer_cuts, make_slab,
                                       ouc_frame, parse_pairs, unit)
from mlip_calculators import add_model_args, get_calculator, model_info, model_label, resolve_model_args

EV_PER_A2_TO_J_PER_M2 = 16.021766
OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}
DEFAULT_OUTDIR = {"mace": "mh1_surfaces", "uma": "uma_surfaces"}
# settings that --like copies from the run_args.json of the earlier run
LIKE_KEYS = ("symprec", "max_index", "n_facets", "ranking", "lco_thickness", "lpo_thickness", "vacuum", "ftol",
             "max_slabs", "bonds", "keep_broken", "charges", "optimizer", "fmax_bulk", "fmax", "max_steps",
             "no_fix_symmetry", "fix_bottom")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MACE-MH-1 or UMA relaxation of LCO / LPO bulk and surfaces (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--like", metavar="DIR",
                   help="repeat an earlier run: its input bulk cells, facets and settings become the defaults")
    g = p.add_argument_group("materials")
    g.add_argument("--lco-formula", default="LiCoO2", help="LCO formula for the MP query")
    g.add_argument("--lpo-formula", default="Li3PO4", help="LPO formula for the MP query")
    g.add_argument("--lco-id", help="MP id for LCO (default: lowest-E_hull ICSD polymorph)")
    g.add_argument("--lpo-id", help="MP id for LPO (default: lowest-E_hull ICSD polymorph)")
    g.add_argument("--lco-file", help="local LCO bulk structure used instead of MP")
    g.add_argument("--lpo-file", help="local LPO bulk structure used instead of MP")
    g.add_argument("--max-ehull", type=float, default=0.05, help="E_hull window of the MP query (eV/atom)")
    g.add_argument("--include-theoretical", action="store_true", help="also accept MP entries without ICSD match")
    g.add_argument("--api-key", default=None, help="MP API key (default: MP_API_KEY environment variable)")
    g.add_argument("--symprec", type=float, default=0.1, help="spglib tolerance (A) for standardising cells")

    g = p.add_argument_group("facets and slabs")
    g.add_argument("--max-index", type=int, default=4, help="largest |h|,|k|,|l| for the automatic facets")
    g.add_argument("--n-facets", type=int, default=8, help="facets per material (automatic selection)")
    g.add_argument("--ranking", choices=["dh", "lattice"], default="dh", help="facet ranking (as in the screening)")
    g.add_argument("--lco-millers", nargs="+", help="explicit LCO facets, e.g. 001 104 012 211")
    g.add_argument("--lpo-millers", nargs="+", help="explicit LPO facets, e.g. 011 100")
    g.add_argument("--extra-lco-millers", nargs="+", default=[], help="LCO facets added to the list, e.g. 012 021")
    g.add_argument("--extra-lpo-millers", nargs="+", default=[], help="LPO facets added to the list")
    g.add_argument("--lco-thickness", type=float, default=12.0, help="minimum LCO slab thickness (A)")
    g.add_argument("--lpo-thickness", type=float, default=12.0, help="minimum LPO slab thickness (A)")
    g.add_argument("--vacuum", type=float, default=20.0, help="vacuum thickness (A), larger than ~12 A")
    g.add_argument("--ftol", type=float, default=0.1, help="atoms closer than this along the normal form one layer (A)")
    g.add_argument("--max-slabs", type=int, default=0, help="max slabs (terminations) per facet, 0 = all")
    g.add_argument("--bonds", nargs="+", default=["P-O:2.0"],
                   help="cation-anion bonds (A-B:cutoff) a slab must not cut; 'none' disables the filter")
    g.add_argument("--keep-broken", action="store_true", help="also relax slabs that cut --bonds")
    g.add_argument("--charges", nargs="+", default=["Li:1", "Co:3", "P:5", "O:-2"],
                   help="formal charges for the reported slab dipole")

    add_model_args(p, dtype_default="float64", uma_settings_default="batch")

    g = p.add_argument_group("relaxation and output")
    g.add_argument("--bulk-only", action="store_true", help="relax the bulk cells, write bulk.csv and stop")
    g.add_argument("--optimizer", choices=list(OPTIMIZERS), default="FIRE", help="ASE optimizer")
    g.add_argument("--fmax-bulk", type=float, default=0.005, help="bulk force/stress convergence (eV/A)")
    g.add_argument("--fmax", type=float, default=0.02, help="slab force/stress convergence (eV/A)")
    g.add_argument("--max-steps", type=int, default=2000, help="max optimizer steps per relaxation")
    g.add_argument("--no-fix-symmetry", action="store_true", help="do not constrain the bulk space group")
    g.add_argument("--fix-bottom", type=float, default=0.0,
                   help="freeze atoms within this distance (A) of the slab bottom (0 = all atoms free)")
    g.add_argument("--outdir", default=None, help="output directory (default: mh1_surfaces / uma_surfaces)")
    g.add_argument("--overwrite", action="store_true",
                   help="re-relax slabs that already have info.json; allow an --outdir of another model")

    pre, _ = p.parse_known_args(argv)
    if pre.like:
        p.set_defaults(**like_defaults(Path(pre.like)))
    args = resolve_model_args(p.parse_args(argv))
    if args.outdir is None:
        args.outdir = DEFAULT_OUTDIR[args.backend]
    return args


def like_defaults(run: Path) -> dict:
    """Defaults that repeat an earlier run of this script: its slab and relaxation settings
    (run_args.json), its input bulk cells (bulk/<side>_<mpid>_input_conv.vasp) and its facets
    (pairs.csv, in the original order)."""
    try:
        old = json.loads((run / "run_args.json").read_text())
        with open(run / "bulk.csv", newline="") as fh:
            mpids = {r["material"]: r["mpid"] for r in csv.DictReader(fh)}
    except FileNotFoundError as exc:
        raise SystemExit(f"--like {run}: {exc.filename} not found") from exc
    out = {k: old[k] for k in LIKE_KEYS if k in old}
    for side in ("LCO", "LPO"):
        f = run / "bulk" / f"{side}_{mpids[side]}_input_conv.vasp"
        if not f.exists():
            raise SystemExit(f"--like {run}: {f} not found")
        out[f"{side.lower()}_file"] = str(f)
    pairs = run / "pairs.csv"
    if pairs.exists():
        with open(pairs, newline="") as fh:
            rows = list(csv.DictReader(fh))
        out["lco_millers"] = list(dict.fromkeys(r["lco_hkl"] for r in rows))
        out["lpo_millers"] = list(dict.fromkeys(r["lpo_hkl"] for r in rows))
    else:
        out["lco_millers"], out["lpo_millers"] = old.get("lco_millers"), old.get("lpo_millers")
        print(f"note: {pairs} not found - facets as given in that run, or selected again if it chose them itself")
    shown = ", ".join(f"{k}={out[k]}" for k in ("lco_thickness", "lpo_thickness", "vacuum", "fmax") if k in out)
    print(f"--like {run}: input cells {out['lco_file']}, {out['lpo_file']}; {shown}")
    print(f"  LCO facets {out['lco_millers'] or 'automatic'}; LPO facets {out['lpo_millers'] or 'automatic'}")
    return out


def file_label(path: str, side: str) -> str | None:
    """MP id (or the label of an earlier run) from a file name such as LCO_mp-22526_input_conv.vasp."""
    stem = Path(path).stem
    m = re.search(r"mp-\d+", stem)
    if m:
        return m.group(0)
    m = re.fullmatch(rf"{side}_(.+)_input_conv", stem)
    return m.group(1) if m else None


def write_csv(rows: list[dict], path: Path) -> None:
    """Table with the union of the keys of all rows: rows resumed from an older run may have fewer."""
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def check_outdir(outdir: Path, args) -> None:
    """Refuse to mix the results of two models in one directory (unless --overwrite)."""
    f = outdir / "run_args.json"
    if not f.exists() or args.overwrite:
        return
    old = json.loads(f.read_text())
    if (old.get("model"), old.get("head")) != (args.model, args.head):
        raise SystemExit(f"{outdir} holds results of {old.get('model')} ({old.get('head')}); use another --outdir "
                         f"or --overwrite to replace them")


# =============================================================================
# Relaxations
# =============================================================================
def run_opt(target, atoms: Atoms, args, fmax: float, logfile: Path, trajfile: Path) -> tuple[bool, int]:
    """Optimise `target` (Atoms or cell filter); the trajectory stores the underlying atoms."""
    opt = OPTIMIZERS[args.optimizer](target, logfile=str(logfile))
    traj = Trajectory(str(trajfile), "w", atoms)
    opt.attach(traj.write, interval=1)
    try:
        result = opt.run(fmax=fmax, steps=args.max_steps)
    finally:
        traj.close()
    if result is None:                       # older ase: run() returns nothing
        result = float(np.linalg.norm(target.get_forces(), axis=1).max()) < fmax
    return bool(result), int(opt.nsteps)


def relax_bulk(phase, calc, args, outdir: Path, side: str) -> tuple[Structure, float, bool, int]:
    """Positions + full cell, space group conserved. Returns (structure, energy per atom, converged, steps)."""
    atoms = AseAtomsAdaptor.get_atoms(phase.conv)
    atoms.pbc = True
    atoms.calc = calc
    if not args.no_fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms, symprec=0.01))
    ok, steps = run_opt(CellFilter(atoms), atoms, args, args.fmax_bulk,
                        outdir / f"{side}_relax.log", outdir / f"{side}_relax.traj")
    e_atom = atoms.get_potential_energy() / len(atoms)
    relaxed = Structure(np.array(atoms.cell), atoms.get_chemical_symbols(), atoms.get_scaled_positions())
    return relaxed, float(e_atom), ok, steps


def relax_slab(atoms: Atoms, calc, args, d: Path) -> tuple[bool, int]:
    """All atoms (optionally minus a frozen bottom region) and the in-plane cell; c is fixed."""
    atoms.calc = calc
    if args.fix_bottom > 0:
        z = atoms.positions[:, 2]
        atoms.set_constraint(FixAtoms(indices=np.nonzero(z < z.min() + args.fix_bottom)[0]))
    # Voigt mask (xx, yy, zz, yz, xz, xy): only the in-plane strains are free
    return run_opt(CellFilter(atoms, mask=[1, 1, 0, 0, 0, 1]), atoms, args, args.fmax,
                   d / "relax.log", d / "relax.traj")


# =============================================================================
# Slabs
# =============================================================================
def distinct_slab_cuts(cuts, u, d: float, ops, tol: float = 0.05) -> list[float]:
    """One cut per class of congruent slabs. An operation with R u = u maps the cut at height z to
    z + u.t; one with R u = -u turns the slab over and maps z to -z + u.t (modulo the plane spacing d)."""
    maps = []
    for rot, t in ops:
        if np.allclose(rot @ u, u, atol=1e-3):
            maps.append((1.0, float(u @ t)))
        elif np.allclose(rot @ u, -u, atol=1e-3):
            maps.append((-1.0, float(u @ t)))
    reps = []
    for z in cuts:
        if not any(abs(((z - s * zr - dl) / d + 0.5) % 1.0 - 0.5) * d < tol for zr in reps for s, dl in maps):
            reps.append(z)
    return reps


def slab_atoms(species, cart, mesh, u, vacuum: float) -> Atoms:
    """Fully periodic ASE slab: Gauss-reduced surface cell, x || a, z || surface normal,
    c = (0, 0, thickness + vacuum), slab centred in z."""
    a, b = reduce_2d(mesh[0], mesh[1])      # same lattice, basis with a 60-90 deg angle
    if u @ np.cross(a, b) < 0:               # right-handed (a, b, normal)
        b = -b
    ex = unit(a)
    rot = np.array([ex, np.cross(u, ex), u])
    pos = np.asarray(cart, dtype=float) @ rot.T
    pos[:, 2] -= pos[:, 2].min()
    height = pos[:, 2].max() + vacuum
    pos[:, 2] += 0.5 * vacuum
    a2, b2 = a @ rot.T, b @ rot.T
    cell = np.array([[a2[0], a2[1], 0.0], [b2[0], b2[1], 0.0], [0.0, 0.0, height]])
    atoms = Atoms([s.symbol for s in species], positions=pos, cell=cell, pbc=True)
    atoms.wrap()
    return atoms


def cut_bonds(atoms: Atoms, bonds: dict, ref: dict) -> list[str]:
    """Bond types (cation-anion pairs of `bonds` present in the bulk) whose bulk coordination is
    lost somewhere in the slab, e.g. 'P-O' if a PO4 tetrahedron is cut."""
    if not ref:
        return []
    struct = AseAtomsAdaptor.get_structure(atoms)
    return [f"{a}-{b}" for (a, b), n_ref in ref.items()
            if any(n != n_ref for n in coordination(struct, a, b, bonds[(a, b)]).values())]


def dipole_per_area(atoms: Atoms, charges: dict) -> float:
    """Formal-charge dipole along the normal per area (e/A); non-zero = polar slab."""
    q = np.array([charges.get(s, 0.0) for s in atoms.get_chemical_symbols()])
    z = atoms.positions[:, 2]
    area = float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))
    return float(q @ (z - z.mean())) / area


def inplane(atoms: Atoms) -> dict:
    a, b = np.array(atoms.cell[0]), np.array(atoms.cell[1])
    return dict(a=float(np.linalg.norm(a)), b=float(np.linalg.norm(b)), gamma=float(angle_deg(a, b)),
                area=float(np.linalg.norm(np.cross(a, b))))


def save_extxyz(atoms: Atoms, energy: float, forces, path: Path) -> None:
    """Relaxed slab with energy and forces (no calculator attached)."""
    out = atoms.copy()
    out.info["energy"] = energy
    out.new_array("forces", np.asarray(forces, dtype=float))
    write(str(path), out, format="extxyz")


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> None:
    args = parse_args(argv)
    label = model_label(args)                       # MACE or UMA, for file names and columns
    bonds = {} if [t.lower() for t in args.bonds] == ["none"] else parse_pairs(args.bonds, "-")
    charges = parse_pairs(args.charges, "")
    outdir = Path(args.outdir)
    check_outdir(outdir, args)
    (outdir / "bulk").mkdir(parents=True, exist_ok=True)
    (outdir / "surfaces").mkdir(exist_ok=True)
    (outdir / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    spec = {
        "LCO": dict(formula=args.lco_formula, mpid=args.lco_id, file=args.lco_file, millers=args.lco_millers,
                    extra=args.extra_lco_millers, thickness=args.lco_thickness),
        "LPO": dict(formula=args.lpo_formula, mpid=args.lpo_id, file=args.lpo_file, millers=args.lpo_millers,
                    extra=args.extra_lpo_millers, thickness=args.lpo_thickness),
    }

    # ---- 1. input bulk cells (MP or files) -------------------------------------------
    inp = {}
    for side, sp in spec.items():
        if sp["file"]:
            ph = phases_from_files([sp["file"]], side, args.symprec)[0]
            tag = file_label(sp["file"], side)       # keep the MP id of a file written by an earlier run
            if tag:
                ph = dataclasses.replace(ph, mpid=tag)
                print(f"  {side}: id {tag} from the file name")
            inp[side] = ph
    if len(inp) < 2:
        from mp_api.client import MPRester

        with MPRester(args.api_key) as mpr:
            for side, sp in spec.items():
                if side not in inp:
                    ids = [sp["mpid"]] if sp["mpid"] else None
                    inp[side] = phases_from_mp(mpr, sp["formula"], ids, 1, args, side)[0]

    # ---- 2. bulk relaxation ------------------------------------------------------------
    calc = get_calculator(args)
    rel, e_bulk, bulk_rows = {}, {}, []
    for side in ("LCO", "LPO"):
        ph = inp[side]
        ph.conv.to(filename=str(outdir / "bulk" / f"{side}_{ph.mpid}_input_conv.vasp"), fmt="poscar")
        print(f"\nrelaxing bulk {side}: {ph.label}, {len(ph.conv)} atoms")
        s, e_bulk[side], ok, steps = relax_bulk(ph, calc, args, outdir / "bulk", side)
        rel[side] = make_phase(s, f"{ph.mpid}-{label}", None, ph.spacegroup, args.symprec)
        rel[side].conv.to(filename=str(outdir / "bulk" / f"{side}_{ph.mpid}_{label}_conv.vasp"), fmt="poscar")
        l0, l1 = ph.conv.lattice, rel[side].conv.lattice
        row = dict(material=side, mpid=ph.mpid, spacegroup=ph.spacegroup, spacegroup_relaxed=rel[side].spacegroup,
                   model=args.model, head=args.head, backend=args.backend)
        for key, v0, v1 in zip(("a", "b", "c", "alpha", "beta", "gamma"), (*l0.abc, *l0.angles), (*l1.abc, *l1.angles)):
            row[f"{key}_MP"], row[f"{key}_{label}"] = round(v0, 4), round(v1, 4)
        row.update({"V_MP": round(l0.volume, 3), f"V_{label}": round(l1.volume, 3),
                    "dV_pct": round(100 * (l1.volume / l0.volume - 1), 3),
                    f"e_{label}_eV_atom": round(e_bulk[side], 6), "converged": ok, "steps": steps})
        bulk_rows.append(row)
        print(f"  MP   a, b, c = {l0.a:.4f}, {l0.b:.4f}, {l0.c:.4f} A, V = {l0.volume:.2f} A^3")
        print(f"  {label:<4} a, b, c = {l1.a:.4f}, {l1.b:.4f}, {l1.c:.4f} A, V = {l1.volume:.2f} A^3 "
              f"({row['dV_pct']:+.2f} %), converged: {ok} after {steps} steps")
        if max(abs(x1 / x0 - 1) for x0, x1 in zip(l0.abc, l1.abc)) > 0.05:
            print("  ! a lattice parameter changed by more than 5 % - check the model and the cell setting")
    write_csv(bulk_rows, outdir / "bulk.csv")
    if args.bulk_only:
        print(f"\n--bulk-only: relaxed bulk cells in {outdir}/bulk, comparison in {outdir}/bulk.csv")
        return

    # ---- 3. facets (selected on the MP cells, as in the screening) ----------------------
    facets = {}
    for side, sp in spec.items():
        explicit = parse_millers(sp["millers"]) if sp["millers"] else None
        facets[side] = [tuple(int(x) for x in f.hkl)
                        for f in select_facets(inp[side], args.max_index, args.n_facets, args.ranking, explicit)]
        for hkl in parse_millers(sp["extra"]):
            if hkl not in facets[side]:
                facets[side].append(hkl)
        print(f"{side} facets: " + ", ".join(f"({fmt_hkl(h)})" for h in facets[side]))

    # ---- 4./5. slabs from the relaxed bulk: build, filter, relax --------------------------
    rows, skipped, by_facet = [], [], {}
    for side, sp in spec.items():
        ph = rel[side]
        ops = cartesian_ops(ph.conv, args.symprec)
        ref = bulk_coordination(ph.conv, bonds)
        for hkl in facets[side]:
            sg = SlabGenerator(ph.conv, hkl, min_slab_size=sp["thickness"], min_vacuum_size=15.0,
                               in_unit_planes=False, center_slab=True, primitive=True, reorient_lattice=False)
            u, h = ouc_frame(sg)
            cuts = distinct_slab_cuts(layer_cuts(sg, h, args.ftol), u, ph.conv.lattice.d_hkl(hkl), ops)
            if args.max_slabs:
                cuts = cuts[: args.max_slabs]
            print(f"\n{side} ({fmt_hkl(hkl)}): {len(cuts)} distinct slabs")
            by_facet[(side, hkl)] = []
            for k, z in enumerate(cuts):
                name = f"{side}_{''.join(str(x) for x in hkl)}_s{k}"
                d = outdir / "surfaces" / name
                if (d / "info.json").exists() and not args.overwrite:
                    old = json.loads((d / "info.json").read_text())
                    if old.get("model") == args.model and old.get("head") == args.head:
                        rows.append(old)
                        by_facet[(side, hkl)].append(name)
                        print(f"  {name}: already relaxed with this model/head, skipped")
                        continue
                species, cart, mesh = make_slab(ph.conv, hkl, z / h, sp["thickness"], u, h)
                atoms = slab_atoms(species, cart, mesh, u, args.vacuum)
                broken = cut_bonds(atoms, bonds, ref)
                if broken and not args.keep_broken:
                    skipped.append(dict(name=name, material=side, hkl=fmt_hkl(hkl), slab=k, cut=round(z / h, 5),
                                        n_atoms=len(atoms), reason="cuts " + "/".join(broken) + " bonds"))
                    print(f"  {name}: cuts {'/'.join(broken)} bonds - not relaxed")
                    continue
                by_facet[(side, hkl)].append(name)
                d.mkdir(parents=True, exist_ok=True)
                write(str(d / "POSCAR_initial"), atoms, format="vasp", direct=True, sort=True)
                cell0, vec0 = inplane(atoms), np.array(atoms.cell[:2])
                thick0 = float(np.ptp(atoms.positions[:, 2]))
                dipole = dipole_per_area(atoms, charges)

                atoms.calc = calc
                e0 = float(atoms.get_potential_energy())                      # unrelaxed single point
                gamma0 = (e0 - len(atoms) * e_bulk[side]) / (2.0 * cell0["area"])
                ok, steps = relax_slab(atoms, calc, args, d)

                cell1, vec1 = inplane(atoms), np.array(atoms.cell[:2])
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()
                met = strain_metrics(vec0[0], vec0[1], vec1[0], vec1[1])      # initial -> relaxed cell
                gamma1 = (energy - len(atoms) * e_bulk[side]) / (2.0 * cell1["area"])
                write(str(d / "POSCAR_relaxed"), atoms, format="vasp", direct=True, sort=True)
                save_extxyz(atoms, energy, forces, d / "relaxed.extxyz")
                info = dict(name=name, material=side, mpid=inp[side].mpid, hkl=fmt_hkl(hkl), slab=k,
                            cut=round(z / h, 5), n_atoms=len(atoms), formula=atoms.get_chemical_formula(),
                            **model_info(args), cut_bonds="/".join(broken) or "-",
                            dipole_e_per_A=round(dipole, 5), polar=abs(dipole) > 1e-3,
                            thickness_initial=round(thick0, 3),
                            thickness_relaxed=round(float(np.ptp(atoms.positions[:, 2])), 3))
                for tag, cell in (("initial", cell0), ("relaxed", cell1)):
                    info.update({f"a_{tag}": round(cell["a"], 5), f"b_{tag}": round(cell["b"], 5),
                                 f"gamma_{tag}": round(cell["gamma"], 4), f"area_{tag}": round(cell["area"], 4)})
                info.update(relax_strain_e1_pct=round(100 * met["e1"], 4), relax_strain_e2_pct=round(100 * met["e2"], 4),
                            energy_unrelaxed_eV=round(e0, 6), energy_eV=round(energy, 6),
                            e_bulk_eV_atom=round(e_bulk[side], 6),
                            surface_energy_unrelaxed_J_m2=round(gamma0 * EV_PER_A2_TO_J_PER_M2, 4),
                            surface_energy_J_m2=round(gamma1 * EV_PER_A2_TO_J_PER_M2, 4),
                            max_force_eV_A=round(float(np.linalg.norm(forces, axis=1).max()), 4),
                            steps=steps, converged=ok)
                (d / "info.json").write_text(json.dumps(info, indent=2))
                rows.append(info)
                print(f"  {name}: {len(atoms)} atoms, {'polar' if info['polar'] else 'non-polar'}; "
                      f"a, b, gamma {cell0['a']:.3f}, {cell0['b']:.3f}, {cell0['gamma']:.2f} -> "
                      f"{cell1['a']:.3f}, {cell1['b']:.3f}, {cell1['gamma']:.2f}; E_surf "
                      f"{gamma0 * EV_PER_A2_TO_J_PER_M2:.3f} -> {gamma1 * EV_PER_A2_TO_J_PER_M2:.3f} J/m^2; "
                      f"converged {ok} ({steps} steps)")

    write_csv(rows, outdir / "surfaces.csv")
    write_csv(skipped, outdir / "skipped.csv")
    pair_rows = [dict(lco_hkl=fmt_hkl(hs), lpo_hkl=fmt_hkl(hf), lco_slabs=" ".join(by_facet[("LCO", hs)]),
                      lpo_slabs=" ".join(by_facet[("LPO", hf)]))
                 for hs in facets["LCO"] for hf in facets["LPO"]]
    write_csv(pair_rows, outdir / "pairs.csv")
    n_bad = sum(not r["converged"] for r in rows)
    print(f"\n{len(rows)} slabs relaxed ({n_bad} not converged), {len(skipped)} skipped (skipped.csv); "
          f"{len(pair_rows)} facet pairs in pairs.csv; bulk comparison in bulk.csv")


if __name__ == "__main__":
    main()

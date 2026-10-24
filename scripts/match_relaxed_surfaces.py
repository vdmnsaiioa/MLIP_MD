#!/usr/bin/env python
"""
match_relaxed_surfaces.py
=========================
Lattice-match every relaxed LCO slab with every relaxed LPO slab (output of
relax_lco_lpo_surfaces.py) and build one interface per match.

Matching
  The relaxed in-plane cells of each (LCO slab, LPO slab) pair are matched with the Zur-McGill
  algorithm (pymatgen ZSL). A coincidence cell is kept if the largest principal in-plane strain
  of LPO (strained onto LCO) is <= --max-strain and the interface has <= --max-atoms atoms. For
  each pair and face orientation, up to --matches-per-pair cells are kept along the size/strain
  Pareto front: the smallest cell first, then larger ones only if they lower the strain.

Faces
  A slab has two faces. With --orientations all (default) both are tried at the interface unless
  the slab is symmetric (a symmetry operation maps z -> -z). LCO "top" = the top face of its
  POSCAR_relaxed touches LPO, "bottom" = slab turned over; LPO "bottom" = its bottom face touches
  LCO, "top" = slab turned over. Turning over is a proper rotation (180 deg about x), not a mirror.

Each interface directory NNN_<LCO slab>-<face>__<LPO slab>-<face>_<atoms>at contains
  POSCAR_interface   LCO supercell + LPO supercell strained onto the LCO cell; c perpendicular to
                     the interface; --lco-vacuum below LCO, --separation between the parts,
                     --lpo-vacuum above LPO
  POSCAR_LCO         LCO part alone: same cell and positions as the interface
  POSCAR_LPO         LPO part alone in its own (unstrained, relaxed) lattice, with the same c, z and
                     fractional in-plane coordinates as in the interface: replacing its a and b by
                     those of POSCAR_interface gives exactly the strained LPO of the interface
  summary.txt/.json  mismatch, cells, supercell matrices, atoms, volume, thicknesses, dipoles and
                     the source slabs (surface energies, polarity, convergence)
interfaces.csv lists all interfaces, smallest first (column rank).

Numbering: a new --outdir is numbered by size (000 = fewest atoms). Re-running into an existing
--outdir (e.g. after adding facets) keeps the folder of every interface found before, so the
registry scans and MD inside it stay where they are; new interfaces get the next free numbers.
A re-run with another --separation, --lco-vacuum or --lpo-vacuum would change the structures in
those folders, so it needs another --outdir (or --overwrite).

Picking an interface by name (find_interface, used by mp_to_interf_relax.py and md_interface.py):
the name is matched on the slabs and faces, e.g. 001_LCO_113_s1-bot__LPO_110_s2-bot, so the number
and the atom count, which can differ between models and runs, may be left out or differ.

The default --outdir follows the surfaces directory: mh1_surfaces -> mh1_interfaces, uma_surfaces
-> uma_interfaces. An --outdir made from another surfaces directory is not overwritten unless
--overwrite is given (its interface directories may hold registry scans and MD).

Run next to lco_lpo_lattice_match.py and make_lco_lpo_terminations.py (helpers are imported).
Untested.

Examples
  python match_relaxed_surfaces.py mh1_surfaces --max-strain 0.01 --max-atoms 1000
  python match_relaxed_surfaces.py uma_surfaces --max-strain 0.01 --max-atoms 1000   # -> uma_interfaces
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from lco_lpo_lattice_match import angle_deg, int_matrix, make_zsl, strain_metrics, write_csv
from make_lco_lpo_terminations import parse_pairs, tile

R_FLIP = np.diag([1.0, -1.0, -1.0])        # proper rotation by 180 deg about x: turns a slab over
Z = np.array([0.0, 0.0, 1.0])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lattice matching and interface building for relaxed LCO / LPO slabs (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("surfaces_dir", nargs="?", default="mh1_surfaces",
                   help="output directory of relax_lco_lpo_surfaces.py")
    p.add_argument("--max-strain", type=float, default=0.01,
                   help="largest principal in-plane strain of LPO (0.01 = 1 per cent)")
    p.add_argument("--max-atoms", type=int, default=1000, help="largest interface (atoms)")
    p.add_argument("--matches-per-pair", type=int, default=2,
                   help="cells per slab pair and orientation (Pareto front in atoms and strain)")
    p.add_argument("--orientations", choices=["all", "as-is"], default="all",
                   help="try both faces of asymmetric slabs, or only the slabs as relaxed")
    p.add_argument("--separation", type=float, default=2.0, help="top LCO atom to bottom LPO atom (A)")
    p.add_argument("--lco-vacuum", type=float, default=10.0, help="vacuum below LCO (A)")
    p.add_argument("--lpo-vacuum", type=float, default=10.0, help="vacuum above LPO (A)")
    p.add_argument("--charges", nargs="+", default=["Li:1", "Co:3", "P:5", "O:-2"],
                   help="formal charges for the reported dipoles")
    p.add_argument("--symprec", type=float, default=0.1, help="tolerance (A) for detecting symmetric slabs")
    p.add_argument("--include-unconverged", action="store_true", help="also use slabs whose relaxation did not converge")
    p.add_argument("--max-interfaces", type=int, default=0, help="write only the N smallest interfaces (0 = all)")
    p.add_argument("--outdir", default=None,
                   help="output directory (default: the surfaces directory name with 'surfaces' -> 'interfaces')")
    p.add_argument("--overwrite", action="store_true", help="allow an --outdir made from another surfaces directory")
    args = p.parse_args(argv)
    if args.outdir is None:
        src = Path(args.surfaces_dir).resolve()
        name = src.name[: -len("surfaces")] + "interfaces" if src.name.endswith("surfaces") else src.name + "_interfaces"
        args.outdir = str(src.parent / name)
    return args


# =============================================================================
# Relaxed slabs
# =============================================================================
@dataclass
class Slab:
    name: str
    info: dict
    species: list
    pos: np.ndarray          # Cartesian positions, c along z
    mesh: np.ndarray         # (2, 3) relaxed in-plane cell vectors (z = 0)
    n: int
    area: float
    flips: list              # 0 = as relaxed, 1 = turned over


def symmetric(atoms, symprec: float) -> bool:
    """True if a symmetry operation of the slab maps z -> -z, i.e. both faces are equivalent."""
    try:
        ops = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms),
                                 symprec=symprec).get_symmetry_operations(cartesian=True)
    except Exception:
        return False
    return any(op.rotation_matrix[2, 2] < -0.5 for op in ops)


def load_slabs(root: Path, args) -> dict:
    slabs, models = {"LCO": [], "LPO": []}, set()
    for d in sorted((root / "surfaces").iterdir()):
        info_f, pos_f = d / "info.json", d / "POSCAR_relaxed"
        if not (info_f.exists() and pos_f.exists()):
            continue
        info = json.loads(info_f.read_text())
        models.add((str(info.get("model")), str(info.get("head"))))
        if not info.get("converged", False) and not args.include_unconverged:
            print(f"  {d.name}: relaxation not converged - not used")
            continue
        atoms = read(str(pos_f), format="vasp")
        cell = np.array(atoms.cell)
        if np.abs(cell[:2, 2]).max() > 1e-6 or np.linalg.norm(cell[2, :2]) > 1e-6:
            raise RuntimeError(f"{d.name}: expected a, b in the xy plane and c along z")
        mesh = cell[:2].copy()
        flips = [0] if args.orientations == "as-is" or symmetric(atoms, args.symprec) else [0, 1]
        slabs[info["material"]].append(Slab(
            name=d.name, info=info, species=atoms.get_chemical_symbols(), pos=atoms.positions.copy(),
            mesh=mesh, n=len(atoms), area=float(np.linalg.norm(np.cross(mesh[0], mesh[1]))), flips=flips))
    if len(models) > 1:
        print("  ! the slabs come from more than one model: " + "; ".join(f"{m} ({h})" for m, h in sorted(models)))
    return slabs


def check_outdir(outdir: Path, args) -> None:
    """Do not overwrite interfaces built from another surfaces directory or with another geometry
    (unless --overwrite)."""
    f = outdir / "run_args.json"
    if not f.exists() or args.overwrite:
        return
    old = json.loads(f.read_text())
    if Path(old.get("surfaces_dir", "")).name != Path(args.surfaces_dir).name:
        raise SystemExit(f"{outdir} was made from {old.get('surfaces_dir')}; use another --outdir or --overwrite")
    changed = [f"{k} {old[k]} -> {getattr(args, k)}" for k in ("separation", "lco_vacuum", "lpo_vacuum")
               if k in old and old[k] != getattr(args, k)]
    if changed:
        raise SystemExit(f"{outdir} was made with other interface geometry ({', '.join(changed)}); its folders "
                         f"(and any scans in them) would no longer fit: use another --outdir or --overwrite")


INTERFACE_NAME = re.compile(r"^(?:(?P<num>\d{3})_)?(?P<core>.+?)(?:_(?P<atoms>\d+)at)?$")


def split_interface_name(name: str) -> tuple[str | None, str, int | None]:
    """'001_LCO_113_s1-bot__LPO_110_s2-bot_432at' -> ('001', 'LCO_113_s1-bot__LPO_110_s2-bot', 432);
    the number and the atom count are optional."""
    m = INTERFACE_NAME.match(Path(str(name).rstrip("/")).name)
    return m["num"], m["core"], (int(m["atoms"]) if m["atoms"] else None)


def find_interface(interfaces_dir: Path, spec: str | None = None) -> tuple[Path, str]:
    """An interface folder of interfaces_dir (listed in its interfaces.csv) and why it was chosen.
    No spec: the smallest lattice mismatch (largest principal strain), fewest atoms on ties.
    A spec is matched on the slabs and faces; when several cells share them, a given atom count,
    then a given number, decides, else the smallest mismatch."""
    table = Path(interfaces_dir) / "interfaces.csv"
    if not table.exists():
        raise SystemExit(f"{table} not found - run the lattice matching first")
    with open(table, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (table.parent / r["directory"]).is_dir()]
    if not rows:
        raise SystemExit(f"no interfaces in {table}")

    def best(rs):
        return min(rs, key=lambda r: (float(r["max_strain_pct"]), int(r["n_atoms"])))

    if not spec:
        r = best(rows)
        return table.parent / r["directory"], (f"smallest lattice mismatch: {r['max_strain_pct']} %, "
                                              f"{r['n_atoms']} atoms")
    num, core, atoms = split_interface_name(spec)
    parts = {r["directory"]: split_interface_name(r["directory"]) for r in rows}
    cands = [r for r in rows if parts[r["directory"]][1] == core]
    if not cands:
        known = sorted({f"{v[1]} ({k})" for k, v in parts.items()})
        raise SystemExit(f"no interface {core} in {table}; there are:\n  " + "\n  ".join(known))
    for value, i in ((atoms, 2), (num, 0)):
        sub = [r for r in cands if value is not None and parts[r["directory"]][i] == value]
        if sub:
            cands = sub
    r = best(cands)
    return table.parent / r["directory"], (f"{spec} matched on slabs and faces: {r['max_strain_pct']} %, "
                                          f"{r['n_atoms']} atoms")


# =============================================================================
# Matching
# =============================================================================
def det2(v) -> float:
    return float(v[0][0] * v[1][1] - v[0][1] * v[1][0])


def find_matches(s: Slab, f: Slab, args) -> list[dict]:
    """Coincidence cells of one slab pair within the strain and size limits, Pareto-selected for
    every face orientation. Turning one slab over reverses the handedness of its in-plane basis
    seen from above, so for orientation (flip_s, flip_f) the map F -> S is a proper (unmirrored)
    deformation only if sign(det F) * sign(det S) = (-1)^(flip_s + flip_f)."""
    rho = s.n / s.area + f.n / f.area                       # interface atoms per A^2
    zsl = make_zsl(f.area, s.area, 1.03 * args.max_atoms / rho + 1.0, args.max_strain)
    cands, seen = [], set()
    for m in zsl(f.mesh, s.mesh, lowest=False):
        F = np.asarray(m.film_sl_vectors, dtype=float)
        S = np.asarray(m.substrate_sl_vectors, dtype=float)
        met = strain_metrics(F[0], F[1], S[0], S[1])
        if met["max_abs"] > args.max_strain:
            continue
        n_s, n_f = round(met["area_s"] / s.area), round(met["area_f"] / f.area)
        n_atoms = n_s * s.n + n_f * f.n
        if n_atoms > args.max_atoms:
            continue
        sigma = 1 if det2(S) * det2(F) > 0 else -1
        key = (sigma, n_s, n_f, round(met["max_abs"], 6), round(met["gamma_s"], 2), round(met["gamma_f"], 2))
        if key not in seen:
            seen.add(key)
            cands.append(dict(F=F, S=S, met=met, n_s=n_s, n_f=n_f, n_atoms=n_atoms, sigma=sigma))
    out = []
    for flip_s in s.flips:
        for flip_f in f.flips:
            need, best, k = (-1) ** (flip_s + flip_f), math.inf, 0
            for c in sorted((c for c in cands if c["sigma"] == need),
                            key=lambda c: (c["n_atoms"], c["met"]["max_abs"])):
                if c["met"]["max_abs"] < best - 1e-9:
                    out.append(dict(c, flip_s=flip_s, flip_f=flip_f, k=k, s=s, f=f))
                    best, k = c["met"]["max_abs"], k + 1
                    if k == args.matches_per_pair:
                        break
    return out


# =============================================================================
# Interface construction
# =============================================================================
def rot_to_x(v) -> np.ndarray:
    """Rotation about z that brings the in-plane vector v onto +x."""
    phi = math.atan2(v[1], v[0])
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def build_interface(m: dict, args) -> dict:
    """Interface (LCO cell), LCO alone (same cell) and LPO alone (own lattice, same c, z and fractional
    in-plane coordinates). All cells have c along z, perpendicular to the interface."""
    s, f = m["s"], m["f"]
    S, F = m["S"].copy(), m["F"].copy()
    ps, ms, pf, mf = s.pos, s.mesh, f.pos, f.mesh
    if m["flip_s"]:
        ps, ms, S = ps @ R_FLIP.T, ms @ R_FLIP.T, S @ R_FLIP.T
    if m["flip_f"]:
        pf, mf, F = pf @ R_FLIP.T, mf @ R_FLIP.T, F @ R_FLIP.T
    if np.cross(S[0], S[1])[2] < 0:         # right-handed LCO cell; flip F2 too so the pairing is kept
        S[1], F[1] = -S[1], -F[1]
    sub_sp, sub = tile(s.species, ps, Z, ms, int_matrix(S, ms))
    film_sp, film = tile(f.species, pf, Z, mf, int_matrix(F, mf))
    rot_s, rot_f = rot_to_x(S[0]), rot_to_x(F[0])
    s2d, f2d = (S @ rot_s.T)[:, :2], (F @ rot_f.T)[:, :2]
    sub = sub @ rot_s.T
    film_own = film @ rot_f.T                                        # LPO in its own lattice
    film_int = film_own.copy()
    film_int[:, :2] = film_own[:, :2] @ np.linalg.inv(f2d) @ s2d     # strained onto the LCO cell
    sub[:, 2] += args.lco_vacuum - sub[:, 2].min()
    dz = sub[:, 2].max() + args.separation - film_own[:, 2].min()
    film_own[:, 2] += dz
    film_int[:, 2] += dz
    c_len = film_int[:, 2].max() + args.lpo_vacuum
    lat_s = Lattice([[s2d[0, 0], s2d[0, 1], 0.0], [s2d[1, 0], s2d[1, 1], 0.0], [0.0, 0.0, c_len]])
    lat_f = Lattice([[f2d[0, 0], f2d[0, 1], 0.0], [f2d[1, 0], f2d[1, 1], 0.0], [0.0, 0.0, c_len]])

    def make(lat, species, xyz) -> Structure:
        return Structure(lat, list(species), xyz, coords_are_cartesian=True, to_unit_cell=True)

    return dict(interface=make(lat_s, sub_sp + film_sp, np.vstack([sub, film_int])),
                lco=make(lat_s, sub_sp, sub), lpo=make(lat_f, film_sp, film_own))


# =============================================================================
# Summary and main
# =============================================================================
def cell2d(v) -> dict:
    a, b = np.asarray(v[0], dtype=float), np.asarray(v[1], dtype=float)
    return dict(a=round(float(np.linalg.norm(a)), 4), b=round(float(np.linalg.norm(b)), 4),
                gamma=round(float(angle_deg(a, b)), 3), area=round(float(np.linalg.norm(np.cross(a, b))), 3))


def dipole(struct: Structure, charges: dict) -> float:
    """Formal-charge dipole along z per area (e/A)."""
    q = np.array([charges.get(site.specie.symbol, 0.0) for site in struct])
    z = struct.cart_coords[:, 2]
    return float(q @ (z - z.mean())) / (struct.lattice.volume / struct.lattice.c)


def slab_line(x: Slab, face: str) -> str:
    i = x.info
    return (f"  {x.name}: ({i.get('hkl')}), {face} face at the interface; E_surf {i.get('surface_energy_J_m2')} J/m^2 "
            f"(unrelaxed {i.get('surface_energy_unrelaxed_J_m2')}); polar {i.get('polar')}; "
            f"{str(i.get('model', '')).split('/')[-1]} head {i.get('head')}; converged {i.get('converged')}")


def summarise(rank: int, name: str, m: dict, b: dict, charges: dict, args) -> tuple[dict, list[str]]:
    s, f, met = m["s"], m["f"], m["met"]
    inter, lco, lpo = b["interface"], b["lco"], b["lpo"]
    lat = inter.lattice
    area = lat.volume / lat.c
    zs, zf = lco.cart_coords[:, 2], lpo.cart_coords[:, 2]
    face_s = "top" if m["flip_s"] == 0 else "bottom"
    face_f = "bottom" if m["flip_f"] == 0 else "top"
    data = dict(
        rank=rank, directory=name,
        lco_slab=s.name, lco_hkl=s.info.get("hkl"), lco_face=face_s,
        lpo_slab=f.name, lpo_hkl=f.info.get("hkl"), lpo_face=face_f,
        n_atoms=len(inter), n_lco=len(lco), n_lpo=len(lpo),
        lco_formula=lco.composition.formula.replace(" ", ""), lpo_formula=lpo.composition.formula.replace(" ", ""),
        max_strain_pct=round(100 * met["max_abs"], 4), e1_pct=round(100 * met["e1"], 4), e2_pct=round(100 * met["e2"], 4),
        eps_a_pct=round(100 * met["eps_a"], 4), eps_b_pct=round(100 * met["eps_b"], 4),
        dgamma_deg=round(met["dgamma"], 4), area_misfit_pct=round(100 * met["area_mis"], 4),
        a=round(lat.a, 4), b=round(lat.b, 4), gamma=round(lat.gamma, 3), c=round(lat.c, 4),
        area_A2=round(area, 3), volume_A3=round(lat.volume, 3),
        slab_volume_A3=round(area * float(zf.max() - zs.min()), 3),
        lco_thickness=round(float(np.ptp(zs)), 3), lpo_thickness=round(float(np.ptp(zf)), 3),
        separation=args.separation, lco_vacuum=args.lco_vacuum, lpo_vacuum=args.lpo_vacuum,
        n_lco_cells=m["n_s"], n_lpo_cells=m["n_f"],
        T_lco=int_matrix(m["S"], s.mesh).tolist(), T_lpo=int_matrix(m["F"], f.mesh).tolist(),
        lco_surface_cell=cell2d(s.mesh), lpo_surface_cell=cell2d(f.mesh),
        lco_supercell=cell2d(m["S"]), lpo_supercell_unstrained=cell2d(m["F"]),
        dipole_lco=round(dipole(lco, charges), 5), dipole_lpo=round(dipole(lpo, charges), 5),
        dipole_interface=round(dipole(inter, charges), 5),
        lco_surface_energy_J_m2=s.info.get("surface_energy_J_m2"),
        lpo_surface_energy_J_m2=f.info.get("surface_energy_J_m2"),
        lco_slab_polar=s.info.get("polar"), lpo_slab_polar=f.info.get("polar"),
    )
    sc, fc = data["lco_surface_cell"], data["lpo_surface_cell"]
    ss, fs = data["lco_supercell"], data["lpo_supercell_unstrained"]
    lines = [
        f"Interface {name.split('_', 1)[0]}: {s.name} ({face_s} face) | {f.name} ({face_f} face); "
        f"rank {rank} by size in this run",
        f"  LCO ({data['lco_hkl']}) at the bottom, LPO ({data['lpo_hkl']}) on top; "
        f"{data['n_atoms']} atoms = {data['n_lco']} LCO + {data['n_lpo']} LPO",
        "",
        "Files",
        "  POSCAR_interface  LCO + LPO, LPO strained onto the LCO cell; c perpendicular to the interface",
        "  POSCAR_LCO        LCO part alone (same cell and positions as the interface)",
        "  POSCAR_LPO        LPO part alone in its own relaxed lattice; same c, z and fractional in-plane",
        "                    coordinates: setting its a, b to those of POSCAR_interface gives the strained LPO",
        "",
        "Lattice mismatch (LPO strained onto LCO; + = tensile in LPO)",
        f"  principal strains  {data['e1_pct']:+.3f} %, {data['e2_pct']:+.3f} %   max |e| = {data['max_strain_pct']:.3f} %",
        f"  along a / b        {data['eps_a_pct']:+.3f} % / {data['eps_b_pct']:+.3f} %",
        f"  angle              {fs['gamma']:.3f} -> {ss['gamma']:.3f} deg ({data['dgamma_deg']:+.3f} deg)",
        f"  area               {fs['area']:.3f} -> {ss['area']:.3f} A^2 ({data['area_misfit_pct']:+.3f} %)",
        "",
        "In-plane cells before matching (from the relaxed slabs; A, deg, A^2)",
        f"  LCO surface cell   a {sc['a']:.4f}  b {sc['b']:.4f}  gamma {sc['gamma']:.3f}  area {sc['area']:.3f}"
        f"  ({s.n} atoms)",
        f"  LPO surface cell   a {fc['a']:.4f}  b {fc['b']:.4f}  gamma {fc['gamma']:.3f}  area {fc['area']:.3f}"
        f"  ({f.n} atoms)",
        f"  LCO supercell      a {ss['a']:.4f}  b {ss['b']:.4f}  gamma {ss['gamma']:.3f}  area {ss['area']:.3f}"
        f"  = {m['n_s']} surface cells, T = {data['T_lco']}",
        f"  LPO supercell      a {fs['a']:.4f}  b {fs['b']:.4f}  gamma {fs['gamma']:.3f}  area {fs['area']:.3f}"
        f"  = {m['n_f']} surface cells, T = {data['T_lpo']}",
        "  (T rows = supercell vectors in units of a, b of the slab's POSCAR_relaxed)",
        "",
        "Interface cell",
        f"  a {lat.a:.4f}  b {lat.b:.4f}  gamma {lat.gamma:.3f} deg  c {lat.c:.4f} A (perpendicular)",
        f"  area {area:.3f} A^2, volume {lat.volume:.2f} A^3 (slab region {data['slab_volume_A3']:.2f} A^3)",
        f"  LCO {data['lco_formula']}, {data['lco_thickness']:.2f} A | separation {args.separation} A | "
        f"LPO {data['lpo_formula']}, {data['lpo_thickness']:.2f} A",
        f"  vacuum {args.lco_vacuum} A below LCO, {args.lpo_vacuum} A above LPO",
        f"  formal-charge dipole / area: LCO {data['dipole_lco']:+.4f}, LPO {data['dipole_lpo']:+.4f}, "
        f"interface {data['dipole_interface']:+.4f} e/A",
        "",
        "Source slabs (relax_lco_lpo_surfaces.py)",
        slab_line(s, face_s),
        slab_line(f, face_f),
    ]
    return data, lines


def main(argv=None) -> None:
    args = parse_args(argv)
    charges = parse_pairs(args.charges, "")
    slabs = load_slabs(Path(args.surfaces_dir), args)
    n_asym = sum(len(x.flips) > 1 for v in slabs.values() for x in v)
    print(f"{len(slabs['LCO'])} LCO and {len(slabs['LPO'])} LPO slabs ({n_asym} asymmetric: both faces tried)")

    found = []
    for s in slabs["LCO"]:
        for f in slabs["LPO"]:
            ms = find_matches(s, f, args)
            found += ms
            smallest = min((x["n_atoms"] for x in ms), default=None)
            print(f"  {s.name:<13} x {f.name:<13}: {len(ms)} interface(s)"
                  + (f", smallest {smallest} atoms" if ms else ""))
    found.sort(key=lambda x: (x["n_atoms"], x["met"]["max_abs"]))
    if args.max_interfaces:
        found = found[: args.max_interfaces]

    outdir = Path(args.outdir)
    check_outdir(outdir, args)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    # folders of a previous run, by interface: they keep their number (scans and MD may live in them)
    existing = {d.name[4:]: d.name for d in outdir.glob("[0-9][0-9][0-9]_*") if d.is_dir()}
    next_id = 1 + max((int(n[:3]) for n in existing.values()), default=-1)
    rows = []
    for rank, m in enumerate(found):
        face_s = "top" if m["flip_s"] == 0 else "bot"
        face_f = "bot" if m["flip_f"] == 0 else "top"
        tag = f"{m['s'].name}-{face_s}__{m['f'].name}-{face_f}_{m['n_atoms']}at"
        if tag in existing:
            name = existing[tag]
        else:
            name, next_id = f"{next_id:03d}_{tag}", next_id + 1
        d = outdir / name
        d.mkdir(exist_ok=True)
        b = build_interface(m, args)
        for key, fname in (("interface", "POSCAR_interface"), ("lco", "POSCAR_LCO"), ("lpo", "POSCAR_LPO")):
            b[key].get_sorted_structure().to(filename=str(d / fname), fmt="poscar")
        data, lines = summarise(rank, name, m, b, charges, args)
        (d / "summary.txt").write_text("\n".join(lines) + "\n")
        (d / "summary.json").write_text(json.dumps(data, indent=2))
        rows.append({k: v for k, v in data.items() if not isinstance(v, dict)})
    write_csv(rows, outdir / "interfaces.csv")
    n_small = sum(r["n_atoms"] <= 300 for r in rows)
    print(f"\n{len(rows)} interfaces written to {outdir}/ ({n_small} with <= 300 atoms); overview in interfaces.csv")


if __name__ == "__main__":
    main()

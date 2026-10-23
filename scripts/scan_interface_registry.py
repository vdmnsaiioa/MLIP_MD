#!/usr/bin/env python
"""
scan_interface_registry.py
==========================
Registry (lateral shift) and gap scan of one interface written by match_relaxed_surfaces.py.

Everything is generated from the existing POSCARs of the interface directory: LCO stays where it
is, the LPO block is shifted in-plane over a grid of shifts and moved along z to every gap in
--gaps. Each (shift, gap) combination is written as its own POSCAR, ready for
relax_interface_scan.py.

Which shifts are distinct
  Moving LPO by one of its own in-plane lattice vectors gives the same structure, and moving it by
  an LCO lattice vector gives the same structure translated. So the energy is periodic in the
  shift with the combined lattice L = L_LCO + L_LPO, and only shifts inside one cell of L are
  distinct. E.g. with 5 LCO and 3 LPO surface cells in the interface cell (interface 001), L is 15
  times denser than the interface lattice and the shift cell is 1/15 of the interface area. Both
  translation sets are found directly from the POSCAR, so no supercell matrices are needed.
    --shift-cell minimal    (default) one cell of L
    --shift-cell lpo        the LPO surface cell (strained)
    --shift-cell interface  the whole interface cell (no reduction)
  The grid has N1 x N2 points along the two Gauss-reduced shift-cell vectors and starts at zero
  shift, i.e. the registry of the original POSCAR_interface.

Gap
  gap = z(lowest LPO atom) - z(highest LCO atom), as --separation in match_relaxed_surfaces.py; the
  current value is read from POSCAR_interface. For every gap the LPO block moves along z and c changes
  by the same amount, so the vacuum below LCO and above LPO stays as it was.

Input (interface directory)
  POSCAR_interface   the structure that is shifted
  POSCAR_LCO         tells the LCO atoms from the LPO atoms (everything above its top atom is LPO)
  summary.json       optional, only used for consistency checks and labels

Output (--outdir, default <interface dir>/registry_scan)
  sII_JJ_gG.GG/POSCAR   one structure per shift (grid indices II, JJ) and gap, same atom order as
                        POSCAR_interface
  reference/POSCAR_LCO  LCO alone and strained LPO alone in the interface cell: references for the
  reference/POSCAR_LPO  work of adhesion in relax_interface_scan.py (shift and gap do not matter)
  scan.csv              per structure: shift (A and fractions), gap, c, shortest LCO-LPO distance
  scan_info.json        shift cell, grid, gaps, LPO atom indices, original gap, area, source files

Requirements: ase, numpy. Untested on real data.

Example
  python scan_interface_registry.py mh1_interfaces/001_LCO_113_s1-bot__LPO_110_s2-bot_432at \
      --grid 6 6 --gaps 1.5 2.0 2.5 3.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from ase.io import read, write


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Registry and gap scan of one LCO | LPO interface (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("interface_dir", help="an interface directory written by match_relaxed_surfaces.py")
    p.add_argument("--grid", type=int, nargs=2, default=[6, 6], metavar=("N1", "N2"),
                   help="shift grid points along the two shift-cell vectors (1 1 = gap scan only)")
    p.add_argument("--gaps", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0],
                   help="gaps to scan (A): lowest LPO atom to highest LCO atom along z")
    p.add_argument("--shift-cell", choices=["minimal", "lpo", "interface"], default="minimal",
                   help="cell the shift grid spans (see module docstring)")
    p.add_argument("--min-contact", type=float, default=0.0,
                   help="skip structures whose shortest LCO-LPO distance is below this (A); 0 keeps all")
    p.add_argument("--tol", type=float, default=0.01,
                   help="position tolerance (A) when detecting the in-plane translations")
    p.add_argument("--outdir", default=None, help="output directory (default <interface_dir>/registry_scan)")
    return p.parse_args(argv)


# =============================================================================
# In-plane lattices of the two parts
# =============================================================================
def translations(symbols, pos: np.ndarray, A2: np.ndarray, tol: float) -> np.ndarray:
    """In-plane translations that map a part (all its atoms) onto itself, as fractions of the
    interface cell vectors a, b (modulo 1). Candidates: vectors from one atom of the rarest
    species to every atom of that species at the same height; each is checked on all atoms."""
    symbols = np.asarray(symbols)
    frac = pos[:, :2] @ np.linalg.inv(A2)
    z = pos[:, 2]
    groups = [np.nonzero(symbols == s)[0] for s in sorted(set(symbols.tolist()))]
    ref = min(groups, key=len)
    r = ref[0]
    found = []
    for j in ref:
        if abs(z[j] - z[r]) > tol:
            continue
        d = frac[j] - frac[r]
        d -= np.round(d)
        if all(maps_onto_itself(frac[g], z[g], d, A2, tol) for g in groups):
            found.append(d)
    if not found:
        raise RuntimeError("no translation found, not even zero - check the input files")
    return np.array(found)


def maps_onto_itself(frac: np.ndarray, z: np.ndarray, d: np.ndarray, A2: np.ndarray, tol: float) -> bool:
    """True if shifting all atoms (one species) by the fractional in-plane vector d lands every
    atom on an atom of the same group (periodic in-plane, same height)."""
    diff = frac[:, None, :] + d - frac[None, :, :]
    diff -= np.round(diff)
    d2 = ((diff @ A2) ** 2).sum(-1) + (z[:, None] - z[None, :]) ** 2
    return bool(np.all(d2.min(axis=1) < tol ** 2))


def common_denominator(vectors, max_den: int = 1000) -> int:
    n = 1
    for v in vectors:
        for x in v:
            f = Fraction(float(x)).limit_denominator(max_den)
            if abs(float(f) - x) > 1e-5:
                raise RuntimeError(f"translation component {x} is not a simple fraction - raise --tol?")
            n = n * f.denominator // math.gcd(n, f.denominator)
    return n


def int_basis_2d(vectors) -> np.ndarray:
    """Basis (rows) of the 2D integer lattice spanned by integer vectors (full rank assumed)."""
    b1 = np.zeros(2, dtype=np.int64)
    ys = []
    for v in vectors:
        v = np.asarray(v, dtype=np.int64)
        if v[0] == 0:
            ys.append(int(v[1]))
        elif b1[0] == 0:
            b1 = v.copy()
        else:
            g, x, y = egcd(int(b1[0]), int(v[0]))
            new = x * b1 + y * v                                   # first component = gcd
            ys.append(int(((v[0] // g) * b1 - (b1[0] // g) * v)[1]))  # first component 0
            b1 = new
    h = 0
    for y in ys:
        h = math.gcd(h, abs(y))
    if b1[0] == 0 or h == 0:
        raise RuntimeError("translation lattice is not two-dimensional")
    if b1[0] < 0:
        b1 = -b1
    b1[1] %= h
    return np.array([b1, [0, h]], dtype=np.int64)


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """g, x, y with a x + b y = g = gcd(a, b) > 0."""
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        a, x0, y0 = -a, -x0, -y0
    return a, x0, y0


def lattice_from_translations(*sets) -> np.ndarray:
    """Fractional basis (rows, in units of a, b) of the lattice generated by the interface cell
    vectors and all given translation sets."""
    vecs = np.vstack(sets)
    n = common_denominator(vecs)
    ints = [np.rint(n * v).astype(np.int64) for v in vecs] + [np.array([n, 0]), np.array([0, n])]
    return int_basis_2d(ints) / n


def gauss_reduce(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shortest basis of the same 2D lattice, angle between 60 and 90 deg."""
    u, v = np.array(u, dtype=float), np.array(v, dtype=float)
    while True:
        if u @ u > v @ v:
            u, v = v, u
        m = round(float(u @ v) / float(u @ u))
        if m == 0:
            break
        v = v - m * u
    if u @ v < 0:
        v = -v
    return u, v


# =============================================================================
# Geometry helpers
# =============================================================================
def shortest_contact(pa, sa, pb, sb, A2, n_img: int = 2) -> tuple[float, str]:
    """Shortest distance between a part-A atom and a part-B atom (periodic in-plane) and the pair."""
    best, pair = math.inf, ""
    for i in range(-n_img, n_img + 1):
        for j in range(-n_img, n_img + 1):
            t = i * A2[0] + j * A2[1]
            d = pa[:, None, :] - (pb[None, :, :] + np.array([t[0], t[1], 0.0]))
            r = np.sqrt((d ** 2).sum(-1))
            k = np.unravel_index(np.argmin(r), r.shape)
            if r[k] < best:
                best, pair = float(r[k]), f"{sa[k[0]]}-{sb[k[1]]}"
    return best, pair


def angle_deg(u, v) -> float:
    return math.degrees(math.acos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> None:
    args = parse_args(argv)
    src = Path(args.interface_dir)
    out = Path(args.outdir) if args.outdir else src / "registry_scan"
    inter = read(str(src / "POSCAR_interface"), format="vasp")
    lco_ref = read(str(src / "POSCAR_LCO"), format="vasp")
    summary = json.loads((src / "summary.json").read_text()) if (src / "summary.json").exists() else {}

    cell = np.array(inter.cell)
    if np.abs(cell[:2, 2]).max() > 1e-6 or np.linalg.norm(cell[2, :2]) > 1e-6:
        raise RuntimeError("expected a, b in the xy plane and c along z (as written by match_relaxed_surfaces.py)")
    A2 = cell[:2, :2]
    area = abs(float(np.linalg.det(A2)))
    pos = inter.positions.copy()
    sym = np.array(inter.get_chemical_symbols())

    # ---- LCO / LPO atoms: everything above the top LCO atom is LPO --------------------------
    z_top = float(lco_ref.positions[:, 2].max())
    is_lpo = pos[:, 2] > z_top + 0.1
    n_lco, n_lpo = int((~is_lpo).sum()), int(is_lpo.sum())
    if n_lco != len(lco_ref) or sorted(sym[~is_lpo]) != sorted(lco_ref.get_chemical_symbols()):
        raise RuntimeError(f"could not split POSCAR_interface: {n_lco} atoms below the top of POSCAR_LCO, "
                           f"POSCAR_LCO has {len(lco_ref)}")
    for key, val in (("n_lco", n_lco), ("n_lpo", n_lpo)):
        if key in summary and summary[key] != val:
            print(f"warning: summary.json says {key} = {summary[key]}, found {val}")
    gap0 = float(pos[is_lpo, 2].min() - pos[~is_lpo, 2].max())

    # ---- shift cell -------------------------------------------------------------------------
    t_lco = translations(sym[~is_lpo], pos[~is_lpo], A2, args.tol)
    t_lpo = translations(sym[is_lpo], pos[is_lpo], A2, args.tol)
    for key, val in (("n_lco_cells", len(t_lco)), ("n_lpo_cells", len(t_lpo))):
        if key in summary and summary[key] != val:
            print(f"warning: summary.json says {key} = {summary[key]}, found {val} in-plane translations")
    zero = np.zeros((1, 2))
    if args.shift_cell == "minimal":
        basis = lattice_from_translations(zero, t_lco, t_lpo)
    elif args.shift_cell == "lpo":
        basis = lattice_from_translations(zero, t_lpo)
    else:
        basis = np.eye(2)
    n_cells = int(round(1.0 / abs(np.linalg.det(basis))))       # shift cells per interface cell
    u1, u2 = basis @ A2
    if args.shift_cell != "interface":
        u1, u2 = gauss_reduce(u1, u2)
    n1, n2 = args.grid
    shift_area = abs(float(u1[0] * u2[1] - u1[1] * u2[0]))
    print(f"{src.name}: {len(inter)} atoms ({n_lco} LCO, {n_lpo} LPO), area {area:.2f} A^2, gap {gap0:.3f} A")
    print(f"  in-plane translations: {len(t_lco)} (LCO), {len(t_lpo)} (LPO) per interface cell")
    print(f"  shift cell ({args.shift_cell}): |u1| {np.linalg.norm(u1):.3f} A, |u2| {np.linalg.norm(u2):.3f} A, "
          f"angle {angle_deg(u1, u2):.1f} deg, area {shift_area:.3f} A^2 = 1/{n_cells} of the interface cell")
    print(f"  grid {n1} x {n2}: steps {np.linalg.norm(u1) / n1:.3f} A x {np.linalg.norm(u2) / n2:.3f} A; "
          f"gaps {', '.join(f'{g:g}' for g in args.gaps)} A")

    # ---- references for the work of adhesion -------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(exist_ok=True)
    write(str(out / "reference" / "POSCAR_LCO"), inter[np.nonzero(~is_lpo)[0]], format="vasp", direct=True)
    write(str(out / "reference" / "POSCAR_LPO"), inter[np.nonzero(is_lpo)[0]], format="vasp", direct=True)

    # ---- structures ---------------------------------------------------------------------------
    inv = np.linalg.inv(A2)
    near_lco = (~is_lpo) & (pos[:, 2] > z_top - 4.0)              # atoms that can touch across the gap
    rows, n_written = [], 0
    for g in args.gaps:
        dz = g - gap0
        for i in range(n1):
            for j in range(n2):
                t = (i / n1) * u1 + (j / n2) * u2
                new = pos.copy()
                xy = new[is_lpo, :2] + t
                new[is_lpo, :2] = ((xy @ inv) % 1.0) @ A2               # wrap in-plane
                new[is_lpo, 2] += dz
                atoms = inter.copy()
                atoms.positions = new
                c = atoms.cell.array.copy()
                c[2, 2] += dz                                          # same vacuum as before
                atoms.set_cell(c, scale_atoms=False)
                z_bot = new[is_lpo, 2].min()
                near_lpo = is_lpo & (new[:, 2] < z_bot + 4.0)
                dmin, pair = shortest_contact(new[near_lco], sym[near_lco], new[near_lpo], sym[near_lpo], A2)
                name = f"s{i:02d}_{j:02d}_g{g:.2f}"
                keep = dmin >= args.min_contact
                if keep:
                    (out / name).mkdir(exist_ok=True)
                    write(str(out / name / "POSCAR"), atoms, format="vasp", direct=True)
                    n_written += 1
                fa, fb = (t @ inv) % 1.0
                rows.append(dict(name=name, written=keep, i=i, j=j, gap_A=round(g, 4),
                                 shift_u1=round(i / n1, 6), shift_u2=round(j / n2, 6),
                                 shift_x_A=round(float(t[0]), 5), shift_y_A=round(float(t[1]), 5),
                                 shift_frac_a=round(float(fa), 6), shift_frac_b=round(float(fb), 6),
                                 c_A=round(float(c[2, 2]), 4), min_contact_A=round(dmin, 4), contact_pair=pair))
    write_csv(rows, out / "scan.csv")
    info = dict(
        source=str(src.resolve()), interface=summary.get("directory", src.name),
        lco_slab=summary.get("lco_slab"), lco_face=summary.get("lco_face"),
        lpo_slab=summary.get("lpo_slab"), lpo_face=summary.get("lpo_face"),
        n_atoms=len(inter), n_lco=n_lco, n_lpo=n_lpo, lpo_indices=np.nonzero(is_lpo)[0].tolist(),
        area_A2=area, cell_a=cell[0].tolist(), cell_b=cell[1].tolist(), c_original=float(cell[2, 2]),
        gap_original=gap0, grid=[n1, n2], gaps=list(args.gaps),
        shift_cell=dict(type=args.shift_cell, u1=u1.tolist(), u2=u2.tolist(), area_A2=shift_area,
                        cells_per_interface_cell=n_cells),
        n_translations_lco=len(t_lco), n_translations_lpo=len(t_lpo),
        n_structures=n_written, min_contact=args.min_contact,
    )
    (out / "scan_info.json").write_text(json.dumps(info, indent=2))
    close = [r for r in rows if r["min_contact_A"] < 1.2]
    print(f"\n{n_written} structures written to {out}/ (scan.csv, scan_info.json, reference/)")
    if close:
        print(f"  {len(close)} have an LCO-LPO contact below 1.2 A (shortest {min(r['min_contact_A'] for r in close):.2f} A); "
              f"see min_contact_A in scan.csv or use --min-contact")


if __name__ == "__main__":
    main()
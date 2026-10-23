#!/usr/bin/env python
"""
lco_lpo_lattice_match.py
========================
Lattice-matching screen for LiCoO2 (LCO, substrate) | Li3PO4 (LPO, film)
interfaces, starting from Materials Project bulk structures.

Pipeline
--------
1. Bulk phases from MP: by default the lowest-E_hull polymorph that has an
   ICSD match (theoretical=False); or explicit mp-ids; or your own relaxed
   structure files (see the caveat below).
2. Standard conventional cell (spglib / ITA setting), so Miller indices are in
   the usual setting (hexagonal axes for R-3m LCO, Pnma / Pmn2_1 axes for LPO).
3. "Reasonable" facets: every symmetry-distinct (hkl) with max(|h|,|k|,|l|) <=
   --max-index, ranked by the Bravais-Friedel-Donnay-Harker rule (largest
   effective interplanar spacing first = most important growth/cleavage
   faces): d_hkl is divided by the order n of the first reflection n(hkl) that
   is not systematically absent (lattice centring, screw axes, glide planes).
   The top --n-facets are kept. With centring included, LCO (001) gets d = c/3
   and (104) ranks among the lowest-index planes, as it should.
   --ranking lattice uses centring only (pure lattice-plane spacing);
   --substrate-millers / --film-millers screen your own list instead.
4. True primitive 2D surface cell of each facet, built from the primitive
   lattice. (Building it from the conventional R-3m cell would hide 2/3 of the
   in-plane lattice points of LCO (104) and miss the smallest coincidence cells.)
5. For every (substrate facet, film facet) pair: coincident super-lattices up to
   --max-area from pymatgen's Zur-McGill (ZSL) generator, used as a loose
   pre-filter, followed by the exact misfit below. Cells with
   max|principal strain| <= --max-strain (default 0.01 = 1 %) are kept.
6. Output (in --outdir): facets.csv, matches.csv (up to --matches-per-pair cells
   per facet pair), best_matches.csv (best cell per facet pair; smallest area
   by default, see --sort), the conventional bulk cells used
   (structures/*.cif) and run_args.json.
   build_lco_lpo_interfaces.py turns rows of these CSVs into interface POSCARs.

Misfit definition (the film is strained onto the substrate)
------------------------------------------------------------
Matched super-cell vectors (f1, f2) -> (s1, s2) define a 2D deformation F with
F f_i = s_i. Its principal stretches lambda_1,2 solve  G_s v = lambda^2 G_f v
(G = 2x2 Gram matrices); the principal strains are e_i = lambda_i - 1
(positive = tensile in the film) and the criterion is max|e_i|. This is
basis-independent and bounds the strain along *any* in-plane direction, so the
per-vector misfits |s_i|/|f_i| - 1 (also reported, with the angle misfit) are
automatically below it. As in pymatgen's ZSL, cells are matched up to a mirror
of the film mesh, i.e. using either the (hkl) or the (-h -k -l) face of the film.

Caveat: 1 % is comparable to the lattice-constant error of GGA(+U). For
production runs, feed bulk cells relaxed with the functional you will use for
the interface (--substrate-file / --film-file), or experimental cells.

Requirements: pymatgen (2023 or newer), mp-api, numpy, scipy.
MP API key:   --api-key, or the MP_API_KEY environment variable.

Examples
--------
  python lco_lpo_lattice_match.py
  python lco_lpo_lattice_match.py --film-polymorphs 2        # two lowest-E Li3PO4 polymorphs
  python lco_lpo_lattice_match.py --substrate-ids mp-22526 --max-area 300
  python lco_lpo_lattice_match.py --substrate-millers 001 104 012 --film-millers 100 010 001
  python lco_lpo_lattice_match.py --substrate-file LCO_relaxed.vasp --film-file LPO_relaxed.vasp
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.core import Structure
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

MP_FIELDS = ["material_id", "formula_pretty", "symmetry", "energy_above_hull", "structure", "theoretical"]

SORT_KEYS = {   # ranking of coincidence cells (per facet pair and globally)
    "area": lambda r: (r["area_A2"], r["max_strain_pct"]),
    "strain": lambda r: (r["max_strain_pct"], r["area_A2"]),
    "atoms": lambda r: (r["n_atoms_est"], r["max_strain_pct"]),
}


# =============================================================================
# Small geometry / formatting helpers
# =============================================================================
def norm(v) -> float:
    return float(np.linalg.norm(v))


def area2d(u, v) -> float:
    """Area of the parallelogram spanned by two 3D vectors."""
    return norm(np.cross(u, v))


def angle_deg(u, v) -> float:
    return math.degrees(math.atan2(area2d(u, v), float(np.dot(u, v))))


def reduce_2d(a, b, tol: float = 1e-8):
    """Gauss (Zur-McGill) reduction of a 2D basis.

    Returns (a, b) with |a| <= |b| and 0 <= a.b <= |a|^2 / 2, i.e. an angle of
    60-90 deg (same convention as pymatgen's reduce_vectors)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    for _ in range(10_000):
        if a @ b < 0:
            b = -b
        if a @ a > b @ b + tol:
            a, b = b, a
            continue
        c = b - a
        if b @ b > c @ c + tol:
            b = c
            continue
        return a, b
    raise RuntimeError("2D lattice reduction did not converge")


def coprime_ints(v, max_den: int = 24) -> np.ndarray:
    """Smallest integer vector parallel to a (numerically) rational vector."""
    fr = [Fraction(float(x)).limit_denominator(max_den) for x in v]
    den = reduce(math.lcm, (f.denominator for f in fr), 1)
    ints = [int(f * den) for f in fr]
    g = reduce(math.gcd, (abs(i) for i in ints), 0)
    if g == 0:
        raise ValueError(f"zero vector {v}")
    return np.array([i // g for i in ints], dtype=int)


def fmt_hkl(hkl) -> str:
    return " ".join(str(int(x)) for x in hkl)


def fmt_dir(vec, conv: Structure) -> str:
    """Direction [uvw] (conventional basis) of a Cartesian lattice vector."""
    uvw = coprime_ints(conv.lattice.get_fractional_coords(vec))
    return "[" + " ".join(str(i) for i in uvw) + "]"


def parse_millers(tokens) -> list[tuple[int, int, int]]:
    """Accepts '104', '10-4', '1,0,4', '1 0 4' and Miller-Bravais '10-14' / '1,0,-1,4'."""
    out = []
    for tok in tokens:
        t = tok.strip().strip("()[]").strip()
        if "," in t or " " in t:
            nums = [int(x) for x in re.split(r"[,\s]+", t) if x]
        else:
            nums = [int(x) for x in re.findall(r"-?\d", t)]
        if len(nums) == 4:  # (h k i l) -> (h k l)
            h, k, i, l = nums
            if i != -(h + k):
                raise ValueError(f"inconsistent Miller-Bravais index '{tok}'")
            nums = [h, k, l]
        if len(nums) != 3 or not any(nums):
            raise ValueError(f"cannot parse Miller index '{tok}'")
        g = reduce(math.gcd, (abs(n) for n in nums))
        out.append(tuple(n // g for n in nums))
    return out


# =============================================================================
# Bulk phases
# =============================================================================
@dataclass
class Phase:
    formula: str
    mpid: str                 # MP id, or a label for local files
    spacegroup: str
    e_hull: float | None
    conv: Structure           # standard conventional cell (spglib / ITA setting)
    prim: Structure           # primitive cell, same Cartesian frame as conv
    symops: list              # space-group ops of conv as (W int, w), x' = W x + w (fractional)
    cif: str = ""             # conventional cell on disk, relative to --outdir

    @property
    def label(self) -> str:
        return f"{self.formula} {self.mpid} ({self.spacegroup})"


def make_phase(structure: Structure, mpid: str, e_hull, mp_symbol, symprec: float) -> Phase:
    clean = Structure(structure.lattice, structure.species, structure.frac_coords)  # drop magmoms etc.
    sga = SpacegroupAnalyzer(clean, symprec=symprec)
    conv = sga.get_refined_structure()          # spglib-standardised conventional cell
    spg = sga.get_space_group_symbol()
    if mp_symbol and mp_symbol != spg:
        print(f"  ! {mpid}: MP lists {mp_symbol}, spglib (symprec={symprec}) finds {spg}")
    prim = conv.get_primitive_structure()       # keeps the Cartesian frame of conv
    m = prim.lattice.matrix @ np.linalg.inv(conv.lattice.matrix)   # P = M C
    if not np.allclose(12 * m, np.round(12 * m), atol=1e-3):
        raise RuntimeError(f"{mpid}: primitive cell is not a rational sub-cell of the conventional cell")
    ops = SpacegroupAnalyzer(conv, symprec=symprec).get_symmetry_operations(cartesian=False)
    return Phase(
        formula=conv.composition.reduced_formula,
        mpid=mpid,
        spacegroup=spg,
        e_hull=e_hull,
        conv=conv,
        prim=prim,
        symops=[(np.rint(op.rotation_matrix).astype(int), np.asarray(op.translation_vector, dtype=float))
                for op in ops],
    )


def phases_from_files(paths, role: str, symprec: float) -> list[Phase]:
    phases = [make_phase(Structure.from_file(p), f"{role}-{Path(p).stem}", None, None, symprec) for p in paths]
    for ph in phases:
        print(f"{role} from file: {ph.label}")
    return phases


def phases_from_mp(mpr, formula: str, ids, n_keep: int, args, role: str) -> list[Phase]:
    try:
        summary = mpr.materials.summary         # current mp-api
    except AttributeError:                      # older mp-api
        summary = mpr.summary
    if ids:
        docs = summary.search(material_ids=list(ids), fields=MP_FIELDS)
    else:
        query = dict(formula=formula, energy_above_hull=(0.0, args.max_ehull), deprecated=False, fields=MP_FIELDS)
        if not args.include_theoretical:
            query["theoretical"] = False
        docs = summary.search(**query)
    if not docs:
        sys.exit(f"No Materials Project entries found for the {role} ({ids or formula}).")

    def ehull(doc):
        return doc.energy_above_hull if doc.energy_above_hull is not None else math.inf

    docs = sorted(docs, key=ehull)
    n_use = len(docs) if ids else n_keep
    print(f"\n{role} entries from MP ({', '.join(ids) if ids else formula}); * = screened")
    phases = []
    for i, doc in enumerate(docs):
        use = i < n_use
        print(f"  {'*' if use else ' '} {str(doc.material_id):<12} {doc.symmetry.symbol:<10} "
              f"E_hull = {ehull(doc):.3f} eV/atom   theoretical = {doc.theoretical}")
        if use:
            phases.append(make_phase(doc.structure, str(doc.material_id), doc.energy_above_hull,
                                     doc.symmetry.symbol, args.symprec))
    return phases


# =============================================================================
# Facets: true interplanar spacing and primitive 2D surface cell
# =============================================================================
@dataclass
class Facet:
    hkl: tuple                # Miller index w.r.t. the conventional cell
    hkl_prim: tuple           # the same plane w.r.t. the primitive lattice
    d: float                  # spacing of the lattice planes (centring included) (A)
    d_dh: float               # Donnay-Harker spacing (all systematic absences) (A)
    area: float               # primitive 2D surface-cell area (A^2)
    mesh: np.ndarray          # (2, 3) reduced in-plane basis, Cartesian, frame of conv


def dh_spacing(phase: Phase, hkl) -> float:
    """d_hkl / n, with n the lowest order such that n(hkl) is not systematically absent.
    A reflection h is absent if some op {W|w} has h W = h and h.w non-integer."""
    h = np.array(hkl, dtype=int)
    d0 = float(phase.conv.lattice.d_hkl(h))
    for n in range(1, 25):
        nh = n * h
        absent = False
        for rot, trans in phase.symops:
            if np.array_equal(nh @ rot, nh):
                shift = float(nh @ trans)
                if abs(shift - round(shift)) > 1e-3:
                    absent = True
                    break
        if not absent:
            return d0 / n
    return d0 / 24


def facet_geometry(phase: Phase, hkl) -> Facet:
    hkl = tuple(int(x) for x in hkl)
    C, P = phase.conv.lattice.matrix, phase.prim.lattice.matrix
    hkl_p = coprime_ints(P @ np.linalg.inv(C) @ np.array(hkl, dtype=float))  # h_prim = M h_conv
    d = float(phase.prim.lattice.d_hkl(hkl_p))      # spacing of *lattice* planes (centring included)
    cell_area = phase.prim.volume / d               # area of the primitive 2D mesh
    nmax = 2 * int(np.abs(hkl_p).max()) + 2
    for _ in range(3):
        rng = np.arange(-nmax, nmax + 1)
        uvw = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
        uvw = uvw[(uvw @ hkl_p == 0) & np.any(uvw != 0, axis=1)]   # lattice vectors lying in the plane
        cart = uvw @ P
        cart = cart[np.argsort(np.linalg.norm(cart, axis=1), kind="stable")]
        a = cart[0]                                                 # shortest in-plane vector
        for b in cart[1:]:
            if abs(area2d(a, b) - cell_area) < 1e-6 * cell_area:    # (a, b) spans the primitive mesh
                a, b = reduce_2d(a, b)
                return Facet(hkl, tuple(int(x) for x in hkl_p), d, dh_spacing(phase, hkl), cell_area,
                             np.array([a, b]))
        nmax *= 2
    raise RuntimeError(f"could not build the 2D mesh of ({fmt_hkl(hkl)}) for {phase.label}")


def canonical_hkl(hkl, symops) -> tuple[int, int, int]:
    """Nicest symmetry-equivalent index: fewest minus signs, then largest leading entries."""
    h = np.array(hkl, dtype=float)
    cands = set()
    for rot, _ in symops:
        hp = tuple(int(round(x)) for x in h @ np.linalg.inv(rot))   # Miller indices transform as h W^-1
        cands.add(hp)
        cands.add(tuple(-x for x in hp))                            # (hkl), (-h-k-l): same 2D mesh
    return min(cands, key=lambda t: (sum(x < 0 for x in t), tuple(-x for x in t)))


def select_facets(phase: Phase, max_index: int, n_facets: int, ranking: str, explicit=None) -> list[Facet]:
    if explicit:
        return [facet_geometry(phase, hkl) for hkl in explicit]
    millers = get_symmetrically_distinct_miller_indices(phase.conv, max_index)
    facets = [facet_geometry(phase, canonical_hkl(hkl, phase.symops)) for hkl in millers]
    if ranking == "dh":
        facets.sort(key=lambda f: (-round(f.d_dh, 4), -round(f.d, 4), max(abs(x) for x in f.hkl), f.hkl))
    else:
        facets.sort(key=lambda f: (-round(f.d, 4), max(abs(x) for x in f.hkl), f.hkl))
    return facets[:n_facets]


# =============================================================================
# Lattice matching
# =============================================================================
_ZSL_ARGS = inspect.signature(ZSLGenerator.__init__).parameters


def make_zsl(film_area: float, sub_area: float, max_area: float, max_strain: float) -> ZSLGenerator:
    """ZSL generator used as a deliberately loose pre-filter; the exact misfit
    criterion is applied afterwards (strain_metrics)."""
    r = film_area / sub_area
    kwargs = dict(
        max_area=max_area,
        # ZSL tests |A_f/A_s - j/i| (absolute), so the tolerance must scale with the area ratio
        max_area_ratio_tol=2.2 * max_strain * max(r, 1.0 / r),
        max_length_tol=1.5 * max_strain,
        max_angle_tol=3.0 * max_strain,   # relative tolerance on the angle
    )
    if "bidirectional" in _ZSL_ARGS:
        kwargs["bidirectional"] = True
    return ZSLGenerator(**kwargs)


def strain_metrics(f1, f2, s1, s2) -> dict:
    """In-plane misfit of the map (f1, f2) -> (s1, s2), i.e. film strained onto substrate."""
    gram_f = np.array([[f1 @ f1, f1 @ f2], [f1 @ f2, f2 @ f2]])
    gram_s = np.array([[s1 @ s1, s1 @ s2], [s1 @ s2, s2 @ s2]])
    e = np.sqrt(eigh(gram_s, gram_f, eigvals_only=True)) - 1.0   # principal strains, ascending
    a_f, a_s = area2d(f1, f2), area2d(s1, s2)
    g_f, g_s = angle_deg(f1, f2), angle_deg(s1, s2)
    return dict(
        e1=float(e[0]), e2=float(e[1]), max_abs=float(np.abs(e).max()),
        eps_a=norm(s1) / norm(f1) - 1.0, eps_b=norm(s2) / norm(f2) - 1.0,
        gamma_f=g_f, gamma_s=g_s, dgamma=g_s - g_f,
        area_f=a_f, area_s=a_s, area_mis=a_s / a_f - 1.0,
    )


def int_matrix(vecs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Integer 2x2 T with vecs = T @ basis (rows are vectors)."""
    t = np.linalg.lstsq(basis.T, vecs.T, rcond=None)[0].T
    ti = np.rint(t).astype(int)
    if not np.allclose(ti @ basis, vecs, atol=1e-4):
        raise RuntimeError("super-cell vectors are not integer combinations of the surface cell")
    return ti


def match_facets(sub: Phase, sf: Facet, film: Phase, ff: Facet, args):
    """Distinct coincidence cells with max|strain| <= args.max_strain for one facet pair
    (ranked by --sort), plus the closest candidate seen: (max strain, area) or None."""
    # ZSL caps the film *and* substrate cell areas; inflate its cap so a compressed film
    # cell slightly above --max-area is not lost, and apply --max-area to the substrate cell here
    zsl = make_zsl(ff.area, sf.area, args.max_area / (1.0 - args.max_strain) ** 2, args.max_strain)
    rho_s = len(sub.conv) / sub.conv.volume
    rho_f = len(film.conv) / film.conv.volume
    rows, seen, closest = [], set(), None
    for m in zsl(ff.mesh, sf.mesh, lowest=False):
        f1, f2 = np.asarray(m.film_sl_vectors, dtype=float)
        s1, s2 = np.asarray(m.substrate_sl_vectors, dtype=float)
        met = strain_metrics(f1, f2, s1, s2)
        if met["area_s"] > args.max_area:
            continue
        if closest is None or met["max_abs"] < closest[0]:
            closest = (met["max_abs"], met["area_s"])
        if met["max_abs"] > args.max_strain:
            continue
        key = (round(met["area_s"], 1), round(met["max_abs"], 5),
               round(norm(s1), 2), round(norm(s2), 2), round(met["gamma_s"], 1),
               round(norm(f1), 2), round(norm(f2), 2), round(met["gamma_f"], 1))
        if key in seen:      # symmetry-equivalent copy of a cell already kept
            continue
        seen.add(key)
        t_s = int_matrix(np.array([s1, s2]), sf.mesh)
        t_f = int_matrix(np.array([f1, f2]), ff.mesh)
        s_dir_a, s_dir_b = fmt_dir(s1, sub.conv), fmt_dir(s2, sub.conv)
        f_dir_a, f_dir_b = fmt_dir(f1, film.conv), fmt_dir(f2, film.conv)
        rows.append(dict(
            sub_id=sub.mpid, sub_formula=sub.formula, sub_sg=sub.spacegroup, sub_hkl=fmt_hkl(sf.hkl),
            film_id=film.mpid, film_formula=film.formula, film_sg=film.spacegroup, film_hkl=fmt_hkl(ff.hkl),
            area_A2=round(met["area_s"], 2),
            n_sub=int(round(abs(np.linalg.det(t_s)))),
            n_film=int(round(abs(np.linalg.det(t_f)))),
            max_strain_pct=round(100 * met["max_abs"], 4),
            e1_pct=round(100 * met["e1"], 4),
            e2_pct=round(100 * met["e2"], 4),
            eps_a_pct=round(100 * met["eps_a"], 4),
            eps_b_pct=round(100 * met["eps_b"], 4),
            dgamma_deg=round(met["dgamma"], 3),
            area_misfit_pct=round(100 * met["area_mis"], 4),
            sub_a=round(norm(s1), 3), sub_b=round(norm(s2), 3), sub_gamma=round(met["gamma_s"], 2),
            film_a=round(norm(f1), 3), film_b=round(norm(f2), 3), film_gamma=round(met["gamma_f"], 2),
            sub_dir_a=s_dir_a, sub_dir_b=s_dir_b, film_dir_a=f_dir_a, film_dir_b=f_dir_b,
            orientation=(f"{sub.formula}({fmt_hkl(sf.hkl)}){s_dir_a} || "
                         f"{film.formula}({fmt_hkl(ff.hkl)}){f_dir_a}"),
            T_sub=t_s.tolist(), T_film=t_f.tolist(),
            n_atoms_est=int(round(args.est_thickness * (met["area_s"] * rho_s + met["area_f"] * rho_f))),
            sub_mesh_area=round(sf.area, 4), film_mesh_area=round(ff.area, 4),
            sub_cif=sub.cif, film_cif=film.cif,
        ))
    rows.sort(key=SORT_KEYS[args.sort])
    return rows[: args.matches_per_pair], closest


# =============================================================================
# Output
# =============================================================================
def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_facets(phase: Phase, facets: list[Facet]) -> None:
    abc = ", ".join(f"{x:.3f}" for x in phase.conv.lattice.abc)
    print(f"\nFacets for {phase.label}   (a, b, c = {abc} A)")
    print("   (h k l)        d_DH [A]  d_lat [A]  A_2D [A^2]  |a| [A]  |b| [A]  gamma   in-plane a, b")
    for f in facets:
        a, b = f.mesh
        hkl_s = "(" + fmt_hkl(f.hkl) + ")"
        print(f"   {hkl_s:<14} {f.d_dh:8.3f} {f.d:10.3f} {f.area:11.2f} {norm(a):8.3f} {norm(b):8.3f} "
              f"{angle_deg(a, b):6.1f}   {fmt_dir(a, phase.conv)} {fmt_dir(b, phase.conv)}")


def print_best(rows: list[dict], args) -> None:
    print(f"\nBest coincidence cell per facet pair (by {args.sort}): max|strain| <= {100 * args.max_strain:.2f} %, "
          f"area <= {args.max_area:.0f} A^2")
    if not rows:
        print("  none found - try a larger --max-area, more facets or a looser --max-strain")
        return
    print(f"{'row':>4}  {'substrate (hkl)':<22} {'film (hkl)':<22} {'area':>7} {'n_s':>4} {'n_f':>4} "
          f"{'|a|':>7} {'|b|':>7} {'gamma':>6} {'e_a%':>7} {'e_b%':>7} {'dgam':>6} {'max%':>6} {'~atoms':>7}")
    for i, r in enumerate(rows[: args.top]):
        sub = f"{r['sub_formula']}({r['sub_hkl']})"
        film = f"{r['film_formula']}({r['film_hkl']})"
        print(f"{i:>4}  {sub:<22} {film:<22} {r['area_A2']:7.1f} {r['n_sub']:4d} {r['n_film']:4d} "
              f"{r['sub_a']:7.3f} {r['sub_b']:7.3f} {r['sub_gamma']:6.1f} {r['eps_a_pct']:7.3f} "
              f"{r['eps_b_pct']:7.3f} {r['dgamma_deg']:6.2f} {r['max_strain_pct']:6.3f} {r['n_atoms_est']:7d}")


def print_misses(misses, args) -> None:
    if not misses:
        return
    print("\nFacet pairs without a match (closest candidate seen by the ZSL pre-filter):")
    misses = sorted(misses, key=lambda x: x[4][0] if x[4] else math.inf)
    for sub, sf, film, ff, closest in misses[: args.top]:
        txt = (f"closest {100 * closest[0]:.2f} % at {closest[1]:.0f} A^2" if closest
               else "no near-coincidence below --max-area")
        print(f"  {sub.formula}({fmt_hkl(sf.hkl)}) | {film.formula}({fmt_hkl(ff.hkl)}): {txt}")


# =============================================================================
# Main
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lattice-matching screen for LiCoO2 | Li3PO4 interfaces (details in the module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("materials")
    g.add_argument("--substrate", default="LiCoO2", help="substrate formula for the MP query")
    g.add_argument("--film", default="Li3PO4", help="film formula for the MP query")
    g.add_argument("--substrate-ids", nargs="+", help="explicit MP ids for the substrate, e.g. mp-22526")
    g.add_argument("--film-ids", nargs="+", help="explicit MP ids for the film")
    g.add_argument("--substrate-file", nargs="+", help="local structure file(s) used instead of MP")
    g.add_argument("--film-file", nargs="+", help="local structure file(s) used instead of MP")
    g.add_argument("--substrate-polymorphs", type=int, default=1, help="screen the N lowest-E_hull substrate polymorphs")
    g.add_argument("--film-polymorphs", type=int, default=1, help="screen the N lowest-E_hull film polymorphs")
    g.add_argument("--max-ehull", type=float, default=0.05, help="E_hull window of the polymorph query (eV/atom)")
    g.add_argument("--include-theoretical", action="store_true", help="also accept MP entries without an ICSD match")
    g.add_argument("--api-key", default=None, help="MP API key (default: MP_API_KEY environment variable)")
    g.add_argument("--symprec", type=float, default=0.1, help="spglib symmetry tolerance (A)")

    g = p.add_argument_group("facets")
    g.add_argument("--max-index", type=int, default=4,
                   help="largest |h|,|k|,|l| considered (4 is needed to reach LCO (104))")
    g.add_argument("--n-facets", type=int, default=8, help="facets kept per phase after the ranking")
    g.add_argument("--ranking", choices=["dh", "lattice"], default="dh",
                   help="facet ranking: Donnay-Harker (all systematic absences) or lattice-plane spacing only")
    g.add_argument("--substrate-millers", nargs="+",
                   help="explicit substrate facets, e.g. 001 104 012 or 1,0,4 (overrides the auto selection)")
    g.add_argument("--film-millers", nargs="+", help="explicit film facets (overrides the auto selection)")

    g = p.add_argument_group("matching / output")
    g.add_argument("--max-strain", type=float, default=0.01,
                   help="max |principal in-plane strain| of the film (0.01 = 1 per cent)")
    g.add_argument("--max-area", type=float, default=200.0, help="max coincidence-cell area (A^2)")
    g.add_argument("--matches-per-pair", type=int, default=3, help="distinct cells kept per facet pair")
    g.add_argument("--est-thickness", type=float, default=12.0,
                   help="slab thickness per side (A) for the rough atom-count estimate")
    g.add_argument("--sort", choices=["area", "strain", "atoms"], default="area", help="ranking of the output")
    g.add_argument("--top", type=int, default=30, help="rows printed to screen")
    g.add_argument("--outdir", default="lco_lpo_match", help="output directory")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    (outdir / "structures").mkdir(parents=True, exist_ok=True)
    (outdir / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    # ---- bulk phases ---------------------------------------------------------
    subs = phases_from_files(args.substrate_file, "substrate", args.symprec) if args.substrate_file else None
    films = phases_from_files(args.film_file, "film", args.symprec) if args.film_file else None
    if subs is None or films is None:
        from mp_api.client import MPRester

        with MPRester(args.api_key) as mpr:
            if subs is None:
                subs = phases_from_mp(mpr, args.substrate, args.substrate_ids, args.substrate_polymorphs,
                                      args, "substrate")
            if films is None:
                films = phases_from_mp(mpr, args.film, args.film_ids, args.film_polymorphs, args, "film")

    # ---- facets --------------------------------------------------------------
    sub_millers = parse_millers(args.substrate_millers) if args.substrate_millers else None
    film_millers = parse_millers(args.film_millers) if args.film_millers else None
    facets, facet_rows = {}, []
    for phases, explicit in ((subs, sub_millers), (films, film_millers)):
        for ph in phases:
            cif = outdir / "structures" / f"{ph.mpid}_conv.cif"
            ph.conv.to(filename=str(cif))
            ph.cif = cif.relative_to(outdir).as_posix()
            facets[ph.mpid] = select_facets(ph, args.max_index, args.n_facets, args.ranking, explicit)
            print_facets(ph, facets[ph.mpid])
            for rank, f in enumerate(facets[ph.mpid], 1):
                a, b = f.mesh
                facet_rows.append(dict(
                    phase=ph.label, mpid=ph.mpid, rank=rank, hkl=fmt_hkl(f.hkl),
                    hkl_primitive=fmt_hkl(f.hkl_prim), d_DH_A=round(f.d_dh, 4), d_lattice_A=round(f.d, 4),
                    cell_area_A2=round(f.area, 3),
                    a_A=round(norm(a), 4), b_A=round(norm(b), 4), gamma_deg=round(angle_deg(a, b), 2),
                    dir_a=fmt_dir(a, ph.conv), dir_b=fmt_dir(b, ph.conv),
                ))

    # ---- matching ------------------------------------------------------------
    n_pairs = sum(len(facets[s.mpid]) * len(facets[f.mpid]) for s in subs for f in films)
    print(f"\nMatching {n_pairs} facet pairs (max area {args.max_area:.0f} A^2, "
          f"max strain {100 * args.max_strain:.2f} %) ...")
    all_rows, best_rows, misses = [], [], []
    for sub in subs:
        for film in films:
            for sf in facets[sub.mpid]:
                n_hit = 0
                for ff in facets[film.mpid]:
                    rows, closest = match_facets(sub, sf, film, ff, args)
                    if rows:
                        n_hit += 1
                        all_rows.extend(rows)
                        best_rows.append(rows[0])
                    else:
                        misses.append((sub, sf, film, ff, closest))
                print(f"  {sub.formula}({fmt_hkl(sf.hkl)}) vs {len(facets[film.mpid])} facets of "
                      f"{film.label}: {n_hit} matched")

    best_rows.sort(key=SORT_KEYS[args.sort])
    all_rows.sort(key=SORT_KEYS[args.sort])

    write_csv(facet_rows, outdir / "facets.csv")
    write_csv(all_rows, outdir / "matches.csv")
    write_csv(best_rows, outdir / "best_matches.csv")
    print_best(best_rows, args)
    print_misses(misses, args)
    print(f"\nWrote {outdir}/facets.csv, matches.csv ({len(all_rows)} rows), "
          f"best_matches.csv ({len(best_rows)} rows); 'row' above = row index in best_matches.csv")


if __name__ == "__main__":
    main()
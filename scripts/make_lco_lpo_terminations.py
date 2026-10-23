"""
make_lco_lpo_terminations.py
============================
Enumerate all symmetry-distinct terminations of one LCO | LPO orientation pair
(a row of best_matches.csv from lco_lpo_lattice_match.py), build every
interface, and characterise each termination.
 
For every (LCO termination, LPO termination) pair a directory is written with
  POSCAR_interface   LCO slab + LPO slab (LPO strained onto the LCO cell)
  POSCAR_LCO         the LCO part alone, same cell and positions
  POSCAR_LPO         the (strained) LPO part alone, same cell and positions
  info.txt           termination analysis (also collected in summary.csv)
Using the same cell for all three files makes
W_adh = E(LCO) + E(LPO) - E(interface) consistent.
 
Geometry (all lengths in Angstrom)
  z = 0 .. lco_vacuum                 vacuum
  LCO slab (>= --lco-thickness)       top face = LCO termination at the interface
  --separation                        top LCO atom -> bottom LPO atom
  LPO slab (>= --lpo-thickness)       bottom face = LPO termination at the interface
  --lpo-vacuum                        vacuum (periodic image of the LCO bottom follows)
The c vector is perpendicular to the interface plane (see the chat notes for why).
LPO is flipped with a proper rotation (180 deg about an in-plane axis), then strained
in-plane onto the LCO super-cell; its out-of-plane geometry is left unstrained.
 
Terminations
  Candidate cuts lie midway between atomic layers (atoms clustered along the
  normal with --ftol). Two cuts are equivalent if a bulk space-group operation
  that keeps the surface normal maps one onto the other (up to a lattice
  translation): this is exact for the face that meets the interface. For
  crystals without an operation that reverses the normal (e.g. polar Pmn2_1
  beta-Li3PO4), the (hkl) and (-h-k-l) faces are different and both are built.
  Slabs consist of whole bulk repeat units, so they are stoichiometric by
  construction; terminations differ in polarity, broken bonds and layer chemistry.
 
Reported per part (LCO / LPO), see info.txt / summary.csv
  composition, stoichiometric?        vs. the bulk formula
  interface layer                     atoms within --layer-depth of the interface face
  dipole / area (e/A)                 formal charges (--charges); polar if > 1e-3
                                      (Tasker type 3); scales with slab thickness
  broken bonds                        bulk minus slab coordination for each
                                      cation-anion pair in --bonds, split into the
                                      interface face and the vacuum face; any broken
                                      P-O bond means a cut PO4 tetrahedron
 
Run next to lco_lpo_lattice_match.py (imports helpers from it). Untested.
 
Example
  python make_lco_lpo_terminations.py lco_lpo_match/best_matches.csv --row 1 \
      --lco-thickness 12 --lpo-thickness 12 --separation 2.0 --lco-vacuum 10 --lpo-vacuum 10
"""
from __future__ import annotations
 
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
 
import numpy as np
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
 
from lco_lpo_lattice_match import angle_deg, int_matrix, make_zsl, reduce_2d, strain_metrics
 
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="All terminations + interfaces for one LCO | LPO orientation pair (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv", help="best_matches.csv (or matches.csv) from lco_lpo_lattice_match.py")
    p.add_argument("--row", type=int, default=1, help="0-based row of the CSV (substrate = LCO, film = LPO)")
    p.add_argument("--lco-thickness", type=float, default=12.0, help="minimum LCO slab thickness (A)")
    p.add_argument("--lpo-thickness", type=float, default=12.0, help="minimum LPO slab thickness (A)")
    p.add_argument("--separation", type=float, default=2.0, help="top LCO atom to bottom LPO atom (A)")
    p.add_argument("--lco-vacuum", type=float, default=10.0, help="vacuum below the LCO slab (A)")
    p.add_argument("--lpo-vacuum", type=float, default=10.0, help="vacuum above the LPO slab (A)")
    p.add_argument("--ftol", type=float, default=0.1,
                   help="atoms closer than this along the normal form one layer (A); LCO (211) layers are ~0.23 A apart")
    p.add_argument("--bonds", nargs="+", default=["Li-O:2.5", "Co-O:2.4", "P-O:2.0"],
                   help="cation-anion pairs and cutoffs (A) used for coordination / broken bonds")
    p.add_argument("--charges", nargs="+", default=["Li:1", "Co:3", "P:5", "O:-2"],
                   help="formal charges used for the dipole")
    p.add_argument("--layer-depth", type=float, default=1.0,
                   help="atoms within this depth (A) of the interface face define the terminating layer")
    p.add_argument("--max-strain", type=float, default=0.01, help="misfit threshold used in the screening run")
    p.add_argument("--symprec", type=float, default=0.1, help="spglib tolerance (A)")
    p.add_argument("--outdir", default="lco_lpo_terminations", help="output directory")
    return p.parse_args(argv)
 
 
def parse_pairs(tokens, sep: str) -> dict:
    """['Li-O:2.5'] -> {('Li', 'O'): 2.5};  ['Li:1'] -> {'Li': 1.0}"""
    out = {}
    for tok in tokens:
        key, val = tok.split(":")
        out[tuple(x.strip() for x in key.split(sep)) if sep else key.strip()] = float(val)
    return out
 
 
# =============================================================================
# Geometry: faces, cut planes, distinct terminations, slabs, super-cells
# =============================================================================
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)
 
 
def cartesian_ops(bulk: Structure, symprec: float) -> list:
    """Space-group operations (Cartesian rotation, Cartesian translation) incl. centring."""
    ops = SpacegroupAnalyzer(bulk, symprec=symprec).get_symmetry_operations(cartesian=True)
    return [(np.asarray(op.rotation_matrix, dtype=float), np.asarray(op.translation_vector, dtype=float))
            for op in ops]
 
 
def faces(hkl, bulk: Structure, ops) -> tuple[list, bool]:
    """Both faces (hkl), (-h-k-l) and whether they are symmetry-equivalent
    (some operation reverses the normal, e.g. inversion)."""
    g = unit(bulk.lattice.reciprocal_lattice_crystallographic.get_cartesian_coords(hkl))
    equivalent = any(np.allclose(rot @ g, -g, atol=1e-3) for rot, _ in ops)
    return [tuple(hkl), tuple(-x for x in hkl)], equivalent
 
 
def ouc_frame(sg: SlabGenerator) -> tuple[np.ndarray, float]:
    """Outward normal u of the slab's top face and repeat height h of the oriented unit cell."""
    a, b, c = sg.oriented_unit_cell.lattice.matrix
    u = unit(np.cross(a, b))
    if u @ c < 0:
        u = -u
    return u, float(u @ c)
 
 
def layer_cuts(sg: SlabGenerator, h: float, ftol: float) -> list[float]:
    """Cut heights (0 <= z < h) midway between atomic layers of the oriented unit cell."""
    z = np.sort((sg.oriented_unit_cell.frac_coords[:, 2] % 1.0) * h)
    gaps = np.diff(np.concatenate([z, [z[0] + h]]))          # gap above each atom (periodic)
    cuts = [float((z[i] + gaps[i] / 2) % h) for i in range(len(z)) if gaps[i] > ftol]
    if not cuts:                                             # no gap > ftol: use the largest one
        i = int(np.argmax(gaps))
        cuts = [float((z[i] + gaps[i] / 2) % h)]
    return sorted(cuts)
 
 
def distinct_cuts(cuts, u, d: float, ops, tol: float = 0.05) -> list[float]:
    """One cut per symmetry class: cuts z1, z2 give equivalent top faces if an operation with
    R u = u has z2 = z1 + u.t (mod the lattice-plane spacing d). Exact for the top face."""
    deltas = [float(u @ t) for rot, t in ops if np.allclose(rot @ u, u, atol=1e-3)]
    reps = []
    for z in cuts:
        if not any(abs(((z - zr - dl) / d + 0.5) % 1.0 - 0.5) * d < tol for zr in reps for dl in deltas):
            reps.append(z)
    return reps
 
 
def contiguous(struct: Structure, u) -> np.ndarray:
    """Cartesian coordinates with the slab made contiguous along u (undo wrapping through c)."""
    cart = np.array(struct.cart_coords, dtype=float)
    c_vec = struct.lattice.matrix[2] * (1.0 if struct.lattice.matrix[2] @ u > 0 else -1.0)
    period = float(c_vec @ u)
    z = cart @ u
    zs = np.sort(z)
    gaps = np.diff(np.concatenate([zs, [zs[0] + period]]))
    z0 = zs[(int(np.argmax(gaps)) + 1) % len(zs)]            # lowest atom above the vacuum gap
    return cart - np.floor((z - z0 + 1e-6) / period)[:, None] * c_vec
 
 
def make_slab(bulk: Structure, face, shift: float, t_min: float, u, h: float):
    """Slab of whole repeat units cut at `shift` with atomic thickness >= t_min.
    Returns (species, contiguous Cartesian coords, primitive in-plane mesh (2, 3))."""
    t_req = t_min
    for _ in range(5):
        sg = SlabGenerator(bulk, face, min_slab_size=t_req, min_vacuum_size=15.0, in_unit_planes=False,
                           center_slab=True, primitive=True, reorient_lattice=False)
        slab = sg.get_slab(shift=shift)
        cart = contiguous(slab, u)
        z = cart @ u
        if z.max() - z.min() >= t_min - 1e-6:
            mesh = np.array(slab.lattice.matrix[:2], dtype=float)
            if np.abs(mesh @ u).max() > 1e-4:
                raise RuntimeError(f"slab in-plane vectors of {face} are not in-plane")
            return list(slab.species), cart, mesh
        t_req += h
    raise RuntimeError(f"could not reach {t_min} A for face {face}")
 
 
def tile(species, cart, u, mesh, T):
    """Replicate a slab (primitive in-plane mesh) into the super-cell spanned by T @ mesh."""
    T = np.asarray(T, dtype=float)
    z = cart @ u
    frac = np.linalg.lstsq(mesh.T, (cart - np.outer(z, u)).T, rcond=None)[0].T   # (N, 2)
    # pymatgen slabs are stacked along an oblique c, so atoms can sit several in-plane cells
    # away from the origin: wrap them into the primitive cell before tiling
    frac -= np.floor(frac + 1e-9)
    tinv = np.linalg.inv(T)
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ T
    lo, hi = np.floor(corners.min(0)).astype(int) - 1, np.ceil(corners.max(0)).astype(int) + 1
    out_sp, out_xyz = [], []
    for p in range(lo[0], hi[0] + 1):
        for q in range(lo[1], hi[1] + 1):
            shifted = frac + np.array([p, q])
            g = shifted @ tinv                               # fractional w.r.t. the super-cell
            for i in np.nonzero(np.all((g > -1e-6) & (g < 1 - 1e-6), axis=1))[0]:
                out_sp.append(species[i])
                out_xyz.append(shifted[i] @ mesh + z[i] * u)
    n_expect = len(species) * round(abs(np.linalg.det(T)))
    if len(out_sp) != n_expect:
        raise RuntimeError(f"super-cell tiling produced {len(out_sp)} atoms, expected {n_expect}")
    return out_sp, np.array(out_xyz)
 
 
# =============================================================================
# Coincidence cell and interface assembly
# =============================================================================
def area2(mesh) -> float:
    return float(np.linalg.norm(np.cross(mesh[0], mesh[1])))
 
 
def find_match(film_mesh, sub_mesh, u_f, u_s, row: dict, max_strain: float):
    """ZSL coincidence cell that reproduces the screened row (same reduced cells and misfit)
    and can be realised by turning the film over with a proper rotation. Turning a film over
    reverses the handedness of its in-plane basis seen from above, so we need
    sign(u_f . F1 x F2) = -sign(u_s . S1 x S2). Returns (F (2, 3), S (2, 3), metrics) or None."""
    strain = float(row["max_strain_pct"]) / 100.0
    area = float(row["area_A2"])
    film_area = area / (1.0 + float(row["area_misfit_pct"]) / 100.0)
    zsl = make_zsl(area2(film_mesh), area2(sub_mesh), 1.01 * max(area, film_area) + 1.0,
                   max(max_strain, strain + 1e-6))
    want_s = sorted([float(row["sub_a"]), float(row["sub_b"])])
    want_f = sorted([float(row["film_a"]), float(row["film_b"])])
    for m in zsl(np.asarray(film_mesh, dtype=float), np.asarray(sub_mesh, dtype=float), lowest=False):
        f1, f2 = np.asarray(m.film_sl_vectors, dtype=float)
        s1, s2 = np.asarray(m.substrate_sl_vectors, dtype=float)
        met = strain_metrics(f1, f2, s1, s2)
        same = (np.allclose(sorted([np.linalg.norm(s1), np.linalg.norm(s2)]), want_s, atol=5e-3)
                and np.allclose(sorted([np.linalg.norm(f1), np.linalg.norm(f2)]), want_f, atol=5e-3)
                and abs(met["gamma_s"] - float(row["sub_gamma"])) < 0.05
                and abs(met["gamma_f"] - float(row["film_gamma"])) < 0.05
                and abs(met["max_abs"] - strain) < 1e-4)
        if same and (u_f @ np.cross(f1, f2)) * (u_s @ np.cross(s1, s2)) < 0:
            return np.array([f1, f2]), np.array([s1, s2]), met
    return None
 
 
def assemble(sub_sp, sub_xyz, film_sp, film_xyz, S, F, u_s, u_f, args):
    """Interface cell with x || S1, z = outward normal of the LCO top face and c perpendicular
    to the plane. LPO is turned over (proper rotation, 180 deg about F1), strained in-plane
    onto (S1, S2) and stacked with --separation and the two vacua.
    Returns (interface, LCO part, LPO part) as Structures sharing one lattice, plus the LCO and
    the unstrained LPO coincidence cells (2 x 3, in-plane vectors in the interface frame)."""
    S, F = np.array(S, dtype=float), np.array(F, dtype=float)
    if u_s @ np.cross(S[0], S[1]) < 0:            # make (S1, S2, u_s) right-handed;
        S[1], F[1] = -S[1], -F[1]                 # flip F2 too, so the pairing is unchanged
    ex = unit(S[0])
    rot_s = np.array([ex, np.cross(u_s, ex), u_s])          # LCO frame -> interface frame
    fx = unit(F[0])
    rot_f = np.array([fx, -np.cross(u_f, fx), -u_f])       # det = +1: film turned over, no mirror
    s3, f3 = S @ rot_s.T, F @ rot_f.T                       # cells in the interface frame (z = 0)
    s2d, f2d = s3[:, :2], f3[:, :2]
    sub, film = sub_xyz @ rot_s.T, film_xyz @ rot_f.T
    film[:, :2] = film[:, :2] @ np.linalg.inv(f2d) @ s2d    # in-plane strain onto the LCO cell
    sub[:, 2] += args.lco_vacuum - sub[:, 2].min()
    film[:, 2] += sub[:, 2].max() + args.separation - film[:, 2].min()
    lattice = Lattice([[s2d[0, 0], s2d[0, 1], 0.0],
                       [s2d[1, 0], s2d[1, 1], 0.0],
                       [0.0, 0.0, film[:, 2].max() + args.lpo_vacuum]])
 
    def make(species, xyz) -> Structure:
        return Structure(lattice, list(species), xyz, coords_are_cartesian=True, to_unit_cell=True)
 
    return (make(list(sub_sp) + list(film_sp), np.vstack([sub, film])), make(sub_sp, sub), make(film_sp, film),
            s3, f3)
 
 
# =============================================================================
# Termination analysis
# =============================================================================
def coordination(struct: Structure, cation: str, anion: str, cutoff: float) -> dict[int, int]:
    """{site index: number of `anion` neighbours within cutoff} for every `cation` site."""
    return {i: sum(1 for nn in struct.get_neighbors(site, cutoff) if nn.specie.symbol == anion)
            for i, site in enumerate(struct) if site.specie.symbol == cation}
 
 
def bulk_coordination(bulk: Structure, bonds: dict) -> dict:
    """Most common bulk coordination number of every cation-anion pair present in `bulk`."""
    ref = {}
    for (cat, an), cut in bonds.items():
        cns = list(coordination(bulk, cat, an, cut).values())
        if cns:
            ref[(cat, an)] = Counter(cns).most_common(1)[0][0]
    return ref
 
 
def formula_of(struct: Structure, idx) -> str:
    return Composition(Counter(struct[int(i)].specie.symbol for i in idx)).formula.replace(" ", "")
 
 
def outer_layer(z: np.ndarray, from_top: bool, ftol: float) -> np.ndarray:
    """Indices of the outermost atomic layer (atoms chained within ftol along the normal)."""
    order = np.argsort(-z if from_top else z)
    k = 1
    while k < len(order) and abs(z[order[k]] - z[order[k - 1]]) <= ftol:
        k += 1
    return order[:k]
 
 
def analyse_part(part: Structure, bulk: Structure, top_is_interface: bool, args, bonds, ref, charges) -> dict:
    z = part.cart_coords[:, 2]
    area = part.lattice.volume / part.lattice.c
    comp = part.composition.get_el_amt_dict()
    bulk_red = bulk.composition.reduced_composition.get_el_amt_dict()
    ratios = [comp.get(el, 0.0) / n for el, n in bulk_red.items()]
    k = math.floor(min(ratios) + 1e-6)
    excess = {el: comp.get(el, 0.0) - k * bulk_red.get(el, 0.0) for el in sorted(set(comp) | set(bulk_red))}
    if top_is_interface:
        face = np.nonzero(z > z.max() - args.layer_depth)[0]
    else:
        face = np.nonzero(z < z.min() + args.layer_depth)[0]
    q = np.array([charges.get(site.specie.symbol, 0.0) for site in part])
    dipole = float(q @ (z - z.mean())) / area                 # e*A / A^2 = e/A, along +z
    zmid = 0.5 * (z.min() + z.max())
    broken = {}
    for (cat, an), n_ref in ref.items():
        at_interface = at_vacuum = 0
        for i, n in coordination(part, cat, an, bonds[(cat, an)]).items():
            if (z[i] > zmid) == top_is_interface:
                at_interface += max(n_ref - n, 0)
            else:
                at_vacuum += max(n_ref - n, 0)
        broken[f"{cat}-{an}"] = (at_interface, at_vacuum)
    return dict(
        n_atoms=len(part),
        formula=part.composition.formula.replace(" ", ""),
        stoichiometric=all(abs(v) < 1e-6 for v in excess.values()),
        excess=" ".join(f"{el}{v:+g}" for el, v in excess.items() if abs(v) > 1e-6) or "-",
        thickness=float(z.max() - z.min()),
        outer_layer=formula_of(part, outer_layer(z, top_is_interface, args.ftol)),
        face_layer=formula_of(part, face),
        net_charge=float(q.sum()),
        dipole=dipole,
        polar=abs(dipole) > 1e-3,
        broken=broken,
    )
 
 
def describe(side: str, face, t_idx: int, shift: float, a: dict, depth: float) -> list[str]:
    bb = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in a["broken"].items()) or "-"
    return [
        f"{side}: face ({' '.join(str(x) for x in face)}), termination t{t_idx} "
        f"(cut at {shift:.4f} of the oriented-cell repeat)",
        f"  {a['n_atoms']} atoms, {a['formula']}, stoichiometric: {a['stoichiometric']}"
        + ("" if a["stoichiometric"] else f" (excess vs bulk: {a['excess']})"),
        f"  thickness {a['thickness']:.2f} A",
        f"  outermost layer at the interface: {a['outer_layer']};  atoms within {depth:g} A: {a['face_layer']}",
        f"  dipole/area {a['dipole']:+.4f} e/A -> {'POLAR' if a['polar'] else 'non-polar'}"
        f"   (net formal charge {a['net_charge']:+g} e)",
        f"  broken bonds (interface face / vacuum face): {bb}",
    ]
 
 
# =============================================================================
# Main
# =============================================================================
def lattice_block(label: str, bulk: Structure, spg: str, face, mesh, cell) -> list[str]:
    """Lattices of one side before matching: bulk cell, primitive surface cell of the face and the
    unstrained coincidence cell (in-plane vectors in the interface frame, comparable to the final cell)."""
    lat = bulk.lattice
    m1, m2 = reduce_2d(mesh[0], mesh[1])
    c1, c2 = np.asarray(cell[0], dtype=float), np.asarray(cell[1], dtype=float)
    n_cells = round(area2([c1, c2]) / area2([m1, m2]))
    return [
        f"{label} bulk {bulk.composition.reduced_formula} ({spg}): a = {lat.a:.4f}, b = {lat.b:.4f}, "
        f"c = {lat.c:.4f} A; alpha = {lat.alpha:.3f}, beta = {lat.beta:.3f}, gamma = {lat.gamma:.3f} deg; "
        f"V = {lat.volume:.3f} A^3",
        "  bulk lattice vectors (A):",
        *[f"    {row[0]:11.6f} {row[1]:11.6f} {row[2]:11.6f}" for row in lat.matrix],
        f"  ({' '.join(str(x) for x in face)}) primitive surface cell: |a| = {np.linalg.norm(m1):.4f}, "
        f"|b| = {np.linalg.norm(m2):.4f} A, gamma = {angle_deg(m1, m2):.3f} deg, area = {area2([m1, m2]):.3f} A^2",
        f"  coincidence cell before matching ({n_cells} surface cells, unstrained):",
        f"    a = ({c1[0]:11.6f} {c1[1]:11.6f})   |a| = {np.linalg.norm(c1):.4f} A",
        f"    b = ({c2[0]:11.6f} {c2[1]:11.6f})   |b| = {np.linalg.norm(c2):.4f} A",
        f"    gamma = {angle_deg(c1, c2):.3f} deg, area = {area2([c1, c2]):.3f} A^2",
    ]
 
 
def face_tag(face) -> str:
    return "".join(str(x) for x in face)
 
 
def main(argv=None) -> None:
    args = parse_args(argv)
    bonds, charges = parse_pairs(args.bonds, "-"), parse_pairs(args.charges, "")
    csv_path = Path(args.csv)
    with open(csv_path, newline="") as fh:
        row = list(csv.DictReader(fh))[args.row]
    bulk = {"LCO": Structure.from_file(str(csv_path.parent / row["sub_cif"])),
            "LPO": Structure.from_file(str(csv_path.parent / row["film_cif"]))}
    hkl = {"LCO": tuple(int(x) for x in row["sub_hkl"].split()),
           "LPO": tuple(int(x) for x in row["film_hkl"].split())}
    thick = {"LCO": args.lco_thickness, "LPO": args.lpo_thickness}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Row {args.row}: {row['orientation']}, {row['area_A2']} A^2, max strain {row['max_strain_pct']} %")
 
    ops, ref, face_lists, cache, spg = {}, {}, {}, {}, {}
    for side in ("LCO", "LPO"):
        spg[side] = SpacegroupAnalyzer(bulk[side], symprec=args.symprec).get_space_group_symbol()
        ops[side] = cartesian_ops(bulk[side], args.symprec)
        ref[side] = bulk_coordination(bulk[side], bonds)
        face_lists[side] = faces(hkl[side], bulk[side], ops[side])
        print(f"{side}: bulk coordination {ref[side]}; (hkl) and (-h-k-l) faces "
              f"{'equivalent' if face_lists[side][1] else 'DIFFERENT (non-centrosymmetric) - both are built'}")
 
    def face_set(side: str, face) -> dict:
        """Distinct terminations of one face, with their slabs (cached)."""
        if (side, face) not in cache:
            sg = SlabGenerator(bulk[side], face, min_slab_size=thick[side], min_vacuum_size=15.0,
                               in_unit_planes=False, center_slab=True, primitive=True, reorient_lattice=False)
            u, h = ouc_frame(sg)
            cuts = distinct_cuts(layer_cuts(sg, h, args.ftol), u, bulk[side].lattice.d_hkl(face), ops[side])
            slabs = [make_slab(bulk[side], face, z / h, thick[side], u, h) for z in cuts]
            cache[(side, face)] = dict(face=face, u=u, shifts=[z / h for z in cuts], slabs=slabs)
            print(f"  {side} ({' '.join(str(x) for x in face)}): {len(cuts)} symmetry-distinct terminations")
        return cache[(side, face)]
 
    lco_faces, lco_equiv = face_lists["LCO"]
    lpo_faces, lpo_equiv = face_lists["LPO"]
    summary, n = [], 0
    for sface in (lco_faces[:1] if lco_equiv else lco_faces):
        ss = face_set("LCO", sface)
        used_lpo_face = False
        for fface in lpo_faces:
            if lpo_equiv and used_lpo_face:          # equivalent faces: one realisable face is enough
                break
            fs = face_set("LPO", fface)
            match = find_match(fs["slabs"][0][2], ss["slabs"][0][2], fs["u"], ss["u"], row, args.max_strain)
            if match is None:
                print(f"  LPO ({' '.join(str(x) for x in fface)}) on LCO ({' '.join(str(x) for x in sface)}): "
                      "this coincidence cell needs a mirrored film - skipped")
                continue
            used_lpo_face = True
            F, S, met = match
            film_tiles = [tile(sp, xyz, fs["u"], mesh, int_matrix(F, mesh)) for sp, xyz, mesh in fs["slabs"]]
            for i, (sp, xyz, mesh) in enumerate(ss["slabs"]):
                sub_tile = tile(sp, xyz, ss["u"], mesh, int_matrix(S, mesh))
                for j, film_tile in enumerate(film_tiles):
                    full, lco, lpo, cell_lco, cell_lpo = assemble(*sub_tile, *film_tile, S, F, ss["u"], fs["u"], args)
                    a_lco = analyse_part(lco, bulk["LCO"], True, args, bonds, ref["LCO"], charges)
                    a_lpo = analyse_part(lpo, bulk["LPO"], False, args, bonds, ref["LPO"], charges)
                    area = full.lattice.volume / full.lattice.c
                    q = np.array([charges.get(site.specie.symbol, 0.0) for site in full])
                    dip_full = float(q @ (full.cart_coords[:, 2] - full.cart_coords[:, 2].mean())) / area
 
                    name = f"{n:03d}_LCO{face_tag(sface)}-t{i}__LPO{face_tag(fface)}-t{j}"
                    d = outdir / name
                    d.mkdir(exist_ok=True)
                    full.get_sorted_structure().to(filename=str(d / "POSCAR_interface"), fmt="poscar")
                    lco.get_sorted_structure().to(filename=str(d / "POSCAR_LCO"), fmt="poscar")
                    lpo.get_sorted_structure().to(filename=str(d / "POSCAR_LPO"), fmt="poscar")
                    lines = [
                        f"{name}",
                        f"cell: |a| = {full.lattice.a:.3f}, |b| = {full.lattice.b:.3f} A, gamma = "
                        f"{full.lattice.gamma:.2f} deg, c = {full.lattice.c:.3f} A (perpendicular), "
                        f"area {area:.2f} A^2, {len(full)} atoms",
                        f"LPO in-plane strain: principal {100 * met['e1']:+.3f} / {100 * met['e2']:+.3f} %, "
                        f"along a {100 * met['eps_a']:+.3f} %, along b {100 * met['eps_b']:+.3f} %, "
                        f"angle {met['dgamma']:+.2f} deg",
                        f"separation {args.separation} A, vacuum below LCO {args.lco_vacuum} A, "
                        f"above LPO {args.lpo_vacuum} A",
                        f"interface cell dipole/area {dip_full:+.4f} e/A (consider a dipole correction along c)",
                        "",
                        "Lattices before matching (in-plane vectors in the interface frame: x || a, "
                        "z = interface normal; LPO already turned over, not yet strained)",
                        *lattice_block("LCO", bulk["LCO"], spg["LCO"], sface, mesh, cell_lco),
                        *lattice_block("LPO", bulk["LPO"], spg["LPO"], fface, fs["slabs"][j][2], cell_lpo),
                        "After matching both parts share the LCO coincidence cell (LCO is not strained).",
                        "",
                        *describe("LCO", sface, i, ss["shifts"][i], a_lco, args.layer_depth),
                        "",
                        *describe("LPO", fface, j, fs["shifts"][j], a_lpo, args.layer_depth),
                    ]
                    (d / "info.txt").write_text("\n".join(lines) + "\n")
                    rec = dict(dir=name, lco_face=face_tag(sface), lco_term=i, lpo_face=face_tag(fface),
                               lpo_term=j, n_atoms=len(full), area_A2=round(area, 2),
                               max_strain_pct=round(100 * met["max_abs"], 4), cell_dipole=round(dip_full, 5))
                    for side, a in (("lco", a_lco), ("lpo", a_lpo)):
                        rec.update({f"{side}_stoich": a["stoichiometric"], f"{side}_outer_layer": a["outer_layer"],
                                    f"{side}_dipole": round(a["dipole"], 5), f"{side}_polar": a["polar"],
                                    f"{side}_thickness": round(a["thickness"], 2)})
                        for pair, (b_i, b_v) in a["broken"].items():
                            rec[f"{side}_broken_{pair}_interface"] = b_i
                            rec[f"{side}_broken_{pair}_vacuum"] = b_v
                    summary.append(rec)
                    po = a_lpo["broken"].get("P-O", (0, 0))
                    print(f"  {name}: {len(full)} atoms | LCO {a_lco['outer_layer']:<10} "
                          f"{'polar' if a_lco['polar'] else 'non-polar':9} | LPO {a_lpo['outer_layer']:<10} "
                          f"{'polar' if a_lpo['polar'] else 'non-polar':9} | P-O broken {po[0]}/{po[1]}")
                    n += 1
 
    if summary:
        with open(outdir / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(f"\n{n} interfaces written to {outdir}/ (one directory each; overview in summary.csv)")
 
 
if __name__ == "__main__":
    main()
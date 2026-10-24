#!/usr/bin/env python
"""
analyse_md.py
=============
Analysis of md_interface.py runs: stability of the interface during the anneal and the production
runs, and Li transport at every production temperature. Runs on a CPU; no model needed.

Interface plane and Li regions (taken from the first frame of each run)
  The interface plane lies halfway between the highest LCO atom and the lowest LPO atom. Li atoms
  are grouped by where they start: "interface" (within --interface-width of the plane), "LCO
  surface" / "LPO surface" (within --surface-width of the free surfaces facing the vacuum), and
  "LCO interior" / "LPO interior" (the rest). Frozen atoms are left out.

Stability (anneal and every production run), per analysed frame
  Co_in_LPO   Co atoms that crossed the interface plane into LPO
  P_in_LCO    P atoms that crossed into LCO
  PO4_broken  P atoms without exactly 4 O within --po-cutoff
  lost        atoms more than 3 A beyond the slab surfaces (gone into the vacuum)
  slide_A     lateral slide of LPO relative to LCO (centre of mass of P vs Co)
  plus element density profiles along z, first vs last 10 % of the run

Li transport (production runs)
  MSD per region, in-plane (xy) and along the normal (z), averaged over time origins, after
  removing the drift of the host framework (Co for Li that start on the LCO side, P for the LPO
  side). D_xy = slope / 4 and D_z = slope / 2, fitted over the lag window --fit (fractions of the
  run length). A region counts as diffusive if its in-plane MSD at the end of the window exceeds
  --min-msd. D_z of thin regions is limited by their width; use D_xy for comparisons.
  Crossings of the interface plane (a crossing counts once the Li is 1 A past the plane) and the
  net number of Li that end on the other side.
  Arrhenius fit of D_xy for every region that is diffusive at two or more temperatures:
  Ea, D0 and the extrapolated D(300 K).

Output (<md dir>/analysis/)
  stability.csv            summary per run; stability_<run>.csv  per-frame values
  zprofile_<run>.csv       element densities along z (atoms per A per frame), first vs last 10 %
  msd_T####.csv            MSD vs lag per region
  diffusion.csv, crossings.csv, arrhenius.csv
  *.png                    MSD, z-profile and Arrhenius plots if matplotlib is installed

Requirements: ase, numpy (matplotlib optional). Untested on real data.

Example
  python analyse_md.py mh1_interfaces/001_LCO_113_s1-bot__LPO_110_s2-bot_432at/md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory

KB_EV = 8.617333262e-5
A2_PER_PS_TO_CM2_PER_S = 1e-4
REGIONS = ["LCO surface", "LCO interior", "interface", "LPO interior", "LPO surface"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stability and Li transport from md_interface.py runs (see module docstring).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("md_dir", help="output directory of md_interface.py")
    p.add_argument("--interface-width", type=float, default=3.0, help="half-width of the interface region (A)")
    p.add_argument("--surface-width", type=float, default=3.0, help="depth of the free-surface regions (A)")
    p.add_argument("--fit", type=float, nargs=2, default=[0.1, 0.5], metavar=("START", "END"),
                   help="lag window of the MSD fit, as fractions of the run length")
    p.add_argument("--min-msd", type=float, default=1.0, help="in-plane MSD (A^2) above which a region counts as diffusive")
    p.add_argument("--po-cutoff", type=float, default=2.0, help="P-O bond cutoff (A)")
    p.add_argument("--stride", type=int, default=5, help="analyse every n-th frame for the stability metrics")
    p.add_argument("--hysteresis", type=float, default=1.0, help="distance past the interface plane for a crossing (A)")
    return p.parse_args(argv)


# =============================================================================
# Helpers
# =============================================================================
def load(path: Path):
    """Symbols, cell and unwrapped positions (frames, atoms, 3) of a trajectory."""
    with Trajectory(str(path)) as traj:
        first = traj[0]
        pos = np.array([a.positions for a in traj])
    return np.array(first.get_chemical_symbols()), np.array(first.cell), pos


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def geometry(pos0: np.ndarray, is_lpo: np.ndarray) -> dict:
    lco, lpo = ~is_lpo, is_lpo
    top_lco, bot_lpo = pos0[lco, 2].max(), pos0[lpo, 2].min()
    return dict(z_int=0.5 * (top_lco + bot_lpo), bot=pos0[lco, 2].min(), top=pos0[lpo, 2].max())


def regions(z0: np.ndarray, is_lpo: np.ndarray, g: dict, args) -> np.ndarray:
    """Region name for every atom from its starting height."""
    out = np.where(is_lpo, "LPO interior", "LCO interior").astype(object)
    out[(~is_lpo) & (z0 < g["bot"] + args.surface_width)] = "LCO surface"
    out[is_lpo & (z0 > g["top"] - args.surface_width)] = "LPO surface"
    out[np.abs(z0 - g["z_int"]) <= args.interface_width] = "interface"
    return out


def coordination(pos_a: np.ndarray, pos_b: np.ndarray, A2: np.ndarray, cutoff: float) -> np.ndarray:
    """Number of B atoms within cutoff of every A atom; minimum image in-plane (cell >> cutoff)."""
    d = pos_b[None, :, :] - pos_a[:, None, :]
    frac = d[..., :2] @ np.linalg.inv(A2)
    frac -= np.round(frac)
    d[..., :2] = frac @ A2
    return (np.sqrt((d ** 2).sum(-1)) < cutoff).sum(axis=1)


# =============================================================================
# Stability
# =============================================================================
def stability(name: str, sym, cell, pos, is_lpo, dt_ps: float, args, out: Path) -> dict:
    A2 = cell[:2, :2]
    g = geometry(pos[0], is_lpo)
    co, p_at, o_at = sym == "Co", sym == "P", sym == "O"
    rows = []
    for f in range(0, len(pos), args.stride):
        x = pos[f]
        slide = (x[p_at, :2].mean(0) - pos[0][p_at, :2].mean(0)) - (x[co, :2].mean(0) - pos[0][co, :2].mean(0))
        n_o = coordination(x[p_at], x[o_at], A2, args.po_cutoff) if p_at.any() else np.array([])
        rows.append(dict(time_ps=round(f * dt_ps, 4),
                         Co_in_LPO=int((x[co, 2] > g["z_int"]).sum()),
                         P_in_LCO=int((x[p_at, 2] < g["z_int"]).sum()),
                         PO4_broken=int((n_o != 4).sum()),
                         lost=int(((x[:, 2] < g["bot"] - 3.0) | (x[:, 2] > g["top"] + 3.0)).sum()),
                         slide_A=round(float(np.linalg.norm(slide)), 4)))
    write_csv(rows, out / f"stability_{name}.csv")
    zprofile(name, sym, pos, out)
    last = rows[-1]
    return dict(run=name, frames=len(pos), time_ps=last["time_ps"],
                **{f"{k}_final": last[k] for k in ("Co_in_LPO", "P_in_LCO", "PO4_broken", "lost", "slide_A")},
                **{f"{k}_max": max(r[k] for r in rows) for k in ("Co_in_LPO", "P_in_LCO", "PO4_broken", "lost")})


def zprofile(name: str, sym, pos, out: Path, dz: float = 0.2) -> None:
    n = max(1, len(pos) // 10)
    zmin, zmax = pos[..., 2].min() - 1.0, pos[..., 2].max() + 1.0
    edges = np.arange(zmin, zmax + dz, dz)
    centres = 0.5 * (edges[1:] + edges[:-1])
    data = {"z_A": np.round(centres, 3)}
    for el in sorted(set(sym.tolist())):
        m = sym == el
        for tag, sl in (("first", slice(0, n)), ("last", slice(len(pos) - n, len(pos)))):
            h, _ = np.histogram(pos[sl][:, m, 2].ravel(), bins=edges)
            data[f"{el}_{tag}"] = np.round(h / (dz * n), 5)
    rows = [{k: v[i] for k, v in data.items()} for i in range(len(centres))]
    write_csv(rows, out / f"zprofile_{name}.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        els = [e for e in ("Li", "Co", "P", "O") if f"{e}_first" in data]
        fig, axes = plt.subplots(len(els), 1, figsize=(7, 1.8 * len(els)), sharex=True)
        for ax, el in zip(np.atleast_1d(axes), els):
            ax.plot(centres, data[f"{el}_first"], label="first 10 %", lw=1)
            ax.plot(centres, data[f"{el}_last"], label="last 10 %", lw=1)
            ax.set_ylabel(f"{el} (1/A)")
        np.atleast_1d(axes)[0].legend(fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel("z (A)")
        fig.suptitle(f"density along z: {name}")
        fig.tight_layout()
        fig.savefig(out / f"zprofile_{name}.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


# =============================================================================
# Li transport
# =============================================================================
def msd_curves(x: np.ndarray, lags) -> np.ndarray:
    """Time-origin averaged MSD (xy, z) for positions x (frames, atoms, 3)."""
    out = np.zeros((len(lags), 2))
    for k, L in enumerate(lags):
        d = x[L:] - x[:-L]
        out[k, 0] = (d[..., :2] ** 2).sum(-1).mean()
        out[k, 1] = (d[..., 2] ** 2).mean()
    return out


def transport(T: float, sym, cell, pos, is_lpo, frozen, dt_ps: float, args, out: Path):
    g = geometry(pos[0], is_lpo)
    li = (sym == "Li") & ~frozen
    reg = regions(pos[0][:, 2], is_lpo, g, args)
    co_free = (sym == "Co") & ~frozen
    p_at = sym == "P"
    drift_lco = pos[:, co_free].mean(1) - pos[0, co_free].mean(0)      # (frames, 3)
    drift_lpo = pos[:, p_at].mean(1) - pos[0, p_at].mean(0)
    lpo_side = is_lpo & (reg != "interface")                           # framework used for the drift correction
    corr = pos - np.where(lpo_side[None, :, None], drift_lpo[:, None, :], drift_lco[:, None, :])

    n_f = len(pos)
    t_run = (n_f - 1) * dt_ps
    lags = np.unique(np.linspace(1, max(1, (n_f - 1) // 2), num=min(100, max(1, (n_f - 1) // 2))).astype(int))
    lag_t = lags * dt_ps
    win = (lag_t >= args.fit[0] * t_run) & (lag_t <= args.fit[1] * t_run)
    msd_rows = {"lag_ps": np.round(lag_t, 4)}
    diff_rows = []
    for r in REGIONS:
        m = li & (reg == r)
        if not m.any():
            continue
        c = msd_curves(corr[:, m], lags)
        msd_rows[f"{r} xy"] = np.round(c[:, 0], 5)
        msd_rows[f"{r} z"] = np.round(c[:, 1], 5)
        row = dict(T_K=T, region=r, n_Li=int(m.sum()))
        if win.sum() >= 3:
            sxy = np.polyfit(lag_t[win], c[win, 0], 1)[0]
            sz = np.polyfit(lag_t[win], c[win, 1], 1)[0]
            msd_end = float(c[win, 0][-1])
            row.update(D_xy_cm2_s=sxy / 4.0 * A2_PER_PS_TO_CM2_PER_S, D_z_cm2_s=sz / 2.0 * A2_PER_PS_TO_CM2_PER_S,
                       msd_xy_end_A2=round(msd_end, 4), diffusive=bool(msd_end > args.min_msd and sxy > 0))
        diff_rows.append(row)
    write_csv([{k: v[i] for k, v in msd_rows.items()} for i in range(len(lags))], out / f"msd_T{int(round(T)):04d}.csv")
    plot_msd(T, msd_rows, out)

    # crossings of the interface plane, with hysteresis
    z = pos[:, li, 2]
    side = np.where(z[0] > g["z_int"], 1, -1)
    start_side = side.copy()
    up = down = 0
    for f in range(1, n_f):
        go_up = (side < 0) & (z[f] > g["z_int"] + args.hysteresis)
        go_down = (side > 0) & (z[f] < g["z_int"] - args.hysteresis)
        up += int(go_up.sum())
        down += int(go_down.sum())
        side[go_up], side[go_down] = 1, -1
    cross = dict(T_K=T, time_ps=round(t_run, 3), crossings_LCO_to_LPO=up, crossings_LPO_to_LCO=down,
                 net_Li_now_in_LPO=int(((start_side < 0) & (side > 0)).sum()),
                 net_Li_now_in_LCO=int(((start_side > 0) & (side < 0)).sum()))
    return diff_rows, cross


def plot_msd(T: float, rows: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in rows.items():
        if k.endswith(" xy"):
            ax.plot(rows["lag_ps"], v, label=k[:-3])
    ax.set_xlabel("lag (ps)")
    ax.set_ylabel("Li MSD in-plane (A$^2$)")
    ax.set_title(f"{T:.0f} K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"msd_T{int(round(T)):04d}.png", dpi=150)
    plt.close(fig)


def arrhenius(diff_rows: list[dict], out: Path) -> list[dict]:
    rows = []
    for r in REGIONS:
        pts = [(d["T_K"], d["D_xy_cm2_s"]) for d in diff_rows
               if d["region"] == r and d.get("diffusive") and d.get("D_xy_cm2_s", 0) > 0]
        if len(pts) < 2:
            continue
        T, D = np.array(pts).T
        slope, icpt = np.polyfit(1.0 / T, np.log(D), 1)
        ea, d0 = -slope * KB_EV, math.exp(icpt)
        rows.append(dict(region=r, n_T=len(pts), Ea_eV=round(ea, 4), D0_cm2_s=d0,
                         D_300K_cm2_s=d0 * math.exp(-ea / (KB_EV * 300.0)), T_range_K=f"{T.min():.0f}-{T.max():.0f}"))
    write_csv(rows, out / "arrhenius.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if rows:
            fig, ax = plt.subplots(figsize=(5, 4))
            for r in rows:
                pts = sorted((d["T_K"], d["D_xy_cm2_s"]) for d in diff_rows
                             if d["region"] == r["region"] and d.get("diffusive") and d.get("D_xy_cm2_s", 0) > 0)
                T, D = np.array(pts).T
                ax.semilogy(1000.0 / T, D, "o-", label=f"{r['region']} (Ea {r['Ea_eV']:.2f} eV)")
            ax.set_xlabel("1000 / T (1/K)")
            ax.set_ylabel("D$_{xy}$ (cm$^2$/s)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "arrhenius.png", dpi=150)
            plt.close(fig)
    except ImportError:
        pass
    return rows


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> None:
    args = parse_args(argv)
    md = Path(args.md_dir)
    info = json.loads((md / "md_info.json").read_text())
    n = info["n_atoms"]
    is_lpo = np.zeros(n, dtype=bool)
    is_lpo[info["lpo_indices"]] = True
    frozen = np.zeros(n, dtype=bool)
    frozen[info["frozen_indices"]] = True
    out = md / "analysis"
    out.mkdir(exist_ok=True)

    stab, diff_rows, cross_rows = [], [], []
    ann = md / "anneal" / "anneal.traj"
    if ann.exists():
        sym, cell, pos = load(ann)
        settings = json.loads((md / "anneal" / "run.json").read_text())
        res_path = md / "anneal" / "anneal.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {}
        s = stability("anneal", sym, cell, pos, is_lpo, settings["timestep_fs"] * settings["interval"] / 1000.0, args, out)
        if res:
            s.update(dE_annealed_vs_start_eV=round(res["dE_eV"], 4))
        stab.append(s)
        print(f"anneal: {len(pos)} frames" + (f", energy change after relaxing {res['dE_eV']:+.4f} eV" if res else
                                              " (not finished)")
              + f"; at the end: {s['Co_in_LPO_final']} Co in LPO, {s['P_in_LCO_final']} P in LCO, "
                f"{s['PO4_broken_final']} broken PO4, {s['lost_final']} atoms lost, LPO slid {s['slide_A_final']:.2f} A")

    for d in sorted(md.glob("T[0-9][0-9][0-9][0-9]")):
        traj = d / "prod.traj"
        if not traj.exists():
            continue
        run = json.loads((d / "run.json").read_text())
        dt_ps = run["timestep_fs"] * run["interval"] / 1000.0
        sym, cell, pos = load(traj)
        if len(pos) < 10:
            print(f"{d.name}: only {len(pos)} frames, skipped")
            continue
        T = float(run["T"])
        s = stability(d.name, sym, cell, pos, is_lpo, dt_ps, args, out)
        stab.append(s)
        rows, cross = transport(T, sym, cell, pos, is_lpo, frozen, dt_ps, args, out)
        diff_rows += rows
        cross_rows.append(cross)
        done = "" if (d / "prod.done").exists() else " (still running)"
        print(f"{d.name}: {len(pos)} frames, {cross['time_ps']:.0f} ps{done}; Li crossings "
              f"{cross['crossings_LCO_to_LPO']} LCO->LPO, {cross['crossings_LPO_to_LCO']} LPO->LCO; at the end "
              f"{s['Co_in_LPO_final']} Co in LPO, {s['P_in_LCO_final']} P in LCO, {s['PO4_broken_final']} broken PO4, "
              f"{s['lost_final']} lost, LPO slid {s['slide_A_final']:.2f} A")
        for r in rows:
            if "D_xy_cm2_s" in r:
                print(f"    {r['region']:<13} {r['n_Li']:4d} Li  D_xy {r['D_xy_cm2_s']:.2e} cm^2/s  "
                      f"MSD_xy {r['msd_xy_end_A2']:.2f} A^2  {'diffusive' if r['diffusive'] else 'not diffusive'}")

    write_csv(stab, out / "stability.csv")
    write_csv(diff_rows, out / "diffusion.csv")
    write_csv(cross_rows, out / "crossings.csv")
    arr = arrhenius(diff_rows, out)
    for r in arr:
        print(f"Arrhenius {r['region']}: Ea {r['Ea_eV']:.3f} eV, D(300 K) {r['D_300K_cm2_s']:.2e} cm^2/s "
              f"({r['n_T']} temperatures, {r['T_range_K']} K)")
    print(f"results in {out}/")


if __name__ == "__main__":
    main()

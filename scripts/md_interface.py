#!/usr/bin/env python
"""
md_interface.py
===============
Molecular dynamics of a relaxed LCO | LPO interface with MACE-MH-1 (--backend mace, default) or UMA
(--backend uma, omat task): an anneal first, then NVT runs at several temperatures for Li
transport. Analyse the output with analyse_md.py. Model options: see mlip_calculators.py.

Start structure
  The lowest-energy converged structure in results.csv of a registry scan (relax_interface_scan.py),
  or the one named with --structure. The LCO / LPO labels of the atoms come from scan_info.json.

Stages (--stages, default: both)
  anneal      Langevin: --anneal-equil-ps at --T-low, linear heating to --anneal-T over --heat-ps,
              --hold-ps at --anneal-T, linear cooling back to --T-low over --cool-ps; then a
              relaxation of all atoms at full precision (float64 for MACE, the FP32 "batch" preset
              for UMA). anneal/POSCAR_annealed is the result; anneal.json
              compares its energy with the start structure (lower = the anneal found a better
              interface).
  production  For every T in --temperatures: velocities at T, --equil-ps of Langevin, then --prod-ps
              with the Bussi thermostat (--thermostat langevin to keep Langevin), starting from
              anneal/POSCAR_annealed (--from relaxed: from the scan structure instead). Each T has
              its own directory, so after the anneal the temperatures can run as separate jobs.

Setup
  NVT, cell fixed (vacuum along c). --repeat N1 N2 makes an in-plane supercell first, worth it when
  a cell vector is short (4.9 A for interface 001). LCO atoms within --fix-bottom of the LCO bottom
  are frozen: a bulk-like substrate, which also keeps the slab from drifting. Time step --timestep
  (1 fs); a frame (positions, momenta, energy) every --interval steps. Positions are not wrapped, as
  the MSDs need. The run stops if two atoms come closer than --min-dist or the temperature runs
  away, both signs that the model has left its comfort zone.

Restarting
  Every phase resumes from the last frame of its trajectory, so a job that hit the wall-time limit
  can be resubmitted with the same command; finished phases are skipped. Rerunning with a larger
  --prod-ps extends a finished production run.

Output (--outdir, default <interface dir>/md)
  md_info.json   start structure, LPO atom indices, frozen atoms, model, settings. A directory
                 belongs to one start structure, supercell, frozen region and model; the script
                 stops instead of continuing a trajectory with another one.
  start/         POSCAR of the start structure
  anneal/        anneal.traj, anneal.log, POSCAR_annealed, anneal.json, relax.log
  T0800/ ...     equil.traj, equil.log, prod.traj, prod.log, run.json   (one directory per T)

Precision and speed: MD runs in float32 for MACE (--dtype) and with the fastest UMA preset,
"default" (UMA Nitro on Ampere or newer GPUs, reduced precision; --uma-settings high-precision for
full FP32). The log prints ps/day after the first ps: check it before committing to ns runs.
MACE-MH-1 is a large model; --cueq switches on cuEquivariance kernels if they are installed.

Requirements: ase >= 3.23, numpy, and mace-torch >= 0.3.14 or fairchem-core >= 2.24. Run next to
mlip_calculators.py. Untested on real data.

Examples
  python md_interface.py mh1_interfaces/001_LCO_113_s1-bot__LPO_110_s2-bot_432at/registry_scan
  python md_interface.py <scan dir> --stages anneal                            # job 1
  python md_interface.py <scan dir> --stages production --temperatures 800     # then one job per T
  python md_interface.py uma_interfaces/<interface>/registry_scan --backend uma --repeat 2 1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE

try:
    from ase.md.bussi import Bussi                  # ase >= 3.23
except ImportError:
    Bussi = None
try:
    from ase.md.velocitydistribution import thermalize_momenta      # ase >= 3.29
except ImportError:
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    def thermalize_momenta(atoms, temperature_K, exact_temperature=False, rng=None):
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, force_temp=exact_temperature, rng=rng)

from mlip_calculators import add_model_args, get_calculator, model_info, needs_precise_copy, resolve_model_args


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anneal + NVT MD of an LCO | LPO interface with MACE-MH-1 or UMA (see module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("scan_dir", help="registry scan directory (scan_interface_registry.py + relax_interface_scan.py)")
    p.add_argument("--structure", help="structure of the scan to start from (default: lowest relaxed energy)")
    p.add_argument("--outdir", help="output directory (default: <interface dir>/md)")
    p.add_argument("--stages", nargs="+", choices=["anneal", "production"], default=["anneal", "production"])
    p.add_argument("--repeat", type=int, nargs=2, default=[1, 1], metavar=("N1", "N2"),
                   help="in-plane supercell of the interface for MD, e.g. 2 1 for a cell that is narrow along a")
    g = p.add_argument_group("anneal")
    g.add_argument("--T-low", type=float, default=300.0, help="start and end temperature of the anneal (K)")
    g.add_argument("--anneal-T", type=float, default=900.0, help="anneal temperature (K)")
    g.add_argument("--anneal-equil-ps", type=float, default=10.0, help="equilibration at --T-low (ps)")
    g.add_argument("--heat-ps", type=float, default=20.0, help="linear heating (ps)")
    g.add_argument("--hold-ps", type=float, default=100.0, help="time at --anneal-T (ps)")
    g.add_argument("--cool-ps", type=float, default=50.0, help="linear cooling (ps)")
    g.add_argument("--fmax", type=float, default=0.02, help="force convergence of the final relaxation (eV/A)")
    g.add_argument("--max-steps", type=int, default=3000, help="max steps of the final relaxation")
    g = p.add_argument_group("production")
    g.add_argument("--temperatures", type=float, nargs="+", default=[600.0, 700.0, 800.0, 900.0], help="K")
    g.add_argument("--from", dest="start_from", choices=["annealed", "relaxed"], default="annealed",
                   help="production start: the annealed structure or the scan structure")
    g.add_argument("--equil-ps", type=float, default=20.0, help="Langevin equilibration at each T (ps)")
    g.add_argument("--prod-ps", type=float, default=500.0, help="production run at each T (ps)")
    g.add_argument("--thermostat", choices=["bussi", "langevin"], default="bussi", help="production thermostat")
    g.add_argument("--taut", type=float, default=100.0, help="Bussi time constant (fs)")
    g = p.add_argument_group("MD settings")
    g.add_argument("--timestep", type=float, default=1.0, help="fs")
    g.add_argument("--friction", type=float, default=0.01, help="Langevin friction (1/fs)")
    g.add_argument("--interval", type=int, default=100, help="steps between saved frames and log lines")
    g.add_argument("--fix-bottom", type=float, default=2.0, help="freeze LCO atoms within this distance of the LCO bottom (A)")
    g.add_argument("--min-dist", type=float, default=0.7, help="stop if two atoms come closer than this (A)")
    g.add_argument("--seed", type=int, default=0, help="random seed (offset by T for each production run)")
    add_model_args(p, dtype_default="float32", uma_settings_default="default", cueq=True)
    return resolve_model_args(p.parse_args(argv))


# =============================================================================
# Start structure
# =============================================================================
def pick_start(scan: Path, name: str | None) -> tuple[str, float | None]:
    """Name and relaxed energy of the start structure (lowest converged energy in results.csv)."""
    with open(scan / "results.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("E_relaxed_eV")]
    if not rows:
        raise SystemExit(f"no relaxed structures in {scan}/results.csv - run relax_interface_scan.py first")
    if name:
        match = [r for r in rows if r["name"] == name]
        if not match:
            raise SystemExit(f"{name} has no relaxed structure in results.csv")
        return name, float(match[0]["E_relaxed_eV"])
    rows.sort(key=lambda r: float(r["E_relaxed_eV"]))
    conv = [r for r in rows if r.get("converged") == "True"]
    if conv and conv[0] is not rows[0]:
        print(f"note: the lowest structure {rows[0]['name']} did not converge; using {conv[0]['name']}")
    best = conv[0] if conv else rows[0]
    return best["name"], float(best["E_relaxed_eV"])


def frozen_indices(atoms, is_lpo: np.ndarray, thickness: float) -> np.ndarray:
    if thickness <= 0:
        return np.array([], dtype=int)
    lco = np.nonzero(~is_lpo)[0]
    z = atoms.positions[lco, 2]
    return lco[z < z.min() + thickness]


# =============================================================================
# MD machinery
# =============================================================================
def n_steps(ps: float, args) -> int:
    """Number of steps for `ps`, rounded up to whole frames."""
    steps = math.ceil(ps * 1000.0 / args.timestep)
    return int(math.ceil(steps / args.interval) * args.interval)


def make_dyn(atoms, kind: str, T: float, args, rng):
    dt = args.timestep * units.fs
    if kind == "bussi":
        return Bussi(atoms, dt, temperature_K=T, taut=args.taut * units.fs, rng=rng)
    return Langevin(atoms, dt, temperature_K=T, friction=args.friction / units.fs, fixcm=False, rng=rng)


def last_frame(path: Path):
    """(last complete frame, number of frames) of a trajectory, or (None, 0)."""
    if not path.exists():
        return None, 0
    try:
        with Trajectory(str(path)) as traj:
            for k in range(len(traj) - 1, -1, -1):     # skip a frame cut off by a killed job
                try:
                    return traj[k], k + 1
                except Exception:
                    continue
    except Exception:
        pass
    return None, 0


def check_sane(atoms, T_max: float, args) -> None:
    """Stop on atoms that nearly overlap, or a temperature far above the highest target of the phase."""
    i, j = neighbor_list("ij", atoms, args.min_dist)
    if len(i):
        raise RuntimeError(f"atoms {i[0]} and {j[0]} are closer than {args.min_dist} A - the model has probably "
                           f"failed; the trajectory so far is kept")
    T = atoms.get_temperature()
    if T > 3.0 * max(T_max, 300.0):
        raise RuntimeError(f"temperature {T:.0f} K, highest target {T_max:.0f} K - the run is unstable")


def run_phase(d: Path, name: str, start, calc, args, steps: int, T_of_step, kind: str, rng):
    """Run, or resume, one MD phase in directory d and return the final atoms (with momenta).
    `start` carries the constraints; zero momenta mean: draw velocities at T_of_step(0)."""
    traj_path, done = d / f"{name}.traj", d / f"{name}.done"
    atoms = start.copy()
    last, n_frames = last_frame(traj_path)
    step0, resumed = 0, False
    if last is not None:
        step0, resumed = (n_frames - 1) * args.interval, True
        atoms.positions = last.positions
        atoms.set_momenta(last.get_momenta())
        if step0 >= steps:                              # finished (a longer run can extend it later)
            done.touch()
            return atoms
        print(f"  {d.name}/{name}: resuming at step {step0} of {steps}")
    elif not np.any(atoms.get_momenta()):
        thermalize_momenta(atoms, T_of_step(0), exact_temperature=True, rng=rng)
        Stationary(atoms)
    if traj_path.exists() and not resumed:              # unreadable leftovers of an earlier attempt
        traj_path.rename(traj_path.with_suffix(".traj.bak"))
    done.unlink(missing_ok=True)
    atoms.calc = calc
    remaining = steps - step0
    if remaining > 0:
        dyn = make_dyn(atoms, kind, T_of_step(step0), args, rng)
        traj = Trajectory(str(traj_path), "a" if resumed else "w", atoms, properties=["energy"])

        def write_frame():
            if not (resumed and dyn.nsteps == 0):       # the first frame of a resumed run is already saved
                traj.write()
        dyn.attach(write_frame, interval=args.interval)
        dyn.attach(MDLogger(dyn, atoms, str(d / f"{name}.log"), header=not resumed, mode="a"),
                   interval=args.interval)
        t0, done_steps, reported = time.time(), 0, False
        T_max = max(T_of_step(s) for s in range(0, steps + 1, args.interval))
        try:
            while done_steps < remaining:
                chunk = min(args.interval, remaining - done_steps)
                T = T_of_step(step0 + done_steps)
                if kind == "langevin":
                    dyn.set_temperature(temperature_K=T)
                dyn.run(chunk)
                done_steps += chunk
                check_sane(atoms, T_max, args)
                elapsed = time.time() - t0
                if not reported and (done_steps * args.timestep >= 1000.0 or done_steps == remaining):
                    rate = done_steps * args.timestep / 1000.0 / elapsed * 86400.0
                    print(f"  {d.name}/{name}: {rate:.1f} ps/day "
                          f"(~{(remaining - done_steps) * args.timestep / 1000.0 / rate * 24:.1f} h left in this phase)")
                    reported = True
        finally:
            traj.close()
    done.touch()
    return atoms


# =============================================================================
# Stages
# =============================================================================
def anneal(start, calc, args, out: Path, e_start_scan) -> None:
    d = out / "anneal"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "anneal.json").exists():
        print("anneal: done earlier (anneal/anneal.json)")
        return
    ps = [args.anneal_equil_ps, args.heat_ps, args.hold_ps, args.cool_ps]
    edges = np.cumsum([0] + [n_steps(x, args) for x in ps])     # step boundaries of the four parts
    Tl, Ta = args.T_low, args.anneal_T

    def T_of_step(s):
        if s < edges[1]:
            return Tl
        if s < edges[2]:
            return Tl + (Ta - Tl) * (s - edges[1]) / max(edges[2] - edges[1], 1)
        if s < edges[3]:
            return Ta
        return Ta + (Tl - Ta) * (s - edges[3]) / max(edges[4] - edges[3], 1)

    print(f"anneal: {Tl:.0f} K {ps[0]:g} ps, heat {ps[1]:g} ps, {Ta:.0f} K {ps[2]:g} ps, cool {ps[3]:g} ps "
          f"({int(edges[-1])} steps)")
    (d / "run.json").write_text(json.dumps(dict(T_low=Tl, anneal_T=Ta, timestep_fs=args.timestep, interval=args.interval,
                                                friction_per_fs=args.friction, seed=args.seed,
                                                ps=dict(equil=ps[0], heat=ps[1], hold=ps[2], cool=ps[3])), indent=2))
    rng = np.random.default_rng(args.seed)
    atoms = run_phase(d, "anneal", start, calc, args, int(edges[-1]), T_of_step, "langevin", rng)

    print("anneal: relaxing the cooled structure (all atoms, full precision)")
    calc_relax = get_calculator(args, precise=True) if needs_precise_copy(args) else calc
    ref = start.copy()
    ref.set_constraint()
    ref.calc = calc_relax
    e_start = float(ref.get_potential_energy())
    relaxed = atoms.copy()
    relaxed.set_constraint()
    relaxed.set_momenta(np.zeros((len(relaxed), 3)))
    relaxed.calc = calc_relax
    opt = FIRE(relaxed, logfile=str(d / "relax.log"))
    conv = opt.run(fmax=args.fmax, steps=args.max_steps)
    if conv is None:
        conv = float(np.linalg.norm(relaxed.get_forces(), axis=1).max()) < args.fmax
    e_ann = float(relaxed.get_potential_energy())
    write(str(d / "POSCAR_annealed"), relaxed, format="vasp", direct=True)
    res = dict(E_start_eV=e_start, E_start_scan_eV_per_cell=e_start_scan, E_annealed_relaxed_eV=e_ann,
               dE_eV=e_ann - e_start, relax_converged=bool(conv), relax_steps=int(opt.nsteps),
               anneal_T=Ta, T_low=Tl, ps=dict(equil=ps[0], heat=ps[1], hold=ps[2], cool=ps[3]))
    (d / "anneal.json").write_text(json.dumps(res, indent=2))
    print(f"anneal: E {e_start:.4f} (start) -> {e_ann:.4f} eV (annealed, relaxed), dE {e_ann - e_start:+.4f} eV")
    if e_ann > e_start:
        print("  the annealed structure is higher in energy than the start; consider --from relaxed")


def production(start, calc, args, out: Path, T: float) -> None:
    d = out / f"T{int(round(T)):04d}"
    d.mkdir(parents=True, exist_ok=True)
    kind = args.thermostat
    if kind == "bussi" and Bussi is None:
        print("  this ase has no Bussi thermostat; using Langevin")
        kind = "langevin"
    info = dict(T=T, timestep_fs=args.timestep, interval=args.interval, equil_ps=args.equil_ps,
                prod_ps=args.prod_ps, thermostat=kind, taut_fs=args.taut, friction_per_fs=args.friction,
                start=args.start_from, seed=args.seed + int(round(T)))
    old = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else None
    if old and any(old.get(k) != info[k] for k in ("timestep_fs", "interval", "start")):
        raise SystemExit(f"{d}: existing run used different --timestep/--interval/--from; use another --outdir")
    (d / "run.json").write_text(json.dumps(info, indent=2))
    print(f"production at {T:.0f} K: {args.equil_ps:g} ps Langevin + {args.prod_ps:g} ps {kind}")
    rng = np.random.default_rng(info["seed"])
    if last_frame(d / "prod.traj")[0] is None:          # once production has started, equilibration is over
        start = run_phase(d, "equil", start, calc, args, n_steps(args.equil_ps, args), lambda s: T, "langevin", rng)
    run_phase(d, "prod", start, calc, args, n_steps(args.prod_ps, args), lambda s: T, kind, rng)
    print(f"  {d.name}: done")


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> None:
    args = parse_args(argv)
    scan = Path(args.scan_dir)
    out = Path(args.outdir) if args.outdir else scan.resolve().parent / "md"
    out.mkdir(parents=True, exist_ok=True)
    scan_info = json.loads((scan / "scan_info.json").read_text())
    name, e_scan = pick_start(scan, args.structure)
    start = read(str(scan / name / "POSCAR_relaxed"), format="vasp")
    start.set_constraint()
    start.set_momenta(np.zeros((len(start), 3)))
    if len(start) != scan_info["n_atoms"]:
        raise SystemExit("POSCAR_relaxed and scan_info.json disagree on the number of atoms")
    is_lpo = np.zeros(len(start), dtype=bool)
    is_lpo[scan_info["lpo_indices"]] = True
    n_rep = args.repeat[0] * args.repeat[1]
    if n_rep > 1:                                   # ase repeat: copy k of atom i gets index k * n + i
        start = start.repeat((args.repeat[0], args.repeat[1], 1))
        is_lpo = np.tile(is_lpo, n_rep)
    frozen = frozen_indices(start, is_lpo, args.fix_bottom)
    start.set_constraint(FixAtoms(indices=frozen))
    (out / "start").mkdir(exist_ok=True)
    write(str(out / "start" / "POSCAR"), start, format="vasp", direct=True)

    info_path = out / "md_info.json"
    info = dict(scan_dir=str(scan.resolve()), start_structure=name, E_relaxed_scan_eV_per_cell=e_scan,
                repeat=args.repeat, n_atoms=len(start), lpo_indices=np.nonzero(is_lpo)[0].tolist(),
                frozen_indices=frozen.tolist(), fix_bottom_A=args.fix_bottom, area_A2=scan_info["area_A2"] * n_rep,
                **model_info(args))
    if info_path.exists():
        old = json.loads(info_path.read_text())
        if (old.get("start_structure") != name or old.get("repeat") != args.repeat
                or old.get("frozen_indices") != info["frozen_indices"]):
            raise SystemExit(f"{info_path} belongs to another start structure, --repeat or --fix-bottom; "
                             f"use another --outdir")
        if (old.get("model"), old.get("head"), bool(old.get("dispersion"))) != (args.model, args.head, args.dispersion):
            raise SystemExit(f"{info_path} belongs to {old.get('model')} (head {old.get('head')}); the trajectories "
                             f"cannot be continued with another model - use another --outdir")
    else:
        info_path.write_text(json.dumps(info, indent=2))
    print(f"start: {name} ({len(start)} atoms" + (f", {args.repeat[0]} x {args.repeat[1]} cells" if n_rep > 1 else "")
          + f"), {len(frozen)} frozen LCO atoms; output in {out}/")

    calc = get_calculator(args)
    if "anneal" in args.stages:
        anneal(start, calc, args, out, e_scan)
    if "production" in args.stages:
        if args.start_from == "annealed":
            path = out / "anneal" / "POSCAR_annealed"
            if not path.exists():
                raise SystemExit(f"{path} not found - run the anneal first (--stages anneal) or use --from relaxed")
            prod_start = read(str(path), format="vasp")
            prod_start.set_constraint(FixAtoms(indices=frozen))
            prod_start.set_momenta(np.zeros((len(prod_start), 3)))
        else:
            prod_start = start
        for T in args.temperatures:
            production(prod_start, calc, args, out, T)


if __name__ == "__main__":
    main()

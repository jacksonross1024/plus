#!/usr/bin/env python3
"""Minimal Hall-potential readout example for ``mumaxplus.poisson``.

Solves one contact-potential frame, then reports the per-contact transverse
Hall voltages from virtual Hall-bar probes (volts, no current normalization).

``--ahe`` uses uniform out-of-plane ``m``. ``--the`` uses a skyrmion-like
texture so the winding is nonzero; add ``--uniform`` to force uniform ``m``
(THE fields ~0, Hall voltages match AHE-only).
"""

from __future__ import annotations

import argparse

import numpy as np


def _uniform_m(shape):
    m = np.zeros(shape, dtype=np.float32)
    m[2, ...] = 1.0
    return m


def _skyrmion_like_m(shape, radius_cells=10.0):
    _, nz, ny, nx = shape
    m = np.zeros(shape, dtype=np.float32)
    cy = 0.5 * (ny - 1)
    cx = 0.5 * (nx - 1)
    yy, xx = np.meshgrid(
        np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij"
    )
    dx = xx - cx
    dy = yy - cy
    rho = np.sqrt(dx * dx + dy * dy)
    psi = np.pi * np.clip(rho / float(radius_cells), 0.0, 1.0)
    mz = np.cos(psi)
    mr = np.sin(psi)
    inv = np.where(rho > 1e-6, 1.0 / rho, 0.0)
    m[0, ...] = mr * dx * inv
    m[1, ...] = mr * dy * inv
    m[2, ...] = mz
    norm = np.linalg.norm(m, axis=0, keepdims=True)
    m = np.where(norm > 1e-12, m / np.maximum(norm, 1e-12), 0.0)
    return np.ascontiguousarray(m, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nt", type=int, default=2)
    parser.add_argument("--num-contacts", type=int, default=1)
    parser.add_argument("--ahe", action="store_true", help="enable AHE transport")
    parser.add_argument("--ahe-ratio", type=float, default=0.05)
    parser.add_argument("--the", action="store_true", help="enable THE transport")
    parser.add_argument(
        "--the-ratio",
        type=float,
        default=0.2,
        help="dimensionless THE scale (solver default is 0; used only with --the)",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="force uniform m even when --the is set (THE fields should vanish)",
    )
    args = parser.parse_args()

    import mumaxplus.poisson as poisson

    world = poisson.build_fgat_world_spec(
        num_contacts=args.num_contacts,
        shape=(4, 96, 96),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=20 if args.num_contacts == 1 else 12,
        contact_spacing_cells=None if args.num_contacts == 1 else 12,
        contact_edge_depth_cells=10,
        void_locations=None,
    )
    potentials = np.full((args.nt, args.num_contacts), 1e-3, dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        world=world,
        contact_potentials=potentials,
        ahe_enabled=args.ahe,
        ahe_ratio=args.ahe_ratio if args.ahe else 0.0,
        the_enabled=args.the,
        the_ratio=args.the_ratio if args.the else 0.0,
        skip_threshold=0.0,
    )

    geom = poisson.resolve_hall_contact_geometry(world)
    print("Hall geometry:")
    print(f"  num_contacts={geom.num_contacts}")
    print(f"  x_ranges={geom.x_ranges}")
    print(f"  low_y_ranges={geom.low_y_ranges}")
    print(f"  high_y_ranges={geom.high_y_ranges}")
    print(f"  z_layers={geom.z_layers}")
    print(f"  cells/contact low={[len(a) for a in geom.low_y_indices]}")
    print(f"  cells/contact high={[len(a) for a in geom.high_y_indices]}")

    m = None
    if solver.transport_enabled:
        if args.the and not args.uniform:
            m = _skyrmion_like_m(solver.output_shape)
        else:
            m = _uniform_m(solver.output_shape)

    frame = solver.iterate(magnetization=m)
    print(
        f"step={frame.stats.step} skipped={frame.stats.skipped} "
        f"iterations={frame.stats.iterations} residual_rel={frame.stats.residual_rel:.3e}"
    )

    v_hall = solver.hall_potentials()
    print("Hall voltages [V]:", v_hall)
    comps = solver.hall_potentials(return_components=True)
    print("high_y means [V]:", comps.high_y_means)
    print("low_y means [V]:", comps.low_y_means)

    if args.the and m is not None:
        stats = solver.winding_stats()
        print(f"winding max |h|={stats['max_abs']:.4e} sum h_z={stats['sum_hz']:.4e}")
        h = solver.winding()
        print(f"winding shape={h.shape} max={float(np.max(np.abs(h))):.4e}")

    if (args.ahe or args.the) and m is not None:
        solver.reset()
        m_neg = -m
        solver.iterate(magnetization=m_neg)
        v_neg = solver.hall_potentials()
        print("Hall voltages with -m [V]:", v_neg)
        print("odd-in-m component [V]:", 0.5 * (v_hall - v_neg))


if __name__ == "__main__":
    main()

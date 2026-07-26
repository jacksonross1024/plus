#!/usr/bin/env python3
"""Minimal Hall-potential readout example for ``mumaxplus.poisson``.

Solves one contact-potential frame, then reports the per-contact transverse
Hall voltages from virtual Hall-bar probes (volts, no current normalization).
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nt", type=int, default=2)
    parser.add_argument("--num-contacts", type=int, default=1)
    parser.add_argument("--ahe", action="store_true", help="enable AHE transport")
    parser.add_argument("--ahe-ratio", type=float, default=0.05)
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
        m = np.zeros((3, solver.output_shape[1], solver.output_shape[2], solver.output_shape[3]), dtype=np.float32)
        m[2, ...] = 1.0

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

    if args.ahe and m is not None:
        solver.reset()
        m_neg = -m
        solver.iterate(magnetization=m_neg)
        v_neg = solver.hall_potentials()
        print("Hall voltages with -m [V]:", v_neg)
        print("odd-in-m component [V]:", 0.5 * (v_hall - v_neg))


if __name__ == "__main__":
    main()

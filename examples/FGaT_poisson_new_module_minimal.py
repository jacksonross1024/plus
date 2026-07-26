#!/usr/env python3
"""Minimal smoke test for ``mumaxplus.poisson`` (CUDA + mumax vector layout)."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=3)
    args = parser.parse_args()

    import mumaxplus.poisson as poisson

    potentials = np.full((args.nt, 3), 1e-3, dtype=np.float64)
    solver = poisson.CudaPoissonSolver(contact_potentials=potentials)

    print("Poisson world shape:", solver.world_shape)
    print("Poisson buffer shape:", solver.internal_output_shape)
    print("Poisson output shape:", solver.output_shape)

    frame = solver.iterate()
    assert frame.jmod.shape == solver.output_shape
    assert frame.jmod.shape[0] == 3
    assert frame.jcur.shape == frame.jmod.shape
    print(
        f"step={frame.stats.step} skipped={frame.stats.skipped} "
        f"iterations={frame.stats.iterations} residual_rel={frame.stats.residual_rel:.3e}"
    )
    print("jmod max abs:", float(np.max(np.abs(frame.jmod))))
    print("jcur max abs:", float(np.max(np.abs(frame.jcur))))

    v_hall = solver.hall_potentials()
    print("Hall voltages [V]:", v_hall)
    assert v_hall.shape == (solver.num_contacts,)


if __name__ == "__main__":
    main()

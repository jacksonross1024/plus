"""Domain-wall motion under Zhang-Li, Slonczewski, and combined spin-transfer torque.

A thin ferromagnetic wire (128 x 4 x 4 cells) with out-of-plane magnetization (+/-z)
is initialized with a domain wall at one third of the length (x), equilibrated,
then driven separately by:

1. Zhang-Li torque only (in-plane current density)
2. Slonczewski torque only (vertical current, fixed layer polarization)
3. Combined torque (split ``jcur_zl`` and ``jcur_stt``)

The domain-wall center along x is tracked versus time for each case.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import twodomain


def dw_position_x(magnet):
    """Estimate domain-wall position (m) along x from the averaged m_z profile."""
    m = magnet.magnetization.eval()
    mz = m[2].mean(axis=(0, 1))
    cx = magnet.world.cellsize[0]
    x = (np.arange(mz.shape[0]) + 0.5) * cx

    for i in range(mz.shape[0] - 1):
        if mz[i] * mz[i + 1] <= 0:
            denom = mz[i + 1] - mz[i]
            if abs(denom) < 1e-30:
                return x[i]
            return x[i] - mz[i] * (x[i + 1] - x[i]) / denom
    return x[mz.shape[0] // 2]


def setup_magnet():
    """Create wire, material parameters, and equilibrated domain-wall state."""
    nx, ny, nz = 128, 4, 4
    cx, cy, cz = 2e-9, 10e-9, 10e-9
    length = nx * cx

    world = World(cellsize=(cx, cy, cz))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))

    # Permalloy-like parameters (see examples/standardproblem5.py, MRAM_bit_switch.py)
    magnet.msat = 800e3
    magnet.aex = 13e-12
    magnet.alpha = 0.1
    magnet.ku1 = 1.0e5
    magnet.anisU = (0, 0, 1)

    wallpos = length / 3
    wallthick = 3 * cx
    # Use an x-directed wall moment so the initial wall is Neel-like.
    magnet.magnetization = twodomain(
        (0, 0, -1), (0, 1, 0), (0, 0, 1), wallpos, wallthick
    )

    print("Minimizing domain-wall structure...")
    magnet.minimize(tol=1e-6, nsamples=10)
    m_eq = magnet.magnetization.eval()
    x0 = dw_position_x(magnet)
    print(f"Equilibrated domain-wall position: {x0 * 1e9:.2f} nm")

    return world, magnet, m_eq, length


def run_dw_motion(world, magnet, m_eq, label, configure_torque):
    """Reset state, apply a torque configuration, and track DW position vs time."""
    magnet.magnetization = m_eq
    configure_torque(magnet)

    tmax = 2e-9
    nsteps = 200
    timepoints = np.linspace(0, tmax, nsteps + 1)
    quantities = {"dw_x": lambda: dw_position_x(magnet)}

    world.timesolver.time = 0.0
    print(f"Running {label}...")
    output = world.timesolver.solve(timepoints, quantities)

    t_ns = np.asarray(output["time"]) * 1e9
    x_nm = np.asarray(output["dw_x"]) * 1e9
    return t_ns, x_nm


def configure_zhang_li(magnet):
    magnet.enable_combined_spin_transfer_torque = False
    magnet.enable_zhang_li_torque = True
    magnet.enable_slonczewski_torque = False
    magnet.xi = 0.2
    magnet.pol = 1.0
    magnet.jcur = (6e11, 0.0, 0.0)
    magnet.jcur_stt = (0.0, 0.0, 0.0)
    magnet.jcur_zl = (0.0, 0.0, 0.0)


def configure_slonczewski(magnet):
    magnet.enable_combined_spin_transfer_torque = False
    magnet.enable_zhang_li_torque = False
    magnet.enable_slonczewski_torque = True
    magnet.pol = 0.57
    magnet.Lambda = 2.0
    magnet.epsilon_prime = 0.5
    magnet.fixed_layer = (0.0, 0.0, 1.0)
    magnet.fixed_layer_on_top = True
    area = magnet.grid.size[0] * magnet.grid.size[1] * magnet.world.cellsize[0] * magnet.world.cellsize[1]
    jz = -4e-3 / area
    magnet.jcur = (0.0, 0.0, jz)
    magnet.jcur_stt = (0.0, 0.0, 0.0)
    magnet.jcur_zl = (0.0, 0.0, 0.0)


def configure_combined(magnet):
    magnet.enable_combined_spin_transfer_torque = True
    magnet.enable_zhang_li_torque = True
    magnet.enable_slonczewski_torque = True
    magnet.xi = 0.2
    magnet.pol = 0.57
    magnet.Lambda = 2.0
    magnet.epsilon_prime = 0.5
    magnet.fixed_layer = (0.0, 0.0, 1.0)
    magnet.fixed_layer_on_top = True
    area = magnet.grid.size[0] * magnet.grid.size[1] * magnet.world.cellsize[0] * magnet.world.cellsize[1]
    magnet.jcur = (0.0, 0.0, 0.0)
    magnet.jcur_zl = (6e11, 0.0, 0.0)
    magnet.jcur_stt = (0.0, 0.0, -4e-3 / area)


def main():
    world, magnet, m_eq, length = setup_magnet()

    results = {}
    for label, configure in [
        ("Zhang-Li", configure_zhang_li),
        ("Slonczewski (STT)", configure_slonczewski),
        ("Combined", configure_combined),
    ]:
        results[label] = run_dw_motion(world, magnet, m_eq, label, configure)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (t_ns, x_nm) in results.items():
        ax.plot(t_ns, x_nm, "-", lw=2, label=label)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Domain-wall position (nm)")
    ax.set_title(
        f"Domain-wall motion (wire {length*1e9:.0f} nm $\\times$ "
        f"{magnet.grid.size[1]} $\\times$ {magnet.grid.size[2]} cells, $m \\parallel z$)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).with_name("DW_combined_stt.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()

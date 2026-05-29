"""Tests for combined Zhang-Li + Slonczewski spin-transfer torque."""

import math

import numpy as np
import pytest

from mumaxplus import Ferromagnet, Grid, World


def _vortex_magnetization(grid):
    nx, ny, nz = grid.size
    m = np.zeros((3, nz, ny, nx))
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = (ix + 0.5) / nx - 0.5
                y = (iy + 0.5) / ny - 0.5
                r = math.hypot(x, y) + 1e-12
                m[0, iz, iy, ix] = -y / r
                m[1, iz, iy, ix] = x / r
                m[2, iz, iy, ix] = 0.2
    return m


def _configure_stt_magnet(magnet):
    magnet.msat = 800e3
    magnet.aex = 13e-12
    magnet.alpha = 0.1
    magnet.magnetization = _vortex_magnetization(magnet.grid)
    magnet.xi = 0.2
    magnet.pol = 0.57
    magnet.Lambda = 2.0
    magnet.epsilon_prime = 0.5
    magnet.jcur_stt = (0.0, 0.0, -4e11)
    magnet.jcur_zl = (6e11, 0.0, 0.0)
    magnet.fixed_layer = (
        math.cos(20 * np.pi / 180),
        math.sin(20 * np.pi / 180),
        0.0,
    )


@pytest.fixture
def combined_magnet():
    world = World(cellsize=(5e-9, 5e-9, 2e-9))
    magnet = Ferromagnet(world, Grid((20, 20, 4)))
    _configure_stt_magnet(magnet)
    magnet.enable_combined_spin_transfer_torque = True
    magnet.enable_zhang_li_torque = True
    magnet.enable_slonczewski_torque = True
    return magnet


def test_combined_equals_sum_of_models(combined_magnet):
    """Combined STT field equals Zhang-Li-only plus Slonczewski-only."""
    magnet = combined_magnet
    t_combined = magnet.spin_transfer_torque.eval()

    magnet.enable_combined_spin_transfer_torque = False
    magnet.enable_slonczewski_torque = False
    magnet.enable_zhang_li_torque = True
    magnet.jcur = magnet.jcur_zl.eval()
    t_zl = magnet.spin_transfer_torque.eval()

    magnet.enable_zhang_li_torque = False
    magnet.enable_slonczewski_torque = True
    magnet.jcur = magnet.jcur_stt.eval()
    t_stt = magnet.spin_transfer_torque.eval()

    err = np.max(np.linalg.norm(t_combined - (t_zl + t_stt), axis=0))
    scale = np.max(np.linalg.norm(t_combined, axis=0)) + 1e-30
    assert err < 1e-6 * scale


def test_combined_rejects_jcur_when_enabled():
    magnet = Ferromagnet(World((5e-9, 5e-9, 2e-9)), Grid((12, 12, 2)))
    _configure_stt_magnet(magnet)
    magnet.enable_combined_spin_transfer_torque = True
    magnet.jcur = (2e11, 0.0, -3e11)

    with pytest.raises(Exception):
        magnet.spin_transfer_torque.eval()


def test_combined_allows_partial_split_current():
    """Only Slonczewski or only Zhang–Li may be driven in combined mode."""
    magnet = Ferromagnet(World((5e-9, 5e-9, 2e-9)), Grid((8, 8, 2)))
    _configure_stt_magnet(magnet)
    magnet.enable_combined_spin_transfer_torque = True
    magnet.jcur_stt = (0.0, 0.0, -4e11)
    magnet.jcur_zl = (0.0, 0.0, 0.0)

    t_stt_only = magnet.spin_transfer_torque.eval()

    magnet.jcur_stt = (0.0, 0.0, 0.0)
    magnet.jcur_zl = (6e11, 0.0, 0.0)
    t_zl_only = magnet.spin_transfer_torque.eval()

    assert np.max(np.linalg.norm(t_stt_only, axis=0)) > 0
    assert np.max(np.linalg.norm(t_zl_only, axis=0)) > 0


def test_combined_rejects_mixed_jcur_and_split():
    magnet = Ferromagnet(World((5e-9, 5e-9, 2e-9)), Grid((8, 8, 2)))
    _configure_stt_magnet(magnet)
    magnet.enable_combined_spin_transfer_torque = True
    magnet.jcur = (1e11, 0.0, 0.0)
    magnet.jcur_stt = (2e11, 0.0, 0.0)
    magnet.jcur_zl = (3e11, 0.0, 0.0)

    with pytest.raises(Exception):
        magnet.spin_transfer_torque.eval()

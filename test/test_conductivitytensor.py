import numpy as np

from mumaxplus import Ferromagnet, Grid, World


def compute_conductivity_tensor(magnet):
    """Computes the conductivity tensor taking AMR into account"""
    mx, my, mz = magnet.magnetization.get()
    conductivity = magnet.conductivity.eval()[0]
    amr_ratio = magnet.amr_ratio.eval()[0]

    prefactor = 6.0 * amr_ratio / (6.0 + amr_ratio)
    cond_XX = conductivity * (1.0 - prefactor * (mx * mx - 1.0 / 3.0))
    cond_YY = conductivity * (1.0 - prefactor * (my * my - 1.0 / 3.0))
    cond_ZZ = conductivity * (1.0 - prefactor * (mz * mz - 1.0 / 3.0))
    cond_XY = -conductivity * prefactor * mx * my
    cond_XZ = -conductivity * prefactor * mx * mz
    cond_YZ = -conductivity * prefactor * my * mz

    return np.array([cond_XX, cond_YY, cond_ZZ, cond_XY, cond_XZ, cond_YZ])


class TestConductivityTensor:
    def test_conductivity_tensor(self):

        world = World((1e3, 2e3, 3e3))
        magnet = Ferromagnet(world, Grid((4, 4, 1)))
        magnet.amr_ratio = 0.2
        magnet.conductivity = 9.1

        result = magnet.conductivity_tensor.eval()
        wanted = compute_conductivity_tensor(magnet)

        relative_error = np.abs(result - wanted) / np.abs(wanted)
        max_relative_error = np.max(relative_error)

        assert max_relative_error < 1e-4

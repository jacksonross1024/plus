"""Export jmod from a 2 Pt + 8×5 nm FM Poisson world.

Poisson cells are 5 nm. The LLG receives either that block packed into 2, 4,
or 8 cells, or cells that are 2 nm thick. Profiles:

- exponential decay of all 8 FM layers
- one average value for all 8 FM layers
- one average value for the first 4 FM layers

The printed ``sheet`` is the thickness-integrated injection for a unit
interface current of 1 A/m². ``average`` rows that share a source depth share
that sheet for every LLG layer count.
"""

import numpy as np

import mumaxplus.poisson as poisson

POISSON_CZ = 5e-9
DECAY_LENGTH = 8e-9
N_PT = 2
N_FM = 8
MUMAX_CZ = 2e-9


def build_world():
    """2 Pt layers and 8 FM layers, every cell 5 nm thick."""

    world = poisson.build_fgat_world_spec(
        n_pt_layers=N_PT,
        n_fm_layers=N_FM,
        nx=32,
        ny=32,
        cellsize=(20e-9, 20e-9, POISSON_CZ),
        sigma_pt=2.0e6,
        sigma_fm=3.93e5,
        theta_sh=0.2,
        decay_length=DECAY_LENGTH,
        void_locations=(),
        num_contacts=1,
        contact_edge_depth_cells=10,
    )
    if world.shape != (N_PT + N_FM, 32, 32):
        raise RuntimeError(f"unexpected Poisson shape {world.shape}")
    if world.first_r2_layer != N_PT:
        raise RuntimeError("FM block does not start after 2 Pt layers")
    if world.cellsize[2] != POISSON_CZ:
        raise RuntimeError("Poisson cz is not 5 nm")
    return world


def unit_fm_stack(n_fm):
    """Native per-cell jmod for an interface current of 1 A/m²."""

    jmod = np.zeros((3, n_fm, 1, 1), dtype=np.float32)
    jcur = np.zeros_like(jmod)
    for layer in range(n_fm):
        factor = poisson.fm_injection_decay_factor(layer, POISSON_CZ, DECAY_LENGTH)
        jmod[0, layer, 0, 0] = np.float32(factor)
        jcur[0, layer, 0, 0] = np.float32(layer + 1)
    return jmod, jcur


def export_jmod(n_source, n_out, profile, fm_cellsize_z=None):
    """Map the first ``n_source`` Poisson FM layers onto ``n_out`` LLG cells."""

    layers = tuple(range(n_source))
    fm_cz = poisson.resolve_fm_cellsize_z(
        layers,
        n_out,
        POISSON_CZ,
        fm_cellsize_z=fm_cellsize_z,
    )
    jmod, jcur = unit_fm_stack(N_FM)
    exported, _exported_jcur = poisson.map_fm_currents(
        jmod,
        jcur,
        source_layers=layers,
        n_out=n_out,
        poisson_cz=POISSON_CZ,
        fm_cz=fm_cz,
        decay_length=DECAY_LENGTH,
        jmod_profile=profile,
    )
    values = np.asarray(exported[0, :, 0, 0], dtype=np.float64)
    film = n_out * fm_cz
    integrated = float(np.sum(values) * fm_cz)
    if profile == "exponential":
        expected = poisson.exponential_integral(0.0, film, DECAY_LENGTH)
        if n_out > 1 and not np.all(np.diff(values) < 0):
            raise RuntimeError("exponential profile does not fall with depth")
    else:
        source_depth = n_source * POISSON_CZ
        expected = poisson.exponential_integral(0.0, source_depth, DECAY_LENGTH)
        if not np.allclose(values, values[0]):
            raise RuntimeError("average profile is not the same on every LLG cell")
    if not np.isclose(integrated, expected, rtol=1e-5):
        raise RuntimeError(
            f"{profile} n_source={n_source} n_out={n_out} "
            f"sheet {integrated} != {expected}"
        )
    return fm_cz, values, integrated


def solver_call(n_source, n_out, profile, fm_cellsize_z=None):
    """Keyword arguments for the same export on :class:`CudaPoissonSolver`."""

    kwargs = {
        "fm_nz": f"0:{n_source}",
        "fm_mumax_nz": n_out,
        "jmod_profile": profile,
    }
    if fm_cellsize_z is not None:
        kwargs["fm_cellsize_z"] = fm_cellsize_z
    return kwargs


def main() -> None:
    world = build_world()
    print(
        f"Poisson world (nz, ny, nx)={world.shape}, "
        f"Pt={world.first_r2_layer}, FM={world.shape[0] - world.first_r2_layer}, "
        f"cz={world.cellsize[2]:.3e} m, decay_length={world.decay_length:.3e} m"
    )
    print(
        f"{'profile':<12} {'source':>6} {'llg':>4} {'mumax':>8} "
        f"{'cz':>10} {'j0':>10} {'jN':>10} {'sheet':>10}"
    )
    layouts = (
        ("exponential", 8),
        ("average", 8),
        ("average", 4),
    )
    for profile, n_source in layouts:
        for n_out in (2, 4, 8):
            for label, fm_cz_override in (("packed", None), ("2 nm", MUMAX_CZ)):
                fm_cz, values, integrated = export_jmod(
                    n_source, n_out, profile, fm_cz_override
                )
                print(
                    f"{profile:<12} {n_source:6d} {n_out:4d} {label:>8} "
                    f"{fm_cz:10.3e} {values[0]:10.4f} {values[-1]:10.4f} "
                    f"{integrated:10.4e}"
                )
                _ = solver_call(n_source, n_out, profile, fm_cz_override)
    print("Passed sheet and profile checks for all 18 exports.")


if __name__ == "__main__":
    main()

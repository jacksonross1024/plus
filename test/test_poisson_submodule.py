from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.modules.setdefault("pyvista", types.ModuleType("pyvista"))

import mumaxplus.poisson as poisson
import mumaxplus.poisson.solver as solver_module


class _FakeImpl:
    def __init__(self, contact_potentials, buffer_shape=(2, 4, 5, 3), first_r2_layer=1):
        self._potentials = np.asarray(contact_potentials, dtype=np.float64)
        self.current_step = 0
        self.n_steps = self._potentials.shape[0]
        nz, ny, nx, _ = buffer_shape
        self.buffer_shape = buffer_shape
        self.output_shape = (3, nz, ny, nx)
        self.world_shape = (3, 4, 5)
        self.cellsize = (1.0, 2.0, 5e-9)
        self.theta_sh = 0.2
        self.decay_length = 8e-9
        self.unknown_count = 17
        self.first_r2_layer = first_r2_layer

    @property
    def exhausted(self):
        return self.current_step >= self.n_steps

    def reset(self):
        self.current_step = 0

    def iterate(self):
        step = self.current_step
        skipped = bool(np.max(np.abs(self._potentials[step])) < 1e-5)
        value = 0.0 if skipped else float(step + 1)
        nz, ny, nx = self.buffer_shape[:3]
        frame = np.empty((3, nz, ny, nx), dtype=np.float32)
        for k in range(nz):
            layer_value = 0.0 if skipped else float(value + k)
            frame[:, k, ...] = layer_value
        self.current_step += 1
        return {
            "jmod": frame,
            "jcur": frame.copy(),
            "stats": {
                "step": step,
                "skipped": skipped,
                "iterations": 0 if skipped else 3,
                "residual_initial": 0.0,
                "residual": 0.0,
                "rhs_inf": 0.0,
                "residual_rel": 0.0,
                "elapsed_s": 0.0,
                "note": "",
            },
        }


class _FakeRawPoissonCudaSolver:
    last_from_arrays = None

    @staticmethod
    def from_manifest(
        manifest_path,
        contact_potentials,
        tolerance,
        max_iterations,
        skip_threshold,
        slice_x,
        slice_y,
        slice_z,
        cuda_tol_batch_first,
        cuda_tol_batch_next,
    ):
        return _FakeImpl(contact_potentials)

    @staticmethod
    def from_arrays(
        nx,
        ny,
        nz,
        cx,
        cy,
        cz,
        first_r2_layer,
        theta_sh,
        decay_length,
        region,
        contact_id,
        sigma,
        contact_potentials,
        tolerance,
        max_iterations,
        skip_threshold,
        slice_x,
        slice_y,
        slice_z,
        cuda_tol_batch_first,
        cuda_tol_batch_next,
    ):
        _FakeRawPoissonCudaSolver.last_from_arrays = {
            "shape": (nz, ny, nx),
            "cellsize": (cx, cy, cz),
            "region_dtype": region.dtype,
            "contact_id_dtype": contact_id.dtype,
            "sigma_dtype": sigma.dtype,
        }
        n_fm = max(1, int(nz) - int(first_r2_layer))
        return _FakeImpl(
            contact_potentials,
            buffer_shape=(n_fm, ny, nx, 3),
            first_r2_layer=first_r2_layer,
        )

    @staticmethod
    def from_signal_file(*args, **kwargs):
        return _FakeImpl(np.ones((2, 3), dtype=np.float64))


@pytest.fixture
def fake_raw_solver(monkeypatch):
    monkeypatch.setattr(
        solver_module._cpp,
        "PoissonCudaSolver",
        _FakeRawPoissonCudaSolver,
        raising=False,
    )
    return _FakeRawPoissonCudaSolver


def test_poisson_submodule_imports():
    assert hasattr(poisson, "CudaPoissonSolver")
    assert hasattr(poisson, "WorldSpec")


def test_default_world_path_is_packaged():
    path = Path(poisson.default_world_path())
    assert path.name == "FGaT-amr-sine-poisson-world.txt"
    assert path.is_file()


def test_load_contact_potentials(tmp_path):
    path = tmp_path / "contacts.txt"
    path.write_text("1 2 3\n4 5 6\n", encoding="ascii")
    values = poisson.load_contact_potentials(str(path))
    assert values.shape == (2, 3)
    assert values.dtype == np.float64
    np.testing.assert_allclose(values[1], [4.0, 5.0, 6.0])


def test_manifest_solver_iterate_and_compatibility(fake_raw_solver):
    potentials = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(contact_potentials=potentials)

    assert solver.check_compatible((2, 4, 5), cellsize=(1.0, 2.0, 5e-9))["n_steps"] == 2
    assert solver.current_step == 0

    first = solver.iterate()
    assert first.jmod.shape == (3, 2, 4, 5)
    assert first.jmod.dtype == np.float32
    assert first.stats.skipped
    assert solver.current_step == 1

    second = solver.iterate()
    assert not second.stats.skipped
    np.testing.assert_allclose(second.jcur[:, 0, ...], 2.0)
    np.testing.assert_allclose(second.jcur[:, 1, ...], 3.0)
    assert solver.exhausted


def test_check_compatible_rejects_nz_mismatch(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(contact_potentials=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="Poisson export shape"):
        solver.check_compatible((10, 4, 5))


def test_check_compatible_rejects_xy_mismatch(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(contact_potentials=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="incompatible"):
        solver.check_compatible((2, 4, 4))


def test_parse_fm_nz_spec():
    assert poisson.parse_fm_nz_spec("0", 4, 1) == (0,)
    assert poisson.parse_fm_nz_spec("1", 4, 10) == (1,)
    assert poisson.parse_fm_nz_spec("0:2", 4, 2) == (0, 1)
    with pytest.raises(ValueError, match="mumax FM has 1"):
        poisson.parse_fm_nz_spec("0:2", 4, 1)


def test_map_layer_broadcast(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        fm_nz="0",
        fm_mumax_nz=10,
    )
    frame = solver.iterate()
    assert frame.jmod.shape == (3, 10, 4, 5)
    np.testing.assert_allclose(frame.jmod[:, 0, ...], frame.jmod[:, 5, ...])
    np.testing.assert_allclose(frame.jmod, 1.0)


def test_map_layer_range(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        fm_nz="0:2",
        fm_mumax_nz=2,
    )
    frame = solver.iterate()
    np.testing.assert_allclose(frame.jmod[:, 0, ...], 1.0)
    np.testing.assert_allclose(frame.jmod[:, 1, ...], 2.0)


def test_map_height_resample(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        fm_height=10e-9,
        fm_mumax_nz=1,
    )
    frame = solver.iterate()
    assert frame.jmod.shape == (3, 1, 4, 5)
    np.testing.assert_allclose(frame.jmod[:, 0, ...], 1.5)


def test_parse_fm_export_layers():
    assert poisson.parse_fm_export_layers(None) is None
    assert poisson.parse_fm_export_layers("0") == 0
    assert poisson.parse_fm_export_layers("0,2") == (0, 2)


def test_world_spec_uses_in_memory_arrays(fake_raw_solver):
    shape = (2, 3, 4)
    region = np.ones(shape, dtype=np.int8)
    contact_id = np.zeros(shape, dtype=np.int8)
    sigma = np.ones(shape, dtype=np.float32)
    spec = poisson.WorldSpec(
        shape=shape,
        cellsize=(1e-9, 2e-9, 3e-9),
        first_r2_layer=1,
        theta_sh=0.2,
        decay_length=8e-9,
        region=region,
        contact_id=contact_id,
        sigma=sigma,
    )
    solver = poisson.CudaPoissonSolver(
        world=spec, contact_potentials=np.zeros((1, 3))
    )
    assert solver.output_shape == (3, 1, 3, 4)
    assert solver.fm_layer_count == 1
    assert fake_raw_solver.last_from_arrays["shape"] == shape
    assert fake_raw_solver.last_from_arrays["region_dtype"] == np.dtype("int8")
    assert fake_raw_solver.last_from_arrays["sigma_dtype"] == np.dtype("float32")

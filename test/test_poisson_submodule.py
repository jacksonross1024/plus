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
        self.num_contacts = self._potentials.shape[1]
        self.transport_enabled = False
        self.amr_enabled = False
        self.ahe_enabled = False
        self.amr_ratio = 0.0
        self.ahe_ratio = 0.0
        self.picard_sweeps = 2
        self.fm_layer_count = nz
        self.last_magnetization = None
        self._hall_high = None
        self._hall_low = None
        self._hall_voltages = None
        self._hall_frame_available = False
        self._last_skipped = False
        self.hall_configure_count = 0

    @property
    def exhausted(self):
        return self.current_step >= self.n_steps

    @property
    def hall_probes_configured(self):
        return self._hall_high is not None

    @property
    def hall_frame_available(self):
        return self._hall_frame_available

    @property
    def last_frame_skipped(self):
        return self._last_skipped

    def reset(self):
        self.current_step = 0
        self._hall_frame_available = False
        self._last_skipped = False
        self._hall_voltages = None

    def set_hall_probe_indices(self, high_y, low_y):
        self._hall_high = [np.asarray(a, dtype=np.int64) for a in high_y]
        self._hall_low = [np.asarray(a, dtype=np.int64) for a in low_y]
        self.hall_configure_count += 1
        if self._hall_frame_available:
            self._update_hall_voltages()

    def _update_hall_voltages(self):
        n = self.num_contacts
        if self._hall_high is None:
            return
        if self._last_skipped:
            self._hall_voltages = np.zeros(n, dtype=np.float64)
            return
        # Deterministic fake: V_hall[c] = (c + 1) * 1e-3 after a solved frame.
        self._hall_voltages = np.asarray(
            [(c + 1) * 1e-3 for c in range(n)], dtype=np.float64
        )

    def hall_potentials(self):
        if self._hall_high is None:
            raise RuntimeError("Hall probe indices are not configured")
        if not self._hall_frame_available:
            raise RuntimeError("hall_potentials() requires at least one iterate() call")
        return np.asarray(self._hall_voltages, dtype=np.float64)

    def hall_potential_components(self):
        voltages = self.hall_potentials()
        n = len(voltages)
        return {
            "voltages": voltages,
            "high_y_means": voltages + 0.5e-3,
            "low_y_means": np.full(n, 0.5e-3, dtype=np.float64),
            "high_y_counts": [int(a.size) for a in self._hall_high],
            "low_y_counts": [int(a.size) for a in self._hall_low],
        }

    def _frame_for_step(self, step):
        skipped = bool(np.max(np.abs(self._potentials[step])) < 1e-5)
        value = 0.0 if skipped else float(step + 1)
        nz, ny, nx = self.buffer_shape[:3]
        frame = np.empty((3, nz, ny, nx), dtype=np.float32)
        for k in range(nz):
            layer_value = 0.0 if skipped else float(value + k)
            frame[:, k, ...] = layer_value
        return skipped, frame

    def iterate(self):
        step = self.current_step
        skipped, frame = self._frame_for_step(step)
        if self.transport_enabled and not skipped:
            raise RuntimeError(
                "magnetization is required when AMR/AHE transport is enabled"
            )
        self.current_step += 1
        self._hall_frame_available = True
        self._last_skipped = skipped
        self._update_hall_voltages()
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

    def iterate_with_magnetization(self, magnetization):
        self.last_magnetization = np.asarray(magnetization)
        step = self.current_step
        skipped, frame = self._frame_for_step(step)
        self.current_step += 1
        self._hall_frame_available = True
        self._last_skipped = skipped
        self._update_hall_voltages()
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
                "note": "picard_sweeps=2" if self.ahe_enabled else "amr",
            },
        }


class _FakeRawPoissonCudaSolver:
    last_from_arrays = None
    last_transport = None

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
        amr_enabled=False,
        amr_ratio=0.0,
        ahe_enabled=False,
        ahe_ratio=0.0,
        picard_sweeps=2,
        picard_tolerance=0.0,
    ):
        impl = _FakeImpl(contact_potentials)
        impl.amr_enabled = bool(amr_enabled)
        impl.ahe_enabled = bool(ahe_enabled)
        impl.amr_ratio = float(amr_ratio)
        impl.ahe_ratio = float(ahe_ratio)
        impl.picard_sweeps = int(picard_sweeps)
        impl.transport_enabled = bool(amr_enabled or ahe_enabled)
        _FakeRawPoissonCudaSolver.last_transport = {
            "amr_enabled": impl.amr_enabled,
            "ahe_enabled": impl.ahe_enabled,
            "amr_ratio": impl.amr_ratio,
            "ahe_ratio": impl.ahe_ratio,
            "picard_sweeps": impl.picard_sweeps,
        }
        return impl

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
        amr_enabled=False,
        amr_ratio=0.0,
        ahe_enabled=False,
        ahe_ratio=0.0,
        picard_sweeps=2,
        picard_tolerance=0.0,
    ):
        _FakeRawPoissonCudaSolver.last_from_arrays = {
            "shape": (nz, ny, nx),
            "cellsize": (cx, cy, cz),
            "region_dtype": region.dtype,
            "contact_id_dtype": contact_id.dtype,
            "sigma_dtype": sigma.dtype,
        }
        n_fm = max(1, int(nz) - int(first_r2_layer))
        impl = _FakeImpl(
            contact_potentials,
            buffer_shape=(n_fm, ny, nx, 3),
            first_r2_layer=first_r2_layer,
        )
        impl.amr_enabled = bool(amr_enabled)
        impl.ahe_enabled = bool(ahe_enabled)
        impl.amr_ratio = float(amr_ratio)
        impl.ahe_ratio = float(ahe_ratio)
        impl.picard_sweeps = int(picard_sweeps)
        impl.transport_enabled = bool(amr_enabled or ahe_enabled)
        _FakeRawPoissonCudaSolver.last_transport = {
            "amr_enabled": impl.amr_enabled,
            "ahe_enabled": impl.ahe_enabled,
            "amr_ratio": impl.amr_ratio,
            "ahe_ratio": impl.ahe_ratio,
            "picard_sweeps": impl.picard_sweeps,
        }
        return impl

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


def test_load_contact_potentials_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "contacts.txt"
    path.write_text("1 2 3\n4 5 6\n", encoding="ascii")
    with pytest.raises(ValueError, match="5 column"):
        poisson.load_contact_potentials(str(path), num_contacts=5)


def test_build_fgat_world_spec_default_one_contact():
    spec = poisson.build_fgat_world_spec()
    assert spec.shape == (10, 512, 512)
    assert poisson.num_contacts_from_world_spec(spec) == 1
    ids = set(int(v) for v in np.unique(spec.contact_id) if int(v) != 0)
    assert ids == {-1, 1}
    assert np.any(spec.contact_id[:, :, 0] == 1)
    assert np.any(spec.contact_id[:, :, -1] == -1)


def test_build_fgat_world_spec_dense_symmetric_contacts():
    spec = poisson.build_fgat_world_spec(num_contacts=4, contact_layout="auto")
    assert poisson.num_contacts_from_world_spec(spec) == 4
    ids = set(int(v) for v in np.unique(spec.contact_id) if int(v) != 0)
    assert ids == {-4, -3, -2, -1, 1, 2, 3, 4}
    for contact in range(1, 5):
        pos = spec.contact_id[:, :, 0] == contact
        neg = spec.contact_id[:, :, -1] == -contact
        assert int(np.count_nonzero(pos)) == int(np.count_nonzero(neg))
        assert np.count_nonzero(pos) > 0


def test_contact_layout_minimums_warn_and_error():
    with pytest.warns(RuntimeWarning):
        layout = poisson.resolve_contact_layout(
            num_contacts=2,
            ny=20,
            cy=5e-9,
            nx=20,
            cx=5e-9,
            contact_layout="manual",
            contact_size_cells=4,
            contact_spacing_cells=4,
            contact_edge_depth_cells=10,
        )
    assert layout.contact_size_cells == 4

    with pytest.raises(ValueError, match="minimum"):
        poisson.resolve_contact_layout(
            num_contacts=1,
            ny=20,
            cy=10e-9,
            nx=20,
            cx=10e-9,
            contact_layout="manual",
            contact_size_cells=2,
            contact_edge_depth_cells=10,
        )


def test_manual_contact_layout_must_fit():
    with pytest.raises(ValueError, match="requires"):
        poisson.resolve_contact_layout(
            num_contacts=3,
            ny=20,
            cy=10e-9,
            nx=20,
            cx=10e-9,
            contact_layout="manual",
            contact_size_cells=10,
            contact_spacing_cells=10,
            contact_edge_depth_cells=10,
        )


def test_signal_file_to_contact_potentials_generalizes_split(tmp_path):
    path = tmp_path / "signal.txt"
    path.write_text("\n".join(str(v) for v in range(30)), encoding="ascii")

    values3 = poisson.load_signal_file_to_contact_potentials(
        str(path),
        5,
        v_scale=1.0,
        skip_first=0,
        num_contacts=3,
    )
    assert values3.shape == (5, 3)

    values5 = poisson.load_signal_file_to_contact_potentials(
        str(path),
        4,
        v_scale=1.0,
        skip_first=0,
        num_contacts=5,
    )
    assert values5.shape == (4, 5)
    assert np.all(np.isfinite(values5))


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


def test_transport_defaults_off(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(contact_potentials=np.zeros((1, 3)))
    assert not solver.transport_enabled
    assert not solver.amr_enabled
    assert not solver.ahe_enabled
    assert fake_raw_solver.last_transport["amr_enabled"] is False
    assert fake_raw_solver.last_transport["ahe_enabled"] is False


def test_amr_ratio_requires_flag(fake_raw_solver):
    with pytest.raises(ValueError, match="amr_ratio requires amr_enabled"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            amr_ratio=0.1,
        )


def test_ahe_ratio_requires_flag(fake_raw_solver):
    with pytest.raises(ValueError, match="ahe_ratio requires ahe_enabled"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            ahe_ratio=0.05,
        )


def test_picard_sweeps_must_be_positive(fake_raw_solver):
    with pytest.raises(ValueError, match="picard_sweeps"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            ahe_enabled=True,
            ahe_ratio=0.05,
            picard_sweeps=0,
        )


def test_transport_iterate_requires_magnetization_for_active_frame(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        amr_enabled=True,
        amr_ratio=0.1,
    )
    with pytest.raises(RuntimeError, match="magnetization is required"):
        solver.iterate()


def test_transport_skip_without_magnetization(fake_raw_solver):
    potentials = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        amr_enabled=True,
        amr_ratio=0.1,
    )
    frame = solver.iterate()
    assert frame.stats.skipped


def test_transport_iterate_with_magnetization(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        amr_enabled=True,
        amr_ratio=0.1,
        ahe_enabled=True,
        ahe_ratio=0.05,
    )
    m = np.zeros((3, 2, 4, 5), dtype=np.float32)
    m[2, ...] = 1.0
    frame = solver.iterate(magnetization=m)
    assert not frame.stats.skipped
    assert frame.jcur.shape == (3, 2, 4, 5)
    assert solver._impl.last_magnetization is not None
    assert solver._impl.last_magnetization.shape == (3, 2, 4, 5)


def test_magnetization_3ny_nx_expands(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        fm_nz="0",
        fm_mumax_nz=1,
        amr_enabled=True,
        amr_ratio=0.05,
    )
    m = np.zeros((3, 4, 5), dtype=np.float32)
    m[2, ...] = 1.0
    frame = solver.iterate(magnetization=m)
    assert frame.jmod.shape == (3, 1, 4, 5)
    assert solver._impl.last_magnetization.shape[0] == 3
    assert solver._impl.last_magnetization.shape[1] == 2  # full FM stack
    # Uniform-z broadcast: every Poisson FM layer gets the same m.
    np.testing.assert_allclose(
        solver._impl.last_magnetization[:, 0, ...],
        solver._impl.last_magnetization[:, 1, ...],
    )


def test_magnetization_averages_and_broadcasts_when_mumax_thinner(fake_raw_solver):
    """mumax nz < Poisson FM layers => z-average then uniform broadcast."""

    shape = (5, 4, 5)  # first_r2=1 => 4 Poisson FM layers
    region = np.ones(shape, dtype=np.int8)
    region[0] = 1
    region[1:] = 2
    contact_id = np.zeros(shape, dtype=np.int8)
    contact_id[0, :, 0] = 1
    contact_id[0, :, -1] = -1
    sigma = np.ones(shape, dtype=np.float32)
    spec = poisson.WorldSpec(
        shape=shape,
        cellsize=(1e-9, 1e-9, 5e-9),
        first_r2_layer=1,
        theta_sh=0.2,
        decay_length=8e-9,
        region=region,
        contact_id=contact_id,
        sigma=sigma,
    )
    potentials = np.array([[1e-3]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        world=spec,
        contact_potentials=potentials,
        fm_height=10e-9,
        fm_mumax_nz=2,
        amr_enabled=True,
        amr_ratio=0.05,
    )
    assert solver.fm_layer_count == 4
    assert solver.output_shape[1] == 2

    m = np.zeros((3, 2, 4, 5), dtype=np.float32)
    m[2, 0, ...] = 1.0
    m[2, 1, ...] = -1.0
    # After average: mz ~ 0; add in-plane so result is nontrivial after renormalize.
    m[0, 0, ...] = 1.0
    m[0, 1, ...] = 1.0
    frame = solver.iterate(magnetization=m)
    assert frame.jcur.shape == (3, 2, 4, 5)
    mag = solver._impl.last_magnetization
    assert mag.shape == (3, 4, 4, 5)
    # All Poisson FM layers identical (uniform z).
    for iz in range(1, 4):
        np.testing.assert_allclose(mag[:, 0, ...], mag[:, iz, ...])
    # Averaged mx dominates; mz cancels.
    assert float(np.mean(np.abs(mag[2]))) < 1e-5
    assert float(np.mean(mag[0])) > 0.9


def test_magnetization_shape_mismatch(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        amr_enabled=True,
        amr_ratio=0.05,
    )
    with pytest.raises(ValueError, match="does not match Poisson export shape"):
        solver.iterate(magnetization=np.ones((3, 1, 4, 5), dtype=np.float32))


def test_magnetization_nonfinite_rejected(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(
        contact_potentials=potentials,
        amr_enabled=True,
        amr_ratio=0.05,
    )
    m = np.zeros((3, 2, 4, 5), dtype=np.float32)
    m[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        solver.iterate(magnetization=m)


def test_scalar_path_ignores_magnetization(fake_raw_solver):
    potentials = np.array([[1e-3, 0.0, 0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(contact_potentials=potentials)
    m = np.zeros((3, 2, 4, 5), dtype=np.float32)
    m[2, ...] = 1.0
    frame = solver.iterate(magnetization=m)
    assert not frame.stats.skipped


def test_void_sigma_builds_conducting_fillers():
    spec = poisson.build_fgat_world_spec(
        shape=(4, 32, 32),
        void_sigma=1.0,
        void_radius=20e-9,
    )
    assert np.any(spec.region[2:] == 0)
    void_cells = spec.region[2:] == 0
    assert np.all(spec.sigma[2:][void_cells] == 1.0)
    fm_cells = spec.region[2:] == 2
    assert np.all(spec.sigma[2:][fm_cells] > 0.0)


def test_resolve_hall_contact_geometry_mirrors_drive_contacts():
    spec = poisson.build_fgat_world_spec(
        num_contacts=2,
        shape=(4, 64, 64),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=10,
        contact_spacing_cells=10,
        contact_edge_depth_cells=10,
        void_locations=None,
    )
    geom = poisson.resolve_hall_contact_geometry(spec)
    assert geom.num_contacts == 2
    assert geom.sign_convention == "high_y_minus_low_y"
    assert len(geom.high_y_indices) == 2
    assert len(geom.low_y_indices) == 2

    nz, ny, nx = spec.shape
    contact_id = np.asarray(spec.contact_id)
    # Infer drive y widths and x depth.
    for c in range(1, 3):
        mask = np.abs(contact_id) == c
        ys = np.where(mask)[1]
        xs = np.where(mask)[2]
        drive_width = int(ys.max()) - int(ys.min()) + 1
        depth_left = int(xs[xs < nx // 2].max()) + 1
        x0, x1 = geom.x_ranges[c - 1]
        assert x1 - x0 == drive_width
        low0, low1 = geom.low_y_ranges[c - 1]
        high0, high1 = geom.high_y_ranges[c - 1]
        assert low1 - low0 == depth_left
        assert high1 - high0 == depth_left
        assert low0 == 0
        assert high1 == ny

    # Disjoint, non-overlapping with Dirichlet contacts, non-empty.
    for c in range(2):
        high = set(int(v) for v in geom.high_y_indices[c])
        low = set(int(v) for v in geom.low_y_indices[c])
        assert high and low
        assert not (high & low)
        for idx in high | low:
            iz = idx // (ny * nx)
            rem = idx % (ny * nx)
            iy = rem // nx
            ix = rem % nx
            assert contact_id[iz, iy, ix] == 0


def test_hall_potentials_helper_with_fake_impl(fake_raw_solver):
    spec = poisson.build_fgat_world_spec(
        num_contacts=2,
        shape=(4, 64, 64),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=10,
        contact_spacing_cells=10,
        contact_edge_depth_cells=10,
        void_locations=None,
    )
    potentials = np.array([[1e-3, 2e-3]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(world=spec, contact_potentials=potentials)

    with pytest.raises(RuntimeError, match="iterate"):
        # Configure geometry first so the native error is about missing iterate.
        solver._ensure_hall_geometry_configured()
        solver.hall_potentials()

    frame = solver.iterate()
    assert not frame.stats.skipped
    v = solver.hall_potentials()
    assert v.shape == (2,)
    assert v.dtype == np.float64
    np.testing.assert_allclose(v, [1e-3, 2e-3])
    assert solver._impl.hall_configure_count == 1

    # Second call reuses cached geometry/indices.
    v2 = solver.hall_potentials()
    np.testing.assert_allclose(v2, v)
    assert solver._impl.hall_configure_count == 1

    comps = solver.hall_potentials(return_components=True)
    assert isinstance(comps, poisson.HallPotentialResult)
    np.testing.assert_allclose(comps.voltages, v)
    assert comps.geometry.num_contacts == 2


def test_hall_potentials_skipped_frame_returns_zeros(fake_raw_solver):
    spec = poisson.build_fgat_world_spec(
        num_contacts=1,
        shape=(4, 64, 64),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=20,
        contact_edge_depth_cells=10,
        void_locations=None,
    )
    potentials = np.array([[1e-3], [0.0]], dtype=np.float64)
    solver = poisson.CudaPoissonSolver(world=spec, contact_potentials=potentials)
    first = solver.iterate()
    assert not first.stats.skipped
    v1 = solver.hall_potentials()
    np.testing.assert_allclose(v1, [1e-3])

    second = solver.iterate()
    assert second.stats.skipped
    v2 = solver.hall_potentials()
    np.testing.assert_allclose(v2, [0.0])


def test_hall_geometry_from_masks_rejects_empty():
    spec = poisson.build_fgat_world_spec(
        num_contacts=1,
        shape=(4, 32, 32),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=10,
        contact_edge_depth_cells=10,
        void_locations=None,
    )
    mask = np.zeros(spec.shape, dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        poisson.hall_geometry_from_masks(spec, mask, mask)


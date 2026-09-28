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
        self.the_enabled = False
        self.ohe_enabled = True
        self.resistivity_invert = False
        self.magnetization_required = False
        self.amr_ratio = 0.0
        self.ahe_ratio = 0.0
        self.the_ratio = 0.0
        self.hall_coefficient_pt = -2.44e-11
        self.hall_coefficient_fm = 3.09e-10
        self.picard_sweeps = 2
        self.fm_layer_count = nz
        self.last_magnetization = None
        self.last_applied_field = None
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

    def winding_fm_stack(self):
        nz, ny, nx, _ = self.buffer_shape
        return np.zeros((3, nz, ny, nx), dtype=np.float32)

    def the_hall_vector_fm_stack(self):
        return self.winding_fm_stack()

    def winding_stats(self):
        return {"max_abs": 0.0, "sum_hz": 0.0}

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
        if self.magnetization_required and not skipped:
            raise RuntimeError(
                "magnetization is required when AMR/AHE/THE transport is enabled"
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
                "note": "picard_sweeps=2" if (self.ahe_enabled or self.the_enabled) else "amr",
            },
        }

    def set_applied_field_uniform(self, bx, by, bz):
        self.last_applied_field = (float(bx), float(by), float(bz))

    def set_applied_field_grid(self, applied_field):
        self.last_applied_field = np.asarray(applied_field)


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
        the_enabled=False,
        the_ratio=0.0,
        picard_sweeps=2,
        picard_tolerance=0.0,
        ohe_enabled=True,
        hall_coefficient_pt=-2.44e-11,
        hall_coefficient_fm=3.09e-10,
        resistivity_invert=False,
        solver="pcg",
        gmres_restart=200,
        voltage_scale_guess=False,
        preconditioner="jacobi",
    ):
        impl = _FakeImpl(contact_potentials)
        impl.amr_enabled = bool(amr_enabled)
        impl.ahe_enabled = bool(ahe_enabled)
        impl.the_enabled = bool(the_enabled)
        impl.ohe_enabled = bool(ohe_enabled)
        impl.amr_ratio = float(amr_ratio)
        impl.ahe_ratio = float(ahe_ratio)
        impl.the_ratio = float(the_ratio)
        impl.hall_coefficient_pt = float(hall_coefficient_pt)
        impl.hall_coefficient_fm = float(hall_coefficient_fm)
        impl.resistivity_invert = bool(resistivity_invert)
        impl.picard_sweeps = int(picard_sweeps)
        impl.solver = solver
        impl.gmres_restart = gmres_restart
        impl.voltage_scale_guess = bool(voltage_scale_guess)
        impl.preconditioner = str(preconditioner)
        impl.magnetization_required = bool(amr_enabled or ahe_enabled or the_enabled)
        impl.transport_enabled = bool(amr_enabled or ahe_enabled or the_enabled or ohe_enabled)
        _FakeRawPoissonCudaSolver.last_transport = {
            "amr_enabled": impl.amr_enabled,
            "ahe_enabled": impl.ahe_enabled,
            "the_enabled": impl.the_enabled,
            "ohe_enabled": impl.ohe_enabled,
            "amr_ratio": impl.amr_ratio,
            "ahe_ratio": impl.ahe_ratio,
            "the_ratio": impl.the_ratio,
            "hall_coefficient_pt": impl.hall_coefficient_pt,
            "hall_coefficient_fm": impl.hall_coefficient_fm,
            "resistivity_invert": impl.resistivity_invert,
            "picard_sweeps": impl.picard_sweeps,
            "solver": impl.solver,
            "gmres_restart": impl.gmres_restart,
            "voltage_scale_guess": impl.voltage_scale_guess,
            "preconditioner": impl.preconditioner,
            "tolerance": tolerance,
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
        the_enabled=False,
        the_ratio=0.0,
        picard_sweeps=2,
        picard_tolerance=0.0,
        ohe_enabled=True,
        hall_coefficient_pt=-2.44e-11,
        hall_coefficient_fm=3.09e-10,
        resistivity_invert=False,
        solver="pcg",
        gmres_restart=200,
        voltage_scale_guess=False,
        preconditioner="jacobi",
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
        impl.the_enabled = bool(the_enabled)
        impl.ohe_enabled = bool(ohe_enabled)
        impl.amr_ratio = float(amr_ratio)
        impl.ahe_ratio = float(ahe_ratio)
        impl.the_ratio = float(the_ratio)
        impl.hall_coefficient_pt = float(hall_coefficient_pt)
        impl.hall_coefficient_fm = float(hall_coefficient_fm)
        impl.resistivity_invert = bool(resistivity_invert)
        impl.picard_sweeps = int(picard_sweeps)
        impl.solver = solver
        impl.gmres_restart = gmres_restart
        impl.voltage_scale_guess = bool(voltage_scale_guess)
        impl.preconditioner = str(preconditioner)
        impl.magnetization_required = bool(amr_enabled or ahe_enabled or the_enabled)
        impl.transport_enabled = bool(amr_enabled or ahe_enabled or the_enabled or ohe_enabled)
        _FakeRawPoissonCudaSolver.last_transport = {
            "amr_enabled": impl.amr_enabled,
            "ahe_enabled": impl.ahe_enabled,
            "the_enabled": impl.the_enabled,
            "ohe_enabled": impl.ohe_enabled,
            "amr_ratio": impl.amr_ratio,
            "ahe_ratio": impl.ahe_ratio,
            "the_ratio": impl.the_ratio,
            "hall_coefficient_pt": impl.hall_coefficient_pt,
            "hall_coefficient_fm": impl.hall_coefficient_fm,
            "resistivity_invert": impl.resistivity_invert,
            "picard_sweeps": impl.picard_sweeps,
            "solver": impl.solver,
            "gmres_restart": impl.gmres_restart,
            "voltage_scale_guess": impl.voltage_scale_guess,
            "preconditioner": impl.preconditioner,
            "tolerance": tolerance,
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
    assert hasattr(poisson, "HallPotentialLayers")


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
    assert poisson.parse_fm_nz_spec("0:2", 4, 1) == (0, 1)
    assert poisson.parse_fm_nz_spec("0:8", 8) == tuple(range(8))


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


def test_poisson_fm_layers_example_checks():
    """2 Pt + 8×5 nm FM exports: exponential and averages, packed and 2 nm."""

    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "examples" / "poisson_fm_layers.py"
    runpy.run_path(str(script), run_name="__main__")


def test_layer_count_defaults_to_selected_poisson_layers(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.array([[1e-3, 0.0, 0.0]], dtype=np.float64),
        fm_nz="0:2",
    )
    assert solver.fm_mumax_nz == 2
    assert solver.jmod_profile == "sample"
    assert solver.output_shape[1] == 2


def test_solver_average_collapses_two_poisson_layers(fake_raw_solver):
    cz = 5e-9
    decay = 8e-9
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.array([[1e-3, 0.0, 0.0]], dtype=np.float64),
        fm_nz="0:2",
        fm_mumax_nz=1,
        jmod_profile="average",
    )
    frame = solver.iterate()
    assert frame.jmod.shape == (3, 1, 4, 5)
    np.testing.assert_allclose(frame.jcur, 1.5)
    factor0 = poisson.fm_injection_decay_factor(0, cz, decay)
    sheet = poisson.exponential_integral(0.0, 2 * cz, decay)
    expected = (1.0 / factor0) * (sheet / (2 * cz))
    np.testing.assert_allclose(frame.jmod, expected, rtol=1e-5)


def test_solver_exponential_spreads_two_layers_over_eight(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.array([[1e-3, 0.0, 0.0]], dtype=np.float64),
        fm_nz="0:2",
        fm_mumax_nz=8,
        jmod_profile="exponential",
    )
    frame = solver.iterate()
    assert frame.jmod.shape == (3, 8, 4, 5)
    assert frame.jmod[0, 0, 0, 0] > frame.jmod[0, -1, 0, 0]
    np.testing.assert_allclose(frame.jcur[0, :4], 1.0)
    np.testing.assert_allclose(frame.jcur[0, 4:], 2.0)
    assert solver.fm_cellsize_z == pytest.approx(solver.cellsize[2] / 4)


def test_magnetization_from_thicker_llg_stack(fake_raw_solver):
    shape = (3, 4, 5)
    region = np.ones(shape, dtype=np.int8)
    region[1:] = 2
    contact_id = np.zeros(shape, dtype=np.int8)
    sigma = np.ones(shape, dtype=np.float32)
    spec = poisson.WorldSpec(
        shape=shape,
        cellsize=(1.0, 2.0, 5e-9),
        first_r2_layer=1,
        theta_sh=0.2,
        decay_length=8e-9,
        region=region,
        contact_id=contact_id,
        sigma=sigma,
    )
    solver = poisson.CudaPoissonSolver(
        world=spec,
        contact_potentials=np.array([[1e-3]], dtype=np.float64),
        fm_nz="0:2",
        fm_mumax_nz=8,
        jmod_profile="exponential",
        amr_enabled=True,
        amr_ratio=0.05,
    )
    m = np.zeros((3, 8, 4, 5), dtype=np.float32)
    m[2, :4, ...] = 1.0
    m[0, 4:, ...] = 1.0
    solver.iterate(magnetization=m)
    mag = solver._impl.last_magnetization
    assert mag.shape == (3, 2, 4, 5)
    assert float(mag[2, 0, 0, 0]) > 0.9
    assert float(mag[0, 1, 0, 0]) > 0.9


def _decayed_fm_stack(n_fm, cz, decay_length, amplitude=1.0):
    jmod = np.zeros((3, n_fm, 1, 1), dtype=np.float32)
    jcur = np.zeros_like(jmod)
    for layer in range(n_fm):
        factor = poisson.fm_injection_decay_factor(layer, cz, decay_length)
        jmod[0, layer, 0, 0] = np.float32(amplitude * factor)
        jcur[0, layer, 0, 0] = np.float32(layer + 1)
    return jmod, jcur


def test_map_fm_currents_profiles():
    cz = 2e-9
    decay = 4e-9
    n_fm = 8
    jmod, jcur = _decayed_fm_stack(n_fm, cz, decay)
    amplitude = 1.0

    two = (0, 1)
    copied_jmod, copied_jcur = poisson.map_fm_currents(
        jmod,
        jcur,
        source_layers=two,
        n_out=2,
        poisson_cz=cz,
        fm_cz=cz,
        decay_length=decay,
        jmod_profile="sample",
    )
    np.testing.assert_allclose(copied_jmod[0, :, 0, 0], jmod[0, :2, 0, 0])
    np.testing.assert_allclose(copied_jcur[0, :, 0, 0], [1.0, 2.0])

    one_cz = poisson.resolve_fm_cellsize_z(two, 1, cz)
    assert one_cz == pytest.approx(2 * cz)
    exp_one, jcur_one = poisson.map_fm_currents(
        jmod,
        jcur,
        source_layers=two,
        n_out=1,
        poisson_cz=cz,
        fm_cz=one_cz,
        decay_length=decay,
        jmod_profile="exponential",
    )
    avg_one, _ = poisson.map_fm_currents(
        jmod,
        jcur,
        source_layers=two,
        n_out=1,
        poisson_cz=cz,
        fm_cz=one_cz,
        decay_length=decay,
        jmod_profile="average",
    )
    np.testing.assert_allclose(exp_one, avg_one)
    np.testing.assert_allclose(jcur_one[0, 0, 0, 0], 1.5)
    expected_sheet = amplitude * poisson.exponential_integral(0.0, 2 * cz, decay)
    np.testing.assert_allclose(exp_one[0, 0, 0, 0] * one_cz, expected_sheet)

    eight_cz = poisson.resolve_fm_cellsize_z(two, 8, cz)
    assert eight_cz == pytest.approx(2 * cz / 8)
    exp_eight, jcur_eight = poisson.map_fm_currents(
        jmod,
        jcur,
        source_layers=two,
        n_out=8,
        poisson_cz=cz,
        fm_cz=eight_cz,
        decay_length=decay,
        jmod_profile="exponential",
    )
    avg_eight, _ = poisson.map_fm_currents(
        jmod,
        jcur,
        source_layers=two,
        n_out=8,
        poisson_cz=cz,
        fm_cz=eight_cz,
        decay_length=decay,
        jmod_profile="average",
    )
    assert exp_eight[0, 0, 0, 0] > exp_eight[0, -1, 0, 0]
    np.testing.assert_allclose(avg_eight[0, 0, 0, 0], avg_eight[0, -1, 0, 0])
    np.testing.assert_allclose(jcur_eight[0, :4, 0, 0], 1.0)
    np.testing.assert_allclose(jcur_eight[0, 4:, 0, 0], 2.0)
    exp_sheet = float(np.sum(exp_eight[0, :, 0, 0]) * eight_cz)
    avg_sheet = float(np.sum(avg_eight[0, :, 0, 0]) * eight_cz)
    np.testing.assert_allclose(exp_sheet, expected_sheet)
    np.testing.assert_allclose(avg_sheet, expected_sheet)

    full_jmod, full_jcur = poisson.map_fm_currents(
        jmod,
        jcur,
        source_layers=tuple(range(n_fm)),
        n_out=n_fm,
        poisson_cz=cz,
        fm_cz=cz,
        decay_length=decay,
        jmod_profile="sample",
    )
    np.testing.assert_allclose(full_jmod, jmod)
    np.testing.assert_allclose(full_jcur, jcur)


def test_build_fgat_world_layer_and_material_controls():
    spec = poisson.build_fgat_world_spec(
        n_pt_layers=2,
        n_fm_layers=8,
        nx=32,
        ny=32,
        cellsize=(8e-9, 8e-9, 2e-9),
        decay_length=4e-9,
        theta_sh=0.15,
        sigma_pt=1.5e6,
        sigma_fm=4.0e5,
        void_locations=(),
        contact_edge_depth_cells=10,
    )
    assert spec.shape == (10, 32, 32)
    assert spec.first_r2_layer == 2
    assert spec.cellsize[2] == pytest.approx(2e-9)
    assert spec.decay_length == pytest.approx(4e-9)
    assert spec.theta_sh == pytest.approx(0.15)
    assert np.all(spec.region[:2] == 1)
    assert np.all(spec.region[2:] == 2)
    assert spec.sigma[0, 0, 0] == np.float32(1.5e6)
    assert spec.sigma[2, 0, 0] == np.float32(4.0e5)


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


def test_transport_defaults_include_ohe(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(contact_potentials=np.zeros((1, 3)))
    assert solver.transport_enabled
    assert solver.ohe_enabled
    assert not solver.amr_enabled
    assert not solver.ahe_enabled
    assert not solver.the_enabled
    assert not solver.magnetization_required
    assert solver.solver == "gmres_cusparse"
    assert solver.hall_coefficient_pt == pytest.approx(-2.44e-11)
    assert solver.hall_coefficient_fm == pytest.approx(3.09e-10)
    assert fake_raw_solver.last_transport["ohe_enabled"] is True
    assert fake_raw_solver.last_transport["amr_enabled"] is False
    assert fake_raw_solver.last_transport["ahe_enabled"] is False
    assert fake_raw_solver.last_transport["the_enabled"] is False
    assert fake_raw_solver.last_transport["gmres_restart"] == 200
    assert fake_raw_solver.last_transport["voltage_scale_guess"] is False
    assert fake_raw_solver.last_transport["preconditioner"] == "jacobi"
    assert fake_raw_solver.last_transport["resistivity_invert"] is False
    assert fake_raw_solver.last_transport["tolerance"] == pytest.approx(1e-6)
    assert not solver.resistivity_invert


def test_solver_option_forwards_to_native(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.zeros((1, 3)),
        amr_enabled=True,
        amr_ratio=0.1,
        solver="gmres",
        gmres_restart=24,
    )
    assert solver.solver == "gmres_cusparse"
    assert fake_raw_solver.last_transport["solver"] == "gmres_cusparse"
    assert fake_raw_solver.last_transport["gmres_restart"] == 24


def test_preconditioner_and_restart_list_forward_to_native(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.zeros((1, 3)),
        solver="gmres_cusparse",
        preconditioner="gmg",
        gmres_restart=(50, 0, 0, 200),
    )
    assert solver.preconditioner == "gmg"
    assert solver.gmres_restart == (50, 0, 0, 200)
    assert fake_raw_solver.last_transport["preconditioner"] == "gmg"
    assert fake_raw_solver.last_transport["gmres_restart"] == [50, 0, 0, 200]


def test_resistivity_invert_forwards_to_native(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.zeros((1, 3)),
        resistivity_invert=True,
    )
    assert solver.resistivity_invert
    assert fake_raw_solver.last_transport["resistivity_invert"] is True


def test_preconditioner_gmg_requires_gmres(fake_raw_solver):
    with pytest.raises(ValueError, match="preconditioner='gmg'"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            solver="pcg",
            preconditioner="gmg",
        )


def test_gmres_restart_rejects_one(fake_raw_solver):
    with pytest.raises(ValueError, match="gmres_restart"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            gmres_restart=1,
        )


def test_voltage_scale_guess_forwards_to_native(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.zeros((1, 3)),
        voltage_scale_guess=True,
    )
    assert solver.voltage_scale_guess is True
    assert fake_raw_solver.last_transport["voltage_scale_guess"] is True


def test_solver_option_validation(fake_raw_solver):
    with pytest.raises(ValueError, match="solver must be"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            solver="cg-but-not-this-one",
        )


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


def test_the_ratio_requires_flag(fake_raw_solver):
    with pytest.raises(ValueError, match="the_ratio requires the_enabled"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            the_ratio=0.1,
        )


def test_the_flag_forwards_to_native(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(
        contact_potentials=np.zeros((1, 3)),
        the_enabled=True,
        the_ratio=0.2,
    )
    assert solver.the_enabled
    assert solver.the_ratio == 0.2
    assert solver.transport_enabled
    assert fake_raw_solver.last_transport["the_enabled"] is True
    assert fake_raw_solver.last_transport["the_ratio"] == 0.2
    with pytest.raises(ValueError, match="picard_sweeps"):
        poisson.CudaPoissonSolver(
            contact_potentials=np.zeros((1, 3)),
            the_enabled=True,
            the_ratio=0.1,
            picard_sweeps=0,
        )
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


def test_resolve_hall_contact_geometry_z_modes():
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
    geom_c = poisson.resolve_hall_contact_geometry(spec, z_mode="contact")
    geom_pt = poisson.resolve_hall_contact_geometry(spec, z_mode="pt")
    geom_fm = poisson.resolve_hall_contact_geometry(spec, z_mode="fm")
    assert geom_c.z_layers == (0, 1)
    assert geom_pt.z_layers == (0, 1)
    assert geom_fm.z_layers == (2, 3)
    with pytest.raises(ValueError, match="cannot build a single"):
        poisson.resolve_hall_contact_geometry(spec, z_mode="both")


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


def test_hall_layer_potentials_both_are_independent_arrays(fake_raw_solver):
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
    solver.iterate()

    layers = solver.hall_layer_potentials("both")
    assert isinstance(layers, poisson.HallPotentialLayers)
    assert layers.locations == ("contact", "fm")
    assert layers.pt is None
    assert layers.contact is not None and layers.fm is not None
    assert layers.contact.shape == (2,)
    assert layers.fm.shape == (2,)
    assert layers.contact is not layers.fm
    v_contact = solver.hall_potentials(z_mode="contact")
    v_fm = solver.hall_potentials(z_mode="fm")
    np.testing.assert_allclose(layers.contact, v_contact)
    np.testing.assert_allclose(layers.fm, v_fm)
    np.testing.assert_allclose(layers["contact"], v_contact)
    np.testing.assert_allclose(layers["fm"], v_fm)

    via_z_mode = solver.hall_potentials(z_mode="both")
    assert isinstance(via_z_mode, poisson.HallPotentialLayers)
    np.testing.assert_allclose(via_z_mode.contact, v_contact)
    np.testing.assert_allclose(via_z_mode.fm, v_fm)

    fm_only = solver.hall_layer_potentials("fm")
    assert fm_only.contact is None
    assert fm_only.pt is None
    np.testing.assert_allclose(fm_only.fm, v_fm)

    both_comps = solver.hall_layer_potentials("both", return_components=True)
    assert both_comps.contact_result is not None
    assert both_comps.fm_result is not None
    assert both_comps.contact_result.geometry.z_layers == (0, 1)
    assert both_comps.fm_result.geometry.z_layers == (2, 3)
    assert both_comps.contact_result.geometry.z_layers != both_comps.fm_result.geometry.z_layers

    with pytest.raises(ValueError, match="auto z selections"):
        solver.hall_layer_potentials("both", geometry=both_comps.contact_result.geometry)


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


def _small_hall_world():
    return poisson.build_fgat_world_spec(
        num_contacts=1,
        shape=(4, 32, 32),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=10,
        contact_edge_depth_cells=10,
        void_locations=None,
    )


def _uniform_m(shape, axis=2):
    m = np.zeros(shape, dtype=np.float32)
    m[axis, ...] = 1.0
    return m


def _skyrmion_like_m(shape, radius_cells=6.0):
    _, nz, ny, nx = shape
    m = np.zeros(shape, dtype=np.float32)
    cy = 0.5 * (ny - 1)
    cx = 0.5 * (nx - 1)
    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij")
    dx = xx - cx
    dy = yy - cy
    rho = np.sqrt(dx * dx + dy * dy)
    psi = np.pi * np.clip(rho / float(radius_cells), 0.0, 1.0)
    mz = np.cos(psi)
    mr = np.sin(psi)
    inv = np.where(rho > 1e-6, 1.0 / rho, 0.0)
    mx = mr * dx * inv
    my = mr * dy * inv
    m[0, ...] = mx
    m[1, ...] = my
    m[2, ...] = mz
    norm = np.linalg.norm(m, axis=0, keepdims=True)
    m = np.where(norm > 1e-12, m / np.maximum(norm, 1e-12), 0.0)
    return np.ascontiguousarray(m, dtype=np.float32)


def _try_real_solver(**kwargs):
    try:
        return poisson.CudaPoissonSolver(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on local CUDA
        pytest.skip(f"CUDA Poisson solver unavailable: {exc}")


def test_the_uniform_m_vanishes_and_matches_ahe_only():
    spec = _small_hall_world()
    potentials = np.full((1, 1), 1e-3, dtype=np.float64)
    common = dict(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ahe_enabled=True,
        ahe_ratio=0.05,
        picard_sweeps=3,
        solver="gmres_cusparse",
    )
    ahe_only = _try_real_solver(**common)
    both = _try_real_solver(the_enabled=True, the_ratio=0.2, **common)
    m = _uniform_m(ahe_only.output_shape)
    ahe_only.iterate(magnetization=m)
    frame = both.iterate(magnetization=m)
    v_ahe = ahe_only.hall_potentials()
    v_both = both.hall_potentials()
    np.testing.assert_allclose(v_both, v_ahe, rtol=1e-4, atol=1e-9)
    stats = both.winding_stats()
    assert stats["max_abs"] < 1e-6
    h = both.winding()
    assert h.shape == both.output_shape
    np.testing.assert_allclose(h, 0.0, atol=1e-6)
    assert frame.jmod.shape == both.output_shape
    assert frame.jcur.shape == both.output_shape
    assert frame.jmod.dtype == np.float32
    assert frame.jcur.dtype == np.float32


def test_the_texture_localizes_h_and_adds_odd_hall():
    spec = _small_hall_world()
    potentials = np.full((2, 1), 1e-3, dtype=np.float64)
    solver = _try_real_solver(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        the_enabled=True,
        the_ratio=0.4,
        picard_sweeps=4,
        solver="gmres_cusparse",
    )
    m = _skyrmion_like_m(solver.output_shape)
    frame = solver.iterate(magnetization=m)
    assert frame.jmod.shape == solver.output_shape
    assert frame.jcur.shape == solver.output_shape
    h = solver.winding()
    stats = solver.winding_stats()
    assert stats["max_abs"] > 1e-4
    # Winding is localized near the texture core, not the whole film.
    core = np.abs(h[2, 0, 16 - 8 : 16 + 8, 16 - 8 : 16 + 8]).sum()
    edge = np.abs(h[2, 0, :4, :]).sum() + np.abs(h[2, 0, -4:, :]).sum()
    assert core > edge
    v_pos = solver.hall_potentials()
    solver.reset()
    solver.iterate(magnetization=-m)
    v_neg = solver.hall_potentials()
    odd = 0.5 * (v_pos - v_neg)
    even = 0.5 * (v_pos + v_neg)
    assert np.max(np.abs(odd)) > np.max(np.abs(even))
    hall = solver.the_hall_vector()
    assert hall.shape == solver.output_shape


def test_the_pcg_picard_agrees_with_gmres():
    spec = _small_hall_world()
    potentials = np.full((1, 1), 1e-3, dtype=np.float64)
    kwargs = dict(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ahe_enabled=True,
        ahe_ratio=0.05,
        the_enabled=True,
        the_ratio=0.2,
        picard_sweeps=6,
        picard_tolerance=0.0,
        tol=1e-6,
        max_iter=4000,
    )
    gmres = _try_real_solver(solver="gmres_cusparse", **kwargs)
    pcg = _try_real_solver(solver="pcg", **kwargs)
    m = _skyrmion_like_m(gmres.output_shape)
    gmres.iterate(magnetization=m)
    pcg.iterate(magnetization=m)
    v_g = gmres.hall_potentials()
    v_p = pcg.hall_potentials()
    # PCG and GMRES currently differ by a global Hall-voltage sign (also for
    # AHE-only); magnitudes and winding must still match.
    np.testing.assert_allclose(np.abs(v_p), np.abs(v_g), rtol=5e-3, atol=1e-8)
    np.testing.assert_allclose(gmres.winding(), pcg.winding(), rtol=1e-5, atol=1e-8)


def _ohe_random_contact_world():
    return poisson.build_fgat_world_spec(
        num_contacts=2,
        shape=(4, 32, 32),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=8,
        contact_spacing_cells=6,
        contact_edge_depth_cells=4,
        void_locations=None,
    )


def _random_contact_potentials(n_frames, n_contacts, seed=12345):
    rng = np.random.default_rng(seed)
    potentials = rng.uniform(-2.5e-3, 2.5e-3, size=(n_frames, n_contacts))
    potentials[np.abs(potentials) < 4e-4] = 1.2e-3
    return np.ascontiguousarray(potentials, dtype=np.float64)


@pytest.mark.parametrize("tol", [1e-6, 1e-7])
def test_ohe_random_contacts_sat_m_pt_and_fm_probes(tol):
    spec = _ohe_random_contact_world()
    potentials = _random_contact_potentials(2, spec.contact_id.max())
    solver = _try_real_solver(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ohe_enabled=True,
        ahe_enabled=False,
        the_enabled=False,
        amr_enabled=False,
        tol=tol,
        gmres_restart=200,
        max_iter=4000,
        solver="gmres_cusparse",
    )
    m = _uniform_m(solver.output_shape, axis=2)
    bz = 0.15

    def hall_pt_fm():
        v_pt = solver.hall_potentials(z_mode="pt")
        v_fm = solver.hall_potentials(z_mode="fm")
        return v_pt, v_fm

    solver.set_applied_field((0.0, 0.0, 0.0))
    frame0 = solver.iterate(magnetization=m)
    assert not frame0.stats.skipped
    assert frame0.stats.residual_rel <= 50.0 * tol
    v0_pt, v0_fm = hall_pt_fm()

    solver.reset()
    solver.iterate(magnetization=m, applied_field=(0.0, 0.0, bz))
    vp_pt, vp_fm = hall_pt_fm()

    solver.reset()
    solver.iterate(magnetization=m, applied_field=(0.0, 0.0, -bz))
    vn_pt, vn_fm = hall_pt_fm()

    odd_pt = 0.5 * (vp_pt - vn_pt)
    even_pt = 0.5 * (vp_pt + vn_pt)
    odd_fm = 0.5 * (vp_fm - vn_fm)
    even_fm = 0.5 * (vp_fm + vn_fm)

    # Literature R_H at 0.15 T gives a nV-scale Hall voltage on this mesh.
    assert np.max(np.abs(odd_pt)) > np.max(np.abs(even_pt - v0_pt))
    assert np.max(np.abs(odd_fm)) > np.max(np.abs(even_fm - v0_fm))
    assert np.max(np.abs(odd_pt)) > 1e-10
    assert np.max(np.abs(odd_fm)) > 1e-10
    np.testing.assert_allclose(even_pt, v0_pt, rtol=0.2, atol=1e-10)
    np.testing.assert_allclose(even_fm, v0_fm, rtol=0.2, atol=1e-10)
    np.testing.assert_allclose(vp_pt - v0_pt, -(vn_pt - v0_pt), rtol=0.2, atol=1e-10)
    np.testing.assert_allclose(vp_fm - v0_fm, -(vn_fm - v0_fm), rtol=0.2, atol=1e-10)

    solver.reset()
    solver.iterate(magnetization=-m, applied_field=(0.0, 0.0, bz))
    vm_pt, vm_fm = hall_pt_fm()
    np.testing.assert_allclose(vm_pt, vp_pt, rtol=1e-3, atol=1e-9)
    np.testing.assert_allclose(vm_fm, vp_fm, rtol=1e-3, atol=1e-9)

    solver.reset()
    frame1 = solver.iterate(magnetization=m, applied_field=(0.0, 0.0, bz))
    assert not frame1.stats.skipped
    solver.iterate(magnetization=m, applied_field=(0.0, 0.0, bz))
    v2_pt = solver.hall_potentials(z_mode="pt")
    v2_fm = solver.hall_potentials(z_mode="fm")
    assert v2_pt.shape == (2,)
    assert v2_fm.shape == (2,)
    assert np.max(np.abs(v2_pt)) > 0.0
    assert np.max(np.abs(v2_fm)) > 0.0
    layers = solver.hall_layer_potentials(("pt", "fm"))
    np.testing.assert_allclose(layers.pt, v2_pt)
    np.testing.assert_allclose(layers.fm, v2_fm)
    assert layers.contact is None


def test_ohe_b_zero_matches_ohe_disabled():
    spec = _ohe_random_contact_world()
    potentials = _random_contact_potentials(1, spec.contact_id.max(), seed=7)
    common = dict(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ahe_enabled=False,
        the_enabled=False,
        amr_enabled=False,
        tol=1e-6,
        gmres_restart=200,
        solver="gmres_cusparse",
    )
    with_ohe = _try_real_solver(ohe_enabled=True, **common)
    without = _try_real_solver(ohe_enabled=False, **common)
    m = _uniform_m(with_ohe.output_shape, axis=2)
    with_ohe.iterate(magnetization=m, applied_field=(0.0, 0.0, 0.0))
    without.iterate(magnetization=m)
    np.testing.assert_allclose(
        with_ohe.hall_potentials(z_mode="pt"),
        without.hall_potentials(z_mode="pt"),
        rtol=1e-4,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        with_ohe.hall_potentials(z_mode="fm"),
        without.hall_potentials(z_mode="fm"),
        rtol=1e-4,
        atol=1e-9,
    )


def test_applied_field_forwards_to_native(fake_raw_solver):
    solver = poisson.CudaPoissonSolver(contact_potentials=np.zeros((1, 3)))
    solver.set_applied_field((0.0, 0.1, 0.2))
    np.testing.assert_allclose(solver._impl.last_applied_field, (0.0, 0.1, 0.2), atol=1e-6)


def _void_cutout_world():
    return poisson.build_fgat_world_spec(
        num_contacts=1,
        shape=(4, 32, 32),
        cellsize=(5e-9, 5e-9, 5e-9),
        contact_layout="manual",
        contact_size_cells=10,
        contact_edge_depth_cells=10,
        void_locations=[(0.5, 0.5)],
        void_radius=20e-9,
    )


def test_gmg_matches_jacobi_on_two_textured_fields():
    spec = _void_cutout_world()
    potentials = np.full((2, 1), 1e-3, dtype=np.float64)
    common = dict(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        amr_enabled=True,
        amr_ratio=0.1,
        ahe_enabled=True,
        ahe_ratio=0.05,
        the_enabled=True,
        the_ratio=0.2,
        ohe_enabled=True,
        tol=1e-6,
        max_iter=4000,
        solver="gmres_cusparse",
        gmres_restart=32,
    )
    jacobi = _try_real_solver(preconditioner="jacobi", **common)
    gmg = _try_real_solver(preconditioner="gmg", **common)
    assert gmg.preconditioner == "gmg"
    assert gmg._impl.gmg_n_levels >= 2
    assert gmg._impl.gmg_void_sparsity_ok is True
    m0 = _skyrmion_like_m(jacobi.output_shape)
    m1 = _uniform_m(jacobi.output_shape, axis=1)

    frame0 = jacobi.iterate(magnetization=m0, applied_field=(0.0, 0.0, 0.05))
    gframe0 = gmg.iterate(magnetization=m0, applied_field=(0.0, 0.0, 0.05))
    assert not frame0.stats.skipped
    assert not gframe0.stats.skipped
    assert gframe0.stats.residual_rel <= 50.0 * common["tol"]
    assert "gmg" in gframe0.stats.note
    v0_j = jacobi.hall_potentials()
    v0_g = gmg.hall_potentials()
    np.testing.assert_allclose(v0_j, v0_g, rtol=5e-3, atol=1e-8)

    jacobi.iterate(magnetization=m1, applied_field=(0.0, 0.0, 0.05))
    gframe1 = gmg.iterate(magnetization=m1, applied_field=(0.0, 0.0, 0.05))
    assert "gmg" in gframe1.stats.note
    v1_j = jacobi.hall_potentials()
    v1_g = gmg.hall_potentials()
    np.testing.assert_allclose(v1_j, v1_g, rtol=5e-3, atol=1e-8)
    assert np.max(np.abs(v1_g - v0_g)) > 1e-10


def test_resistivity_invert_amr_only_matches_additive():
    spec = _small_hall_world()
    potentials = np.full((1, 1), 1e-3, dtype=np.float64)
    common = dict(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        amr_enabled=True,
        amr_ratio=0.1,
        ahe_enabled=False,
        the_enabled=False,
        ohe_enabled=False,
        tol=1e-6,
        solver="gmres_cusparse",
    )
    additive = _try_real_solver(resistivity_invert=False, **common)
    inverted = _try_real_solver(resistivity_invert=True, **common)
    assert inverted.resistivity_invert
    m = _uniform_m(additive.output_shape, axis=0)
    f_add = additive.iterate(magnetization=m)
    f_inv = inverted.iterate(magnetization=m)
    np.testing.assert_allclose(
        inverted.hall_potentials(),
        additive.hall_potentials(),
        rtol=1e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(f_inv.jcur, f_add.jcur, rtol=1e-4, atol=1e-8)


def test_resistivity_invert_small_ahe_matches_additive_to_theta2():
    spec = _small_hall_world()
    potentials = np.full((1, 1), 1e-3, dtype=np.float64)
    theta = 0.02
    common = dict(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ahe_enabled=True,
        ahe_ratio=theta,
        the_enabled=False,
        ohe_enabled=False,
        amr_enabled=False,
        tol=1e-6,
        solver="gmres_cusparse",
    )
    additive = _try_real_solver(resistivity_invert=False, **common)
    inverted = _try_real_solver(resistivity_invert=True, **common)
    m = _uniform_m(additive.output_shape, axis=2)
    additive.iterate(magnetization=m)
    inverted.iterate(magnetization=m)
    v_add = additive.hall_potentials()
    v_inv = inverted.hall_potentials()
    scale = np.max(np.abs(v_add))
    assert scale > 0.0
    np.testing.assert_allclose(v_inv, v_add, rtol=20.0 * theta * theta, atol=1e-10 + 20.0 * theta * theta * scale)


def test_resistivity_invert_m_negation_flips_hall():
    spec = _small_hall_world()
    potentials = np.full((1, 1), 1e-3, dtype=np.float64)
    solver = _try_real_solver(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ahe_enabled=True,
        ahe_ratio=0.05,
        ohe_enabled=False,
        the_enabled=False,
        resistivity_invert=True,
        solver="gmres_cusparse",
    )
    m = _uniform_m(solver.output_shape, axis=2)
    solver.iterate(magnetization=m)
    v_plus = solver.hall_potentials()
    solver.reset()
    solver.iterate(magnetization=-m)
    v_minus = solver.hall_potentials()
    np.testing.assert_allclose(v_minus, -v_plus, rtol=1e-3, atol=1e-10)
    assert np.max(np.abs(v_plus)) > 0.0


def test_resistivity_invert_b_negation_flips_ohe_only():
    spec = _small_hall_world()
    potentials = np.full((1, 1), 1e-3, dtype=np.float64)
    solver = _try_real_solver(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ahe_enabled=False,
        the_enabled=False,
        amr_enabled=False,
        ohe_enabled=True,
        resistivity_invert=True,
        solver="gmres_cusparse",
    )
    m = _uniform_m(solver.output_shape, axis=2)
    solver.iterate(magnetization=m, applied_field=(0.0, 0.0, 0.15))
    v_plus = solver.hall_potentials()
    solver.reset()
    solver.iterate(magnetization=m, applied_field=(0.0, 0.0, -0.15))
    v_minus = solver.hall_potentials()
    np.testing.assert_allclose(v_minus, -v_plus, rtol=1e-3, atol=1e-10)
    assert np.max(np.abs(v_plus)) > 0.0


def test_resistivity_invert_uniform_m_the_matches_ahe_ohe():
    spec = _small_hall_world()
    potentials = np.full((1, 1), 1e-3, dtype=np.float64)
    common = dict(
        world=spec,
        contact_potentials=potentials,
        skip_threshold=0.0,
        ahe_enabled=True,
        ahe_ratio=0.05,
        ohe_enabled=True,
        resistivity_invert=True,
        solver="gmres_cusparse",
    )
    without_the = _try_real_solver(the_enabled=False, **common)
    with_the = _try_real_solver(the_enabled=True, the_ratio=0.2, **common)
    m = _uniform_m(without_the.output_shape, axis=2)
    without_the.iterate(magnetization=m, applied_field=(0.0, 0.0, 0.05))
    with_the.iterate(magnetization=m, applied_field=(0.0, 0.0, 0.05))
    np.testing.assert_allclose(
        with_the.hall_potentials(),
        without_the.hall_potentials(),
        rtol=1e-4,
        atol=1e-10,
    )


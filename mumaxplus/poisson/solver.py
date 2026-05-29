"""CUDA-backed persistent Poisson solver.

This module wraps a lightweight embedded CUDA Poisson solver as a persistent
object that can be called between mumax+ time-integration steps. It is separate
from :class:`mumaxplus.PoissonSystem` and uses its own world geometry.

Vector fields (``jmod``, ``jcur``) use the same layout as mumax+
:class:`~mumaxplus.FieldQuantity` arrays: ``(3, nz, ny, nx)`` with component
index first. Scalar world arrays in :class:`WorldSpec` remain ``(nz, ny, nx)``.
Grid extents follow :class:`~mumaxplus.Grid.shape`, i.e. ``(nz, ny, nx)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from mumaxplus import _cpp


@dataclass(frozen=True)
class WorldSpec:
    """In-memory Poisson world definition.

    Parameters
    ----------
    shape : tuple of int
        Grid shape as ``(nz, ny, nx)``.
    cellsize : tuple of float
        Cell size as ``(cx, cy, cz)``.
    first_r2_layer : int
        First FM layer in the Poisson world.
    theta_sh : float
        Spin Hall angle metadata. It is not applied by the Poisson solver.
    decay_length : float
        Decay length used by the existing ``jmod`` postprocess.
    region, contact_id, sigma : array_like
        Arrays shaped ``(nz, ny, nx)``. ``region`` and ``contact_id`` are
        converted to ``int8``; ``sigma`` is converted to ``float32``.
    """

    shape: Tuple[int, int, int]
    cellsize: Tuple[float, float, float]
    first_r2_layer: int
    theta_sh: float
    decay_length: float
    region: np.ndarray
    contact_id: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class PoissonStepStats:
    """Solver statistics for one ``CudaPoissonSolver.iterate`` call."""

    step: int
    skipped: bool
    iterations: int
    residual_initial: float
    residual: float
    rhs_inf: float
    residual_rel: float
    elapsed_s: float
    note: str = ""


@dataclass(frozen=True)
class PoissonStepResult:
    """Current-density frames returned by one Poisson iteration.

    ``jmod`` and ``jcur`` are shaped ``(3, nz_export, ny, nx)`` (mumax+ vector layout)
    where ``nz_export`` is the number of FM layers selected at solver construction
    (see :class:`CudaPoissonSolver` ``fm_export_layers``). FM layer index ``0`` is
    the layer directly adjacent to the Pt stack (Poisson ``first_r2_layer``).
    """

    jmod: np.ndarray
    jcur: np.ndarray
    stats: PoissonStepStats

    def iter_fm_layers(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(jmod[:, z], jcur[:, z])`` per FM layer, each shaped ``(3, ny, nx)``."""

        for z in range(self.jmod.shape[1]):
            yield self.jmod[:, z], self.jcur[:, z]


def default_world_path() -> str:
    """Return the packaged default Poisson world manifest path."""

    return str(Path(__file__).resolve().parent / "defaults" / "FGaT-amr-sine-poisson-world.txt")


def load_contact_potentials(path: str) -> np.ndarray:
    """Load three-column contact potentials from a text file.

    Parameters
    ----------
    path : str
        Text file containing one ``V0 V1 V2`` row per timestep.

    Returns
    -------
    numpy.ndarray
        Float64 array with shape ``(nt, 3)``.
    """

    values = np.loadtxt(path, dtype=np.float64, comments="#")
    if values.ndim == 1:
        if values.shape[0] != 3:
            raise ValueError("contact-potential text file must have three columns")
        values = values.reshape(1, 3)
    return _normalize_contact_potentials(values)


def _normalize_contact_potentials(contact_potentials: Any) -> np.ndarray:
    values = np.asarray(contact_potentials, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("contact_potentials must have shape (nt, 3)")
    if values.shape[0] == 0:
        raise ValueError("contact_potentials must contain at least one timestep")
    if not np.all(np.isfinite(values)):
        raise ValueError("contact_potentials contains NaN or Inf")
    return np.ascontiguousarray(values)


def _shape_tuple(shape_like: Any) -> Tuple[int, int, int]:
    """Spatial grid shape ``(nz, ny, nx)`` from a grid or a vector field."""

    return _spatial_shape(shape_like)


def _spatial_shape(shape_like: Any) -> Tuple[int, int, int]:
    """Return ``(nz, ny, nx)`` from ``Grid.shape`` or a ``(3, nz, ny, nx)`` field."""

    shape = getattr(shape_like, "shape", shape_like)
    try:
        out = tuple(int(v) for v in shape)
    except TypeError as exc:
        raise ValueError(
            "shape must be (nz, ny, nx) or a mumax+ object with a .shape attribute"
        ) from exc
    if len(out) == 4:
        if out[0] != 3:
            raise ValueError(
                f"vector field shape must be (3, nz, ny, nx), got leading dim {out[0]}"
            )
        return out[1:]
    if len(out) == 3:
        return out
    raise ValueError(f"expected spatial shape (nz, ny, nx) or (3, nz, ny, nx), got {out}")


def _as_mumax_vector_frame(frame: np.ndarray) -> np.ndarray:
    """Ensure a current frame is mumax+ layout ``(3, nz, ny, nx)``."""

    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"expected a 4D current frame, got shape {arr.shape}")
    if arr.shape[0] == 3:
        return np.ascontiguousarray(arr)
    if arr.shape[-1] == 3:
        return np.ascontiguousarray(np.transpose(arr, (3, 0, 1, 2)))
    raise ValueError(
        f"expected (3, nz, ny, nx) or internal (nz, ny, nx, 3), got shape {arr.shape}"
    )


FmExportLayers = Union[int, Sequence[int], None]


def _parse_first_r2_from_manifest(manifest_path: str) -> int:
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("first_r2_layer="):
            return int(stripped.split("=", 1)[1])
    raise ValueError(f"manifest missing first_r2_layer: {manifest_path}")


def _first_r2_for_init(
    world: Optional[WorldSpec],
    manifest_path: Optional[str],
) -> int:
    if world is not None:
        return int(world.first_r2_layer)
    return _parse_first_r2_from_manifest(manifest_path or default_world_path())


def _normalize_fm_export_layers(
    fm_export_layers: FmExportLayers,
    n_fm_layers: int,
) -> Tuple[int, ...]:
    if n_fm_layers <= 0:
        raise ValueError("Poisson world has no FM layers to export")

    if fm_export_layers is None:
        return tuple(range(n_fm_layers))

    if isinstance(fm_export_layers, int):
        layer = int(fm_export_layers)
        if layer < 0 or layer >= n_fm_layers:
            raise ValueError(
                f"fm_export_layers={layer} out of range [0, {n_fm_layers}) "
                "(0 = FM layer at Pt interface)"
            )
        return (layer,)

    layers = tuple(int(v) for v in fm_export_layers)
    if len(layers) == 0:
        raise ValueError("fm_export_layers must not be empty")
    for layer in layers:
        if layer < 0 or layer >= n_fm_layers:
            raise ValueError(
                f"fm_export_layers contains {layer}, valid range is [0, {n_fm_layers})"
            )
    return layers


def _fm_slice_z_string(first_r2_layer: int) -> str:
    return f"{int(first_r2_layer)}:"


def parse_fm_export_layers(spec: Optional[str]) -> FmExportLayers:
    """Parse ``--fm-export-layers`` style text into :data:`FmExportLayers`.

    Examples
    --------
    ``None`` or ``""`` → export all FM layers.
    ``"0"`` → only the FM layer at the Pt interface.
    ``"0,3,7"`` → three selected layers in order.
    """

    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    parts = [int(p.strip()) for p in text.split(",") if p.strip()]
    if len(parts) == 0:
        raise ValueError("fm_export_layers must list at least one layer index")
    return parts[0] if len(parts) == 1 else tuple(parts)


def _stats_from_dict(stats: Dict[str, Any]) -> PoissonStepStats:
    return PoissonStepStats(
        step=int(stats["step"]),
        skipped=bool(stats["skipped"]),
        iterations=int(stats["iterations"]),
        residual_initial=float(stats["residual_initial"]),
        residual=float(stats["residual"]),
        rhs_inf=float(stats["rhs_inf"]),
        residual_rel=float(stats["residual_rel"]),
        elapsed_s=float(stats["elapsed_s"]),
        note=str(stats.get("note", "")),
    )


class CudaPoissonSolver:
    """Persistent CUDA Poisson solver returning one current frame per call.

    The solver loads the full contact-potential series at construction, keeps
    the CUDA PCG warm start alive across ``iterate`` calls, and returns
    ``float32`` ``jmod`` and ``jcur`` as ``(3, nz, ny, nx)`` for the FM layer(s)
    you choose to export (same layout as mumax+ vector fields).

    All FM-layer postprocessing (raw ``jcur``, Pt injection into ``jmod``, decay)
    runs on the full Poisson FM stack internally. ``fm_export_layers`` only
    selects which FM layers are returned to the mumax+ drive grid.

    This class is intentionally separate from ``mumaxplus.PoissonSystem``. Its
    world geometry can come from the packaged default manifest, an explicit
    manifest path, or a :class:`WorldSpec`.
    """

    def __init__(
        self,
        *,
        world: Optional[WorldSpec] = None,
        manifest_path: Optional[str] = None,
        contact_potentials: Any,
        tol: float = 1e-5,
        max_iter: int = 2000,
        skip_threshold: float = 1e-5,
        fm_export_layers: FmExportLayers = None,
        jmod_slice_x: str = "",
        jmod_slice_y: str = "",
        jmod_slice_z: Optional[str] = None,
        cuda_tol_batch_first: int = 1000,
        cuda_tol_batch_next: int = 500,
    ) -> None:
        potentials = _normalize_contact_potentials(contact_potentials)

        if world is not None and manifest_path is not None:
            raise ValueError("provide either world or manifest_path, not both")

        first_r2 = _first_r2_for_init(world, manifest_path)
        if jmod_slice_z is not None and fm_export_layers is not None:
            raise ValueError("provide either fm_export_layers or jmod_slice_z, not both")
        internal_slice_z = (
            jmod_slice_z if jmod_slice_z is not None else _fm_slice_z_string(first_r2)
        )

        if world is None:
            self._impl = _cpp.PoissonCudaSolver.from_manifest(
                manifest_path or default_world_path(),
                potentials,
                tol,
                max_iter,
                skip_threshold,
                jmod_slice_x,
                jmod_slice_y,
                internal_slice_z,
                cuda_tol_batch_first,
                cuda_tol_batch_next,
            )
        else:
            nz, ny, nx = _shape_tuple(world.shape)
            cx, cy, cz = tuple(float(v) for v in world.cellsize)
            region = np.ascontiguousarray(world.region, dtype=np.int8)
            contact_id = np.ascontiguousarray(world.contact_id, dtype=np.int8)
            sigma = np.ascontiguousarray(world.sigma, dtype=np.float32)
            self._impl = _cpp.PoissonCudaSolver.from_arrays(
                nx,
                ny,
                nz,
                cx,
                cy,
                cz,
                int(world.first_r2_layer),
                float(world.theta_sh),
                float(world.decay_length),
                region,
                contact_id,
                sigma,
                potentials,
                tol,
                max_iter,
                skip_threshold,
                jmod_slice_x,
                jmod_slice_y,
                internal_slice_z,
                cuda_tol_batch_first,
                cuda_tol_batch_next,
            )

        n_fm = int(self._impl.buffer_shape[0])
        self._fm_export_layers = _normalize_fm_export_layers(fm_export_layers, n_fm)
        self._first_r2_layer = int(self._impl.first_r2_layer)

    @classmethod
    def from_signal_file(
        cls,
        signal_path: str,
        *,
        nt: int,
        manifest_path: Optional[str] = None,
        v_scale: float = 0.005,
        skip_first: int = 1000,
        tol: float = 1e-5,
        max_iter: int = 2000,
        skip_threshold: float = 1e-5,
        fm_export_layers: FmExportLayers = None,
        jmod_slice_x: str = "",
        jmod_slice_y: str = "",
        jmod_slice_z: Optional[str] = None,
        cuda_tol_batch_first: int = 1000,
        cuda_tol_batch_next: int = 500,
    ) -> "CudaPoissonSolver":
        """Construct from a single-column signal file using C++ resampling rules."""

        manifest = manifest_path or default_world_path()
        if jmod_slice_z is not None and fm_export_layers is not None:
            raise ValueError("provide either fm_export_layers or jmod_slice_z, not both")
        internal_slice_z = (
            jmod_slice_z
            if jmod_slice_z is not None
            else _fm_slice_z_string(_parse_first_r2_from_manifest(manifest))
        )

        obj = cls.__new__(cls)
        obj._impl = _cpp.PoissonCudaSolver.from_signal_file(
            manifest,
            signal_path,
            int(nt),
            float(v_scale),
            int(skip_first),
            float(tol),
            int(max_iter),
            float(skip_threshold),
            jmod_slice_x,
            jmod_slice_y,
            internal_slice_z,
            int(cuda_tol_batch_first),
            int(cuda_tol_batch_next),
        )
        n_fm = int(obj._impl.buffer_shape[0])
        obj._fm_export_layers = _normalize_fm_export_layers(fm_export_layers, n_fm)
        obj._first_r2_layer = int(obj._impl.first_r2_layer)
        return obj

    @property
    def current_step(self) -> int:
        """Index of the next contact-potential frame to solve."""

        return int(self._impl.current_step)

    @property
    def n_steps(self) -> int:
        """Number of loaded contact-potential frames."""

        return int(self._impl.n_steps)

    @property
    def exhausted(self) -> bool:
        """Whether all contact-potential frames have been consumed."""

        return bool(self._impl.exhausted)

    @property
    def fm_export_layers(self) -> Tuple[int, ...]:
        """FM layer indices exported to mumax+ (0 = layer at Pt interface)."""

        return self._fm_export_layers

    @property
    def fm_layer_count(self) -> int:
        """Number of FM layers in the Poisson world (before export selection)."""

        return int(self._impl.buffer_shape[0])

    @property
    def first_fm_layer(self) -> int:
        """World z-index of the first FM layer (``first_r2_layer``)."""

        return self._first_r2_layer

    @property
    def internal_output_shape(self) -> Tuple[int, int, int, int]:
        """FM-stack buffer shape inside the C++ session: ``(nz, ny, nx, 3)``."""

        return tuple(int(v) for v in self._impl.buffer_shape)

    @property
    def output_shape(self) -> Tuple[int, int, int, int]:
        """Returned ``jmod``/``jcur`` shape as ``(3, nz_export, ny, nx)``."""

        _, ny, nx, _ = self.internal_output_shape
        return (3, len(self._fm_export_layers), ny, nx)

    @property
    def world_shape(self) -> Tuple[int, int, int]:
        """Full Poisson world shape as ``(nz, ny, nx)``."""

        return tuple(int(v) for v in self._impl.world_shape)

    @property
    def cellsize(self) -> Tuple[float, float, float]:
        """Poisson world cell size as ``(cx, cy, cz)``."""

        return tuple(float(v) for v in self._impl.cellsize)

    @property
    def theta_sh(self) -> float:
        """Spin Hall angle metadata from the Poisson world."""

        return float(self._impl.theta_sh)

    @property
    def decay_length(self) -> float:
        """Decay length used for ``jmod`` postprocessing."""

        return float(self._impl.decay_length)

    @property
    def unknown_count(self) -> int:
        """Number of active non-contact Poisson unknowns."""

        return int(self._impl.unknown_count)

    def reset(self) -> None:
        """Restart from contact frame zero and clear the CUDA warm start."""

        self._impl.reset()

    def _select_fm_export(self, frame: np.ndarray) -> np.ndarray:
        arr = _as_mumax_vector_frame(frame)
        if len(self._fm_export_layers) == self.fm_layer_count and self._fm_export_layers == tuple(
            range(self.fm_layer_count)
        ):
            return arr
        return arr[:, list(self._fm_export_layers), ...]

    def iterate(self) -> PoissonStepResult:
        """Solve the next contact-potential frame.

        Returns
        -------
        PoissonStepResult
            ``jmod`` and ``jcur`` are independent ``float32`` NumPy arrays shaped for
            the selected FM export layers. Use :meth:`PoissonStepResult.iter_fm_layers`
            to assign currents layer by layer. The next call to ``iterate`` will not
            mutate previously returned arrays.
        """

        raw = self._impl.iterate()
        return PoissonStepResult(
            jmod=self._select_fm_export(raw["jmod"]),
            jcur=self._select_fm_export(raw["jcur"]),
            stats=_stats_from_dict(raw["stats"]),
        )

    def check_compatible(
        self,
        grid_shape: Any,
        cellsize: Optional[Tuple[float, float, float]] = None,
        *,
        rtol: float = 1e-6,
    ) -> Dict[str, Any]:
        """Check exported current frames against a mumax+ ferromagnet grid.

        Compares the mumax spatial grid ``(nz, ny, nx)`` — or a vector field
        ``(3, nz, ny, nx)`` — to the exported FM layers. Does not compare against
        the full Poisson world shape. Raises if the mumax grid requests more FM
        layers than Poisson provides, or if ``ny`` / ``nx`` differ.

        This method only inspects metadata and does not advance the Poisson
        contact-potential series.
        """

        expected = self.output_shape[1:]
        actual = _spatial_shape(grid_shape)
        if actual[0] > self.fm_layer_count:
            raise ValueError(
                f"mumax grid requests nz={actual[0]} FM layers but Poisson only has "
                f"{self.fm_layer_count} FM layers "
                f"(world z indices {self.first_fm_layer}:{self.world_shape[0]})"
            )
        if actual != expected:
            raise ValueError(
                f"Poisson export shape {expected} is incompatible with mumax grid shape "
                f"{actual} (fm_export_layers={self.fm_export_layers})"
            )

        if cellsize is not None:
            cellsize_actual = tuple(float(v) for v in cellsize)
            if len(cellsize_actual) != 3:
                raise ValueError("cellsize must have length 3")
            if not np.allclose(cellsize_actual, self.cellsize, rtol=rtol, atol=0.0):
                raise ValueError(
                    f"Poisson cellsize {self.cellsize} is incompatible with {cellsize_actual}"
                )

        return {
            "shape": self.output_shape,
            "cellsize": self.cellsize,
            "n_steps": self.n_steps,
            "world_shape": self.world_shape,
            "fm_export_layers": self.fm_export_layers,
            "fm_layer_count": self.fm_layer_count,
            "first_fm_layer": self.first_fm_layer,
        }

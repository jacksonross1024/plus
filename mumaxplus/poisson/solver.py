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

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union
import time
import warnings

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
class ContactLayout:
    """Resolved y-axis contact layout for symmetric left/right contact pairs."""

    num_contacts: int
    contact_size: float
    contact_spacing: float
    contact_size_cells: int
    contact_spacing_cells: int
    contact_edge_depth: float
    contact_edge_depth_cells: int
    y_ranges: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class HallContactGeometry:
    """Resolved virtual Hall-bar probe indices for transverse voltage readout.

    Sign convention is ``V_hall = mean(phi_high_y) - mean(phi_low_y)`` per
    applied-potential contact channel. Indices use C-order flat indexing
    ``(iz * ny + iy) * nx + ix``.
    """

    num_contacts: int
    high_y_indices: Tuple[np.ndarray, ...]
    low_y_indices: Tuple[np.ndarray, ...]
    x_ranges: Tuple[Tuple[int, int], ...]
    low_y_ranges: Tuple[Tuple[int, int], ...]
    high_y_ranges: Tuple[Tuple[int, int], ...]
    z_layers: Tuple[int, ...]
    sign_convention: str = "high_y_minus_low_y"
    source: str = "auto"


@dataclass(frozen=True)
class HallPotentialResult:
    """Diagnostic Hall readout with per-probe means and cell counts."""

    voltages: np.ndarray
    high_y_means: np.ndarray
    low_y_means: np.ndarray
    high_y_counts: Tuple[int, ...]
    low_y_counts: Tuple[int, ...]
    geometry: HallContactGeometry


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
    pcg_error: float = 0.0
    pcg_converged: bool = False
    picard_error: float = 0.0
    picard_sweeps_used: int = 0
    timing_total_s: float = 0.0
    timing_device_magnetization_s: float = 0.0
    timing_transport_s: float = 0.0
    timing_magnetization_set_s: float = 0.0
    timing_transport_rebuild_s: float = 0.0
    timing_operator_upload_s: float = 0.0
    timing_rhs_build_s: float = 0.0
    timing_linear_solve_s: float = 0.0
    timing_fill_phi_s: float = 0.0
    timing_hall_s: float = 0.0
    timing_j_raw_s: float = 0.0
    timing_jcur_extract_s: float = 0.0
    timing_jmod_postprocess_s: float = 0.0
    timing_jmod_extract_s: float = 0.0
    timing_numpy_jmod_s: float = 0.0
    timing_numpy_jcur_s: float = 0.0
    timing_python_magnetization_s: float = 0.0
    timing_native_call_s: float = 0.0
    timing_python_output_map_s: float = 0.0


@dataclass(frozen=True)
class PoissonStepResult:
    """Current-density frames returned by one Poisson iteration.

    ``jmod`` and ``jcur`` are shaped ``(3, nz_export, ny, nx)`` (mumax+ vector
    layout), after applying the layer/height export configured on
    :class:`CudaPoissonSolver`. FM layer index ``0`` is at the Pt interface.
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


DEFAULT_FGAT_SHAPE: Tuple[int, int, int] = (10, 512, 512)
DEFAULT_FGAT_CELLSIZE: Tuple[float, float, float] = (
    1.9531249950688334e-09,
    1.9531249950688334e-09,
    4.999999969612645e-09,
)
DEFAULT_FGAT_FIRST_R2_LAYER = 2
DEFAULT_FGAT_THETA_SH = 0.2
DEFAULT_FGAT_DECAY_LENGTH = 8e-9
DEFAULT_FGAT_SIGMA_PT = 2_000_000.0
DEFAULT_FGAT_SIGMA_FM = 393_000.0
DEFAULT_FGAT_VOID_RADIUS = 30e-9
DEFAULT_FGAT_VOID_LOCATIONS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (0.25, 0.25),
    (0.75, 0.25),
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5),
    (0.25, 0.75),
    (0.75, 0.75),
    (0.0, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
)
MAX_SUPPORTED_CONTACTS = 63
MIN_CONTACT_LENGTH = 20e-9
WARN_CONTACT_CELLS = 10
MIN_CONTACT_CELLS = 3


def _coerce_positive_int(value: Any, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be > 0")
    return out


def _cells_to_length(cells: int, cellsize: float) -> float:
    return float(cells) * float(cellsize)


def _length_to_cells(length: float, cellsize: float, name: str) -> int:
    if float(length) <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return max(1, int(np.ceil(float(length) / float(cellsize))))


def _validate_resolved_length(
    *,
    name: str,
    cells: int,
    cellsize: float,
    require_physical_minimum: bool = True,
) -> None:
    length = _cells_to_length(cells, cellsize)
    if require_physical_minimum and length < MIN_CONTACT_LENGTH:
        raise ValueError(
            f"{name}={length:.3e} m is smaller than the {MIN_CONTACT_LENGTH:.3e} m minimum"
        )
    if cells < MIN_CONTACT_CELLS:
        raise ValueError(f"{name} resolves to {cells} cell(s); minimum is {MIN_CONTACT_CELLS}")
    if cells < WARN_CONTACT_CELLS:
        warnings.warn(
            f"{name} resolves to {cells} cells (< {WARN_CONTACT_CELLS}); "
            "consider using a finer grid or larger contact layout",
            RuntimeWarning,
            stacklevel=3,
        )


def _manual_length_cells(
    *,
    name: str,
    length: Optional[float],
    cells: Optional[int],
    cellsize: float,
) -> int:
    if length is not None and cells is not None:
        raise ValueError(f"provide either {name} or {name}_cells, not both")
    if cells is not None:
        out = _coerce_positive_int(cells, f"{name}_cells")
    elif length is not None:
        out = _length_to_cells(float(length), cellsize, name)
    else:
        raise ValueError(f"{name} must be specified in manual contact layout mode")
    _validate_resolved_length(name=name, cells=out, cellsize=cellsize)
    return out


def _edge_depth_cells(
    *,
    nx: int,
    cx: float,
    contact_edge_depth: Optional[float],
    contact_edge_depth_cells: Optional[int],
) -> int:
    if contact_edge_depth is not None and contact_edge_depth_cells is not None:
        raise ValueError("provide either contact_edge_depth or contact_edge_depth_cells, not both")
    if contact_edge_depth_cells is not None:
        out = _coerce_positive_int(contact_edge_depth_cells, "contact_edge_depth_cells")
    elif contact_edge_depth is not None:
        out = _length_to_cells(float(contact_edge_depth), cx, "contact_edge_depth")
    else:
        # Preserve the old edge-strip convention where possible, but respect the
        # contact-size minimums for newly generated worlds.
        out = max(5, _length_to_cells(MIN_CONTACT_LENGTH, cx, "contact_edge_depth"))
    _validate_resolved_length(name="contact_edge_depth", cells=out, cellsize=cx)
    if 2 * out > nx:
        raise ValueError(
            f"contact_edge_depth_cells={out} on both x edges overlaps for nx={nx}"
        )
    return out


def resolve_contact_layout(
    *,
    num_contacts: int = 1,
    ny: int = DEFAULT_FGAT_SHAPE[1],
    cy: float = DEFAULT_FGAT_CELLSIZE[1],
    nx: int = DEFAULT_FGAT_SHAPE[2],
    cx: float = DEFAULT_FGAT_CELLSIZE[0],
    contact_layout: str = "auto",
    contact_size: Optional[float] = None,
    contact_spacing: Optional[float] = None,
    contact_size_cells: Optional[int] = None,
    contact_spacing_cells: Optional[int] = None,
    contact_edge_depth: Optional[float] = None,
    contact_edge_depth_cells: Optional[int] = None,
) -> ContactLayout:
    """Resolve y-axis size/spacing for symmetric left/right Poisson contacts.

    ``contact_size`` and ``contact_spacing`` are physical lengths in meters.
    Their ``*_cells`` counterparts can be used when exact cell counts are
    desired. Auto mode fills the y-extent with N contacts and N-1 equal gaps.
    """

    n_contacts = _coerce_positive_int(num_contacts, "num_contacts")
    if n_contacts > MAX_SUPPORTED_CONTACTS:
        raise ValueError(f"num_contacts must be <= {MAX_SUPPORTED_CONTACTS}")
    ny = _coerce_positive_int(ny, "ny")
    nx = _coerce_positive_int(nx, "nx")
    if cy <= 0.0 or cx <= 0.0:
        raise ValueError("cell sizes must be > 0")
    mode = str(contact_layout).strip().lower()
    if mode not in {"auto", "manual"}:
        raise ValueError("contact_layout must be 'auto' or 'manual'")

    edge_cells = _edge_depth_cells(
        nx=nx,
        cx=cx,
        contact_edge_depth=contact_edge_depth,
        contact_edge_depth_cells=contact_edge_depth_cells,
    )

    if mode == "manual":
        size_cells = _manual_length_cells(
            name="contact_size",
            length=contact_size,
            cells=contact_size_cells,
            cellsize=cy,
        )
        if n_contacts == 1 and contact_spacing is None and contact_spacing_cells is None:
            spacing_cells = 0
        else:
            spacing_cells = _manual_length_cells(
                name="contact_spacing",
                length=contact_spacing,
                cells=contact_spacing_cells,
                cellsize=cy,
            )
    else:
        if contact_size is not None or contact_spacing is not None:
            raise ValueError(
                "contact_size/contact_spacing require contact_layout='manual'; "
                "use *_cells or manual mode for explicit dimensions"
            )
        if contact_size_cells is not None or contact_spacing_cells is not None:
            raise ValueError("manual contact cell counts require contact_layout='manual'")
        slots = 2 * n_contacts - 1
        min_cells = max(MIN_CONTACT_CELLS, _length_to_cells(MIN_CONTACT_LENGTH, cy, "contact_size"))
        if ny < slots * min_cells:
            raise ValueError(
                f"cannot fit {n_contacts} contacts plus spacing in ny={ny}; "
                f"minimum required cells is {slots * min_cells}"
            )
        base = ny // slots
        size_cells = base
        spacing_cells = 0 if n_contacts == 1 else base

    _validate_resolved_length(name="contact_size", cells=size_cells, cellsize=cy)
    if n_contacts > 1:
        _validate_resolved_length(name="contact_spacing", cells=spacing_cells, cellsize=cy)

    total = n_contacts * size_cells + max(0, n_contacts - 1) * spacing_cells
    if total > ny:
        raise ValueError(
            f"contact layout requires {total} y cells but Poisson world has ny={ny}"
        )

    y0 = (ny - total) // 2
    ranges = []
    cursor = y0
    for _ in range(n_contacts):
        ranges.append((cursor, cursor + size_cells))
        cursor += size_cells + spacing_cells

    return ContactLayout(
        num_contacts=n_contacts,
        contact_size=_cells_to_length(size_cells, cy),
        contact_spacing=_cells_to_length(spacing_cells, cy) if n_contacts > 1 else 0.0,
        contact_size_cells=size_cells,
        contact_spacing_cells=spacing_cells,
        contact_edge_depth=_cells_to_length(edge_cells, cx),
        contact_edge_depth_cells=edge_cells,
        y_ranges=tuple(ranges),
    )


def _build_void_mask(
    *,
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    void_locations: Sequence[Sequence[float]],
    void_radius: float,
) -> np.ndarray:
    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid((ix + 0.5) * cx, (iy + 0.5) * cy)
    width = nx * cx
    height = ny * cy
    mask = np.zeros((ny, nx), dtype=bool)
    r_sq = float(void_radius) * float(void_radius)
    for loc in void_locations:
        fx, fy = float(loc[0]), float(loc[1])
        dx = xx - fx * width
        dy = yy - fy * height
        mask |= (dx * dx + dy * dy) <= r_sq
    return mask


def num_contacts_from_world_spec(world: WorldSpec) -> int:
    """Return the dense contact count implied by ``world.contact_id``."""

    contact_id = np.asarray(world.contact_id, dtype=np.int8)
    if contact_id.size == 0:
        raise ValueError("world.contact_id is empty")
    max_abs = int(np.max(np.abs(contact_id)))
    if max_abs <= 0:
        raise ValueError("Poisson world has no contact cells")
    present = set(int(v) for v in np.unique(contact_id) if int(v) != 0)
    expected = set(range(1, max_abs + 1)) | set(range(-max_abs, 0))
    if present != expected:
        raise ValueError(
            f"contact_id channels are not a dense +/-1..{max_abs} set: found {sorted(present)}"
        )
    return max_abs


def _flat_index(iz: int, iy: int, ix: int, ny: int, nx: int) -> int:
    return (iz * ny + iy) * nx + ix


def _infer_drive_contact_bands(world: WorldSpec) -> Tuple[
    int,
    Tuple[Tuple[int, int], ...],
    Tuple[int, ...],
    Tuple[int, ...],
]:
    """Infer per-channel y-bands, x-depths, and contact z layers from ``contact_id``."""

    contact_id = np.asarray(world.contact_id, dtype=np.int8)
    nz, ny, nx = _shape_tuple(world.shape)
    if contact_id.shape != (nz, ny, nx):
        raise ValueError(
            f"contact_id shape {contact_id.shape} does not match world shape {(nz, ny, nx)}"
        )
    n_contacts = num_contacts_from_world_spec(world)
    y_ranges: list[Tuple[int, int]] = []
    x_depths: list[int] = []
    z_set: set[int] = set()
    for c in range(1, n_contacts + 1):
        mask = np.abs(contact_id) == c
        if not np.any(mask):
            raise ValueError(f"contact channel {c} has no cells")
        zs, ys, xs = np.where(mask)
        y0 = int(np.min(ys))
        y1 = int(np.max(ys)) + 1
        # Depth: how far contacts penetrate from either x edge.
        left = xs[xs < nx // 2]
        right = xs[xs >= nx // 2]
        depth_left = int(np.max(left)) + 1 if left.size else 0
        depth_right = nx - int(np.min(right)) if right.size else 0
        depth = max(depth_left, depth_right)
        if depth <= 0:
            raise ValueError(f"contact channel {c} has invalid x depth")
        y_ranges.append((y0, y1))
        x_depths.append(depth)
        z_set.update(int(z) for z in zs)
    return n_contacts, tuple(y_ranges), tuple(x_depths), tuple(sorted(z_set))


def _resolve_hall_z_layers(
    world: WorldSpec,
    contact_z_layers: Tuple[int, ...],
    z_mode: Any,
) -> Tuple[int, ...]:
    nz, _, _ = _shape_tuple(world.shape)
    region = np.asarray(world.region, dtype=np.int8)
    sigma = np.asarray(world.sigma, dtype=np.float32)
    if z_mode == "contact" or z_mode is None:
        layers = contact_z_layers
    elif z_mode == "pt":
        layers = tuple(
            iz for iz in range(nz) if np.any((region[iz] == 1) & (sigma[iz] > 1e-20))
        )
    elif z_mode == "fm":
        layers = tuple(
            iz for iz in range(nz) if np.any((region[iz] == 2) & (sigma[iz] > 1e-20))
        )
    elif isinstance(z_mode, tuple) and len(z_mode) == 2 and z_mode[0] == "layers":
        layers = tuple(int(v) for v in z_mode[1])
    elif isinstance(z_mode, (list, tuple)) and all(
        isinstance(v, (int, np.integer)) for v in z_mode
    ):
        layers = tuple(int(v) for v in z_mode)
    else:
        raise ValueError(
            "z_mode must be 'contact', 'pt', 'fm', a sequence of z indices, "
            "or ('layers', indices)"
        )
    if len(layers) == 0:
        raise ValueError(f"z_mode={z_mode!r} selected no z layers")
    for iz in layers:
        if iz < 0 or iz >= nz:
            raise ValueError(f"Hall z layer {iz} out of range [0, {nz})")
    return layers


def resolve_hall_contact_geometry(
    world: WorldSpec,
    *,
    z_mode: Any = "contact",
) -> HallContactGeometry:
    """Build virtual Hall-bar probes by rotating applied contacts onto the y edges.

    Applied contacts occupy x-edge strips with per-channel y bands. Virtual Hall
    contacts use the same band widths and edge depths, rotated so each channel
    has low-y and high-y probe pads centered in the x interior (excluding the
    drive contact depths). Insulating and Dirichlet cells are excluded.
    """

    nz, ny, nx = _shape_tuple(world.shape)
    contact_id = np.asarray(world.contact_id, dtype=np.int8)
    sigma = np.asarray(world.sigma, dtype=np.float32)
    n_contacts, y_ranges, x_depths, contact_z = _infer_drive_contact_bands(world)
    z_layers = _resolve_hall_z_layers(world, contact_z, z_mode)

    # Use max drive depth for interior exclusion (contacts are usually uniform).
    max_depth = max(x_depths)
    interior0 = max_depth
    interior1 = nx - max_depth
    if interior1 <= interior0:
        raise ValueError(
            f"Hall probes need an x interior outside drive contacts, but "
            f"drive depth={max_depth} leaves no free cells for nx={nx}"
        )

    widths = [y1 - y0 for y0, y1 in y_ranges]
    gaps: list[int] = []
    for i in range(n_contacts - 1):
        gaps.append(max(0, y_ranges[i + 1][0] - y_ranges[i][1]))
    total = sum(widths) + sum(gaps)
    available = interior1 - interior0
    if total > available:
        raise ValueError(
            f"rotated Hall layout needs {total} x cells but only {available} "
            f"are available in the drive-contact-free interior "
            f"[{interior0}, {interior1}); use fewer/smaller contacts or custom geometry"
        )
    x_cursor = interior0 + (available - total) // 2
    x_ranges: list[Tuple[int, int]] = []
    for i, width in enumerate(widths):
        x_ranges.append((x_cursor, x_cursor + width))
        x_cursor += width
        if i < len(gaps):
            x_cursor += gaps[i]

    high_y_indices: list[np.ndarray] = []
    low_y_indices: list[np.ndarray] = []
    low_y_ranges: list[Tuple[int, int]] = []
    high_y_ranges: list[Tuple[int, int]] = []
    for c in range(n_contacts):
        x0, x1 = x_ranges[c]
        y_depth = x_depths[c]
        if y_depth <= 0 or 2 * y_depth > ny:
            raise ValueError(
                f"Hall y depth {y_depth} for contact {c + 1} is invalid for ny={ny}"
            )
        low_y0, low_y1 = 0, y_depth
        high_y0, high_y1 = ny - y_depth, ny
        low_y_ranges.append((low_y0, low_y1))
        high_y_ranges.append((high_y0, high_y1))

        low_idx: list[int] = []
        high_idx: list[int] = []
        for iz in z_layers:
            for iy in range(low_y0, low_y1):
                for ix in range(x0, x1):
                    if contact_id[iz, iy, ix] != 0:
                        continue
                    if float(sigma[iz, iy, ix]) <= 1e-20:
                        continue
                    low_idx.append(_flat_index(iz, iy, ix, ny, nx))
            for iy in range(high_y0, high_y1):
                for ix in range(x0, x1):
                    if contact_id[iz, iy, ix] != 0:
                        continue
                    if float(sigma[iz, iy, ix]) <= 1e-20:
                        continue
                    high_idx.append(_flat_index(iz, iy, ix, ny, nx))
        if not low_idx or not high_idx:
            raise ValueError(
                f"Hall virtual contacts for channel {c + 1} are empty after "
                "excluding insulating and Dirichlet cells"
            )
        # Virtual high/low regions must not share cells.
        if set(low_idx) & set(high_idx):
            raise ValueError(
                f"Hall high_y and low_y probes overlap for contact channel {c + 1}"
            )
        low_y_indices.append(np.asarray(low_idx, dtype=np.int64))
        high_y_indices.append(np.asarray(high_idx, dtype=np.int64))

    return HallContactGeometry(
        num_contacts=n_contacts,
        high_y_indices=tuple(high_y_indices),
        low_y_indices=tuple(low_y_indices),
        x_ranges=tuple(x_ranges),
        low_y_ranges=tuple(low_y_ranges),
        high_y_ranges=tuple(high_y_ranges),
        z_layers=z_layers,
        sign_convention="high_y_minus_low_y",
        source="auto",
    )


def hall_geometry_from_masks(
    world: WorldSpec,
    high_y_masks: Any,
    low_y_masks: Any,
) -> HallContactGeometry:
    """Build Hall geometry from boolean masks shaped ``(n, nz, ny, nx)`` or ``(nz, ny, nx)``."""

    nz, ny, nx = _shape_tuple(world.shape)
    high = np.asarray(high_y_masks, dtype=bool)
    low = np.asarray(low_y_masks, dtype=bool)
    if high.ndim == 3:
        high = high[np.newaxis, ...]
        low = low[np.newaxis, ...]
    if high.shape != (high.shape[0], nz, ny, nx) or low.shape != high.shape:
        raise ValueError(
            f"Hall masks must have shape (nz, ny, nx) or (n, nz, ny, nx) matching "
            f"{(nz, ny, nx)}; got high={high.shape}, low={low.shape}"
        )
    n_contacts = int(high.shape[0])
    high_idx: list[np.ndarray] = []
    low_idx: list[np.ndarray] = []
    for c in range(n_contacts):
        h = np.flatnonzero(high[c].reshape(-1))
        l = np.flatnonzero(low[c].reshape(-1))
        if h.size == 0 or l.size == 0:
            raise ValueError(f"custom Hall mask for contact {c + 1} is empty")
        high_idx.append(np.asarray(h, dtype=np.int64))
        low_idx.append(np.asarray(l, dtype=np.int64))
    return HallContactGeometry(
        num_contacts=n_contacts,
        high_y_indices=tuple(high_idx),
        low_y_indices=tuple(low_idx),
        x_ranges=tuple((0, nx) for _ in range(n_contacts)),
        low_y_ranges=tuple((0, 0) for _ in range(n_contacts)),
        high_y_ranges=tuple((0, 0) for _ in range(n_contacts)),
        z_layers=tuple(range(nz)),
        sign_convention="high_y_minus_low_y",
        source="custom",
    )


def build_fgat_world_spec(
    *,
    num_contacts: int = 1,
    shape: Tuple[int, int, int] = DEFAULT_FGAT_SHAPE,
    cellsize: Tuple[float, float, float] = DEFAULT_FGAT_CELLSIZE,
    first_r2_layer: int = DEFAULT_FGAT_FIRST_R2_LAYER,
    theta_sh: float = DEFAULT_FGAT_THETA_SH,
    decay_length: float = DEFAULT_FGAT_DECAY_LENGTH,
    sigma_pt: float = DEFAULT_FGAT_SIGMA_PT,
    sigma_fm: float = DEFAULT_FGAT_SIGMA_FM,
    contact_layout: str = "auto",
    contact_size: Optional[float] = None,
    contact_spacing: Optional[float] = None,
    contact_size_cells: Optional[int] = None,
    contact_spacing_cells: Optional[int] = None,
    contact_edge_depth: Optional[float] = None,
    contact_edge_depth_cells: Optional[int] = None,
    void_locations: Optional[Sequence[Sequence[float]]] = DEFAULT_FGAT_VOID_LOCATIONS,
    void_radius: float = DEFAULT_FGAT_VOID_RADIUS,
    void_sigma: float = 0.0,
) -> WorldSpec:
    """Build an in-memory FGaT Poisson world with flexible symmetric contacts.

    ``void_sigma`` controls nonmagnetic void/filler cells (``region==0``):
    ``0.0`` keeps insulating holes; ``>0`` makes them high-resistance scalar
    conductors without AMR/AHE.
    """

    nz, ny, nx = tuple(int(v) for v in shape)
    cx, cy, cz = tuple(float(v) for v in cellsize)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("shape entries must be > 0")
    if cx <= 0.0 or cy <= 0.0 or cz <= 0.0:
        raise ValueError("cellsize entries must be > 0")
    if first_r2_layer <= 0 or first_r2_layer >= nz:
        raise ValueError(f"first_r2_layer must be in (0, {nz}), got {first_r2_layer}")
    if sigma_pt <= 0.0 or sigma_fm <= 0.0:
        raise ValueError("sigma_pt and sigma_fm must be > 0")
    if float(void_sigma) < 0.0:
        raise ValueError("void_sigma must be >= 0")

    layout = resolve_contact_layout(
        num_contacts=num_contacts,
        ny=ny,
        cy=cy,
        nx=nx,
        cx=cx,
        contact_layout=contact_layout,
        contact_size=contact_size,
        contact_spacing=contact_spacing,
        contact_size_cells=contact_size_cells,
        contact_spacing_cells=contact_spacing_cells,
        contact_edge_depth=contact_edge_depth,
        contact_edge_depth_cells=contact_edge_depth_cells,
    )

    region = np.zeros((nz, ny, nx), dtype=np.int8)
    contact_id = np.zeros((nz, ny, nx), dtype=np.int8)
    sigma = np.zeros((nz, ny, nx), dtype=np.float32)

    region[:first_r2_layer, :, :] = 1
    sigma[:first_r2_layer, :, :] = np.float32(sigma_pt)

    if void_locations is None:
        void_mask = np.zeros((ny, nx), dtype=bool)
    else:
        void_mask = _build_void_mask(
            nx=nx,
            ny=ny,
            cx=cx,
            cy=cy,
            void_locations=void_locations,
            void_radius=void_radius,
        )
    fm_region = np.where(void_mask, 0, 2).astype(np.int8)
    fm_sigma = np.where(void_mask, float(void_sigma), sigma_fm).astype(np.float32)
    region[first_r2_layer:, :, :] = fm_region[np.newaxis, :, :]
    sigma[first_r2_layer:, :, :] = fm_sigma[np.newaxis, :, :]

    depth = layout.contact_edge_depth_cells
    for c, (y0, y1) in enumerate(layout.y_ranges, start=1):
        contact_id[:first_r2_layer, y0:y1, :depth] = c
        contact_id[:first_r2_layer, y0:y1, nx - depth : nx] = -c

    spec = WorldSpec(
        shape=(nz, ny, nx),
        cellsize=(cx, cy, cz),
        first_r2_layer=int(first_r2_layer),
        theta_sh=float(theta_sh),
        decay_length=float(decay_length),
        region=region,
        contact_id=contact_id,
        sigma=sigma,
    )
    detected = num_contacts_from_world_spec(spec)
    if detected != layout.num_contacts:
        raise RuntimeError(
            f"internal error: built {layout.num_contacts} contacts but detected {detected}"
        )
    return spec


def _parse_manifest_fields(manifest_path: Union[str, Path]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def world_spec_from_manifest(manifest_path: Union[str, Path]) -> WorldSpec:
    """Load a manifest and its arrays into a :class:`WorldSpec`."""

    path = Path(manifest_path)
    fields = _parse_manifest_fields(path)
    shape = (int(fields["nz"]), int(fields["ny"]), int(fields["nx"]))
    cellsize = (float(fields["cx"]), float(fields["cy"]), float(fields["cz"]))
    base = path.resolve().parent
    return WorldSpec(
        shape=shape,
        cellsize=cellsize,
        first_r2_layer=int(fields["first_r2_layer"]),
        theta_sh=float(fields["theta_sh"]),
        decay_length=float(fields["decay_length"]),
        region=np.load(base / fields["region_file"]).astype(np.int8, copy=False),
        contact_id=np.load(base / fields["contact_id_file"]).astype(np.int8, copy=False),
        sigma=np.load(base / fields["sigma_file"]).astype(np.float32, copy=False),
    )


def export_world_spec(
    world: WorldSpec,
    out_dir: Union[str, Path],
    name: str = "fgat-poisson-world",
    *,
    force: bool = False,
) -> Path:
    """Write ``world`` to manifest + npy arrays and return the manifest path."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{name}-poisson-world.txt"
    region_name = f"{name}-poisson-region.npy"
    contact_id_name = f"{name}-poisson-contact-id.npy"
    sigma_name = f"{name}-poisson-sigma.npy"
    targets = (manifest_path, out / region_name, out / contact_id_name, out / sigma_name)
    if not force:
        for target in targets:
            if target.exists():
                raise FileExistsError(f"{target} already exists (use force=True to overwrite)")

    nz, ny, nx = _shape_tuple(world.shape)
    cx, cy, cz = tuple(float(v) for v in world.cellsize)
    np.save(out / region_name, np.ascontiguousarray(world.region, dtype=np.int8))
    np.save(out / contact_id_name, np.ascontiguousarray(world.contact_id, dtype=np.int8))
    np.save(out / sigma_name, np.ascontiguousarray(world.sigma, dtype=np.float32))
    manifest_path.write_text(
        "format_version=1\n"
        f"nx={nx}\n"
        f"ny={ny}\n"
        f"nz={nz}\n"
        f"cx={cx!r}\n"
        f"cy={cy!r}\n"
        f"cz={cz!r}\n"
        f"first_r2_layer={int(world.first_r2_layer)}\n"
        f"theta_sh={float(world.theta_sh)!r}\n"
        f"decay_length={float(world.decay_length)!r}\n"
        f"region_file={region_name}\n"
        f"contact_id_file={contact_id_name}\n"
        f"sigma_file={sigma_name}\n",
        encoding="utf-8",
    )
    return manifest_path


def _resample_segment_to_nt(segment: np.ndarray, nt: int) -> np.ndarray:
    if segment.size == 0:
        raise ValueError("empty signal segment after split")
    if segment.size == nt:
        return np.ascontiguousarray(segment, dtype=np.float64)
    if nt == 1 or segment.size == 1:
        return np.full((nt,), float(segment[0]), dtype=np.float64)
    u_src = np.linspace(0.0, 1.0, int(segment.size), dtype=np.float64)
    u_tgt = np.linspace(0.0, 1.0, int(nt), dtype=np.float64)
    return np.ascontiguousarray(np.interp(u_tgt, u_src, segment), dtype=np.float64)


def load_signal_file_to_contact_potentials(
    path: str,
    nt: int,
    *,
    v_scale: float = 0.005,
    skip_first: int = 1000,
    num_contacts: int = 1,
) -> np.ndarray:
    """Load a single-column signal and split it into ``num_contacts`` channels."""

    nt = _coerce_positive_int(nt, "nt")
    n_contacts = _coerce_positive_int(num_contacts, "num_contacts")
    if skip_first < 0:
        raise ValueError("skip_first must be >= 0")
    raw = np.loadtxt(path, dtype=np.float64, comments="#")
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if skip_first >= raw.size:
        raise ValueError("skip_first removes the entire signal")
    raw = raw[int(skip_first) :]
    if raw.size < n_contacts:
        raise ValueError(
            f"need at least {n_contacts} samples for {n_contacts} contact segments"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("signal file contains NaN or Inf")
    abs_max = float(np.max(np.abs(raw))) if raw.size else 0.0
    if abs_max == 0.0:
        raw = np.zeros_like(raw, dtype=np.float64)
    else:
        raw = raw * (float(v_scale) / abs_max)

    out = np.empty((nt, n_contacts), dtype=np.float64)
    n = int(raw.size)
    for c in range(n_contacts):
        start = (c * n) // n_contacts
        end = ((c + 1) * n) // n_contacts
        out[:, c] = _resample_segment_to_nt(raw[start:end], nt)
    return np.ascontiguousarray(out)


def load_contact_potentials(path: str, num_contacts: int = 3) -> np.ndarray:
    """Load multi-column contact potentials from a text file.

    Parameters
    ----------
    path : str
        Text file containing one ``V0 V1 ... V(num_contacts-1)`` row per timestep.
    num_contacts : int
        Expected number of contact-potential columns.

    Returns
    -------
    numpy.ndarray
        Float64 array with shape ``(nt, num_contacts)``.
    """

    values = np.loadtxt(path, dtype=np.float64, comments="#")
    if values.ndim == 1:
        if values.shape[0] != num_contacts:
            raise ValueError(
                f"contact-potential text file must have {num_contacts} column(s)"
            )
        values = values.reshape(1, num_contacts)
    elif values.ndim == 2 and values.shape[1] != num_contacts:
        raise ValueError(f"contact-potential text file must have {num_contacts} column(s)")
    return _normalize_contact_potentials(values)


def _normalize_contact_potentials(contact_potentials: Any) -> np.ndarray:
    values = np.asarray(contact_potentials, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("contact_potentials must have shape (nt, num_contacts)")
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


def parse_fm_nz_spec(
    spec: str,
    n_poisson_fm: int,
    fm_mumax_nz: int = 1,
) -> Tuple[int, ...]:
    """Parse ``--fm-nz`` layer-export specification.

    Forms
    -----
    ``n`` : Poisson layer ``n`` (0 = Pt interface), broadcast to all mumax+ z cells.
    ``ni:nf`` : Poisson layers ``ni`` … ``nf - 1`` one-to-one with mumax+ z cells.
    """

    text = str(spec).strip()
    if not text:
        raise ValueError("fm-nz must not be empty")
    if fm_mumax_nz <= 0:
        raise ValueError("fm_mumax_nz must be > 0")
    parts = [int(p.strip()) for p in text.split(":")]
    if len(parts) == 1:
        layer = parts[0]
        _validate_poisson_layer(layer, n_poisson_fm)
        return (layer,)
    if len(parts) == 2:
        ni, nf = parts
        layers = _poisson_layer_range(ni, nf, n_poisson_fm)
        if len(layers) != fm_mumax_nz:
            raise ValueError(
                f"fm-nz range {ni}:{nf} selects {len(layers)} Poisson layer(s), "
                f"but mumax FM has {fm_mumax_nz} z cell(s)"
            )
        return layers
    raise ValueError("fm-nz must be n or ni:nf (e.g. 0 or 0:1)")


def _validate_poisson_layer(layer: int, n_poisson_fm: int) -> None:
    if layer < 0 or layer >= n_poisson_fm:
        raise ValueError(
            f"Poisson FM layer {layer} out of range [0, {n_poisson_fm}) "
            "(0 = at Pt interface)"
        )


def _poisson_layer_range(ni: int, nf: int, n_poisson_fm: int) -> Tuple[int, ...]:
    if nf <= ni:
        raise ValueError(f"Poisson layer range {ni}:{nf} requires nf > ni")
    if nf > n_poisson_fm:
        raise ValueError(
            f"Poisson layer range end nf={nf} exceeds maximum exclusive end "
            f"{n_poisson_fm} (0 = at Pt interface)"
        )
    if ni < 0:
        raise ValueError("Poisson layer range start must be >= 0")
    return tuple(range(ni, nf))


def _interp_poisson_stack_at_z(
    stack: np.ndarray,
    z_m: float,
    poisson_cz: float,
) -> np.ndarray:
    """Linear interpolation of ``(3, nz, ny, nx)`` at height ``z_m`` [m] from Pt interface."""

    arr = _as_mumax_vector_frame(stack)
    n_poisson = arr.shape[1]
    if n_poisson <= 0:
        raise ValueError("Poisson stack has no FM layers")
    if poisson_cz <= 0.0:
        raise ValueError("poisson_cz must be > 0")

    pos = float(z_m) / poisson_cz - 0.5
    if pos <= 0.0:
        return np.ascontiguousarray(arr[:, 0, ...], dtype=np.float32)
    if pos >= n_poisson - 1:
        return np.ascontiguousarray(arr[:, -1, ...], dtype=np.float32)

    lo = int(np.floor(pos))
    hi = lo + 1
    weight = np.float32(pos - lo)
    out = (np.float32(1.0) - weight) * arr[:, lo, ...] + weight * arr[:, hi, ...]
    return np.ascontiguousarray(out, dtype=np.float32)


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
        pcg_error=float(stats.get("pcg_error", stats["residual_rel"])),
        pcg_converged=bool(stats.get("pcg_converged", False)),
        picard_error=float(stats.get("picard_error", 0.0)),
        picard_sweeps_used=int(stats.get("picard_sweeps_used", 0)),
        timing_total_s=float(stats.get("timing_total_s", 0.0)),
        timing_device_magnetization_s=float(
            stats.get("timing_device_magnetization_s", 0.0)
        ),
        timing_transport_s=float(stats.get("timing_transport_s", 0.0)),
        timing_magnetization_set_s=float(stats.get("timing_magnetization_set_s", 0.0)),
        timing_transport_rebuild_s=float(stats.get("timing_transport_rebuild_s", 0.0)),
        timing_operator_upload_s=float(stats.get("timing_operator_upload_s", 0.0)),
        timing_rhs_build_s=float(stats.get("timing_rhs_build_s", 0.0)),
        timing_linear_solve_s=float(stats.get("timing_linear_solve_s", 0.0)),
        timing_fill_phi_s=float(stats.get("timing_fill_phi_s", 0.0)),
        timing_hall_s=float(stats.get("timing_hall_s", 0.0)),
        timing_j_raw_s=float(stats.get("timing_j_raw_s", 0.0)),
        timing_jcur_extract_s=float(stats.get("timing_jcur_extract_s", 0.0)),
        timing_jmod_postprocess_s=float(stats.get("timing_jmod_postprocess_s", 0.0)),
        timing_jmod_extract_s=float(stats.get("timing_jmod_extract_s", 0.0)),
        timing_numpy_jmod_s=float(stats.get("timing_numpy_jmod_s", 0.0)),
        timing_numpy_jcur_s=float(stats.get("timing_numpy_jcur_s", 0.0)),
    )


def _validate_transport_args(
    amr_enabled: bool,
    amr_ratio: float,
    ahe_enabled: bool,
    ahe_ratio: float,
    picard_sweeps: int,
) -> None:
    if not amr_enabled and float(amr_ratio) != 0.0:
        raise ValueError("amr_ratio requires amr_enabled=True")
    if not ahe_enabled and float(ahe_ratio) != 0.0:
        raise ValueError("ahe_ratio requires ahe_enabled=True")
    if amr_enabled and float(amr_ratio) < 0.0:
        raise ValueError("amr_ratio must be >= 0")
    if ahe_enabled and not np.isfinite(float(ahe_ratio)):
        raise ValueError("ahe_ratio must be finite when AHE is enabled")
    if ahe_enabled and int(picard_sweeps) < 1:
        raise ValueError("picard_sweeps must be >= 1 when AHE is enabled")


def _normalize_linear_solver(solver: str) -> str:
    name = str(solver).strip().lower()
    if name == "gmres":
        name = "gmres_cusparse"
    if name not in {"pcg", "gmres_cusparse"}:
        raise ValueError("solver must be 'pcg' or 'gmres_cusparse'")
    return name


def _normalize_magnetization_frame(
    magnetization: Any,
    expected_spatial: Tuple[int, int, int],
) -> np.ndarray:
    """Normalize magnetization to mumax layout ``(3, nz, ny, nx)`` unit vectors."""

    arr = np.asarray(magnetization, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 3:
            raise ValueError(
                f"magnetization shape {arr.shape} is invalid; expected (3, ny, nx)"
            )
        arr = arr[:, np.newaxis, :, :]
    arr = _as_mumax_vector_frame(arr)
    if arr.shape[1:] != expected_spatial:
        raise ValueError(
            f"magnetization shape {arr.shape} does not match Poisson export shape "
            f"(3, {expected_spatial[0]}, {expected_spatial[1]}, {expected_spatial[2]})"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("magnetization contains NaN or Inf")

    norm = np.linalg.norm(arr, axis=0, keepdims=True)
    valid = norm > 1e-12
    arr = np.where(valid, arr / np.maximum(norm, 1e-12), 0.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


class CudaPoissonSolver:
    """Persistent CUDA Poisson solver returning one current frame per call.

    The solver loads the full contact-potential series at construction, keeps
    the CUDA PCG warm start alive across ``iterate`` calls, and returns
    ``float32`` ``jmod`` and ``jcur`` already mapped to the requested mumax+ FM
    z grid. Layer export selects Poisson FM layer indices; height export samples
    the Poisson FM stack at mumax+ cell midpoints in physical z.

    All FM-layer postprocessing (raw ``jcur``, Pt injection into ``jmod``, decay)
    runs on the full Poisson FM stack inside the C++ solver.

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
        fm_nz: Optional[str] = None,
        fm_height: Optional[float] = None,
        fm_mumax_nz: int = 1,
        jmod_slice_x: str = "",
        jmod_slice_y: str = "",
        jmod_slice_z: Optional[str] = None,
        cuda_tol_batch_first: int = 1000,
        cuda_tol_batch_next: int = 500,
        amr_enabled: bool = False,
        amr_ratio: float = 0.0,
        ahe_enabled: bool = False,
        ahe_ratio: float = 0.0,
        picard_sweeps: int = 2,
        picard_tolerance: float = 0.0,
        solver: str = "gmres_cusparse",
        gmres_restart: int = 50,
    ) -> None:
        potentials = _normalize_contact_potentials(contact_potentials)
        _validate_transport_args(
            amr_enabled, amr_ratio, ahe_enabled, ahe_ratio, picard_sweeps
        )

        if world is not None and manifest_path is not None:
            raise ValueError("provide either world or manifest_path, not both")

        first_r2 = _first_r2_for_init(world, manifest_path)
        internal_slice_z = (
            jmod_slice_z if jmod_slice_z is not None else _fm_slice_z_string(first_r2)
        )

        self._world_spec = world
        self._manifest_path = (
            None if world is not None else (manifest_path or default_world_path())
        )
        self._hall_geometry: Optional[HallContactGeometry] = None
        self._hall_z_mode: Any = "contact"

        self._amr_enabled = bool(amr_enabled)
        self._ahe_enabled = bool(ahe_enabled)
        self._amr_ratio = float(amr_ratio)
        self._ahe_ratio = float(ahe_ratio)
        self._picard_sweeps = int(picard_sweeps)
        self._picard_tolerance = float(picard_tolerance)
        self._transport_enabled = bool(self._amr_enabled or self._ahe_enabled)
        self._solver = _normalize_linear_solver(solver)
        self._gmres_restart = int(gmres_restart)
        if self._gmres_restart < 2:
            raise ValueError("gmres_restart must be >= 2")

        transport_kwargs = {
            "amr_enabled": self._amr_enabled,
            "amr_ratio": self._amr_ratio,
            "ahe_enabled": self._ahe_enabled,
            "ahe_ratio": self._ahe_ratio,
            "picard_sweeps": self._picard_sweeps,
            "picard_tolerance": self._picard_tolerance,
            "solver": self._solver,
            "gmres_restart": self._gmres_restart,
        }

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
                **transport_kwargs,
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
                **transport_kwargs,
            )

        self._first_r2_layer = int(self._impl.first_r2_layer)
        self._configure_fm_export(
            fm_nz=fm_nz,
            fm_height=fm_height,
            fm_mumax_nz=fm_mumax_nz,
        )

    @classmethod
    def from_signal_file(
        cls,
        signal_path: str,
        *,
        nt: int,
        manifest_path: Optional[str] = None,
        v_scale: float = 0.005,
        skip_first: int = 1000,
        num_contacts: int = 3,
        tol: float = 1e-5,
        max_iter: int = 2000,
        skip_threshold: float = 1e-5,
        fm_nz: Optional[str] = None,
        fm_height: Optional[float] = None,
        fm_mumax_nz: int = 1,
        jmod_slice_x: str = "",
        jmod_slice_y: str = "",
        jmod_slice_z: Optional[str] = None,
        cuda_tol_batch_first: int = 1000,
        cuda_tol_batch_next: int = 500,
        amr_enabled: bool = False,
        amr_ratio: float = 0.0,
        ahe_enabled: bool = False,
        ahe_ratio: float = 0.0,
        picard_sweeps: int = 2,
        picard_tolerance: float = 0.0,
        solver: str = "gmres_cusparse",
        gmres_restart: int = 50,
    ) -> "CudaPoissonSolver":
        """Construct from a single-column signal file using C++ resampling rules."""

        _validate_transport_args(
            amr_enabled, amr_ratio, ahe_enabled, ahe_ratio, picard_sweeps
        )
        manifest = manifest_path or default_world_path()
        first_r2 = _parse_first_r2_from_manifest(manifest)
        internal_slice_z = (
            jmod_slice_z if jmod_slice_z is not None else _fm_slice_z_string(first_r2)
        )

        obj = cls.__new__(cls)
        obj._world_spec = None
        obj._manifest_path = manifest
        obj._hall_geometry = None
        obj._hall_z_mode = "contact"
        obj._amr_enabled = bool(amr_enabled)
        obj._ahe_enabled = bool(ahe_enabled)
        obj._amr_ratio = float(amr_ratio)
        obj._ahe_ratio = float(ahe_ratio)
        obj._picard_sweeps = int(picard_sweeps)
        obj._picard_tolerance = float(picard_tolerance)
        obj._transport_enabled = bool(obj._amr_enabled or obj._ahe_enabled)
        obj._solver = _normalize_linear_solver(solver)
        obj._gmres_restart = int(gmres_restart)
        if obj._gmres_restart < 2:
            raise ValueError("gmres_restart must be >= 2")
        obj._impl = _cpp.PoissonCudaSolver.from_signal_file(
            manifest,
            signal_path,
            int(nt),
            float(v_scale),
            int(skip_first),
            int(num_contacts),
            float(tol),
            int(max_iter),
            float(skip_threshold),
            jmod_slice_x,
            jmod_slice_y,
            internal_slice_z,
            int(cuda_tol_batch_first),
            int(cuda_tol_batch_next),
            bool(amr_enabled),
            float(amr_ratio),
            bool(ahe_enabled),
            float(ahe_ratio),
            int(picard_sweeps),
            float(picard_tolerance),
            obj._solver,
            obj._gmres_restart,
        )
        obj._first_r2_layer = int(obj._impl.first_r2_layer)
        obj._configure_fm_export(
            fm_nz=fm_nz,
            fm_height=fm_height,
            fm_mumax_nz=fm_mumax_nz,
        )
        return obj

    def _configure_fm_export(
        self,
        *,
        fm_nz: Optional[str],
        fm_height: Optional[float],
        fm_mumax_nz: int,
    ) -> None:
        if fm_nz is not None and fm_height is not None:
            raise ValueError("provide either fm_nz or fm_height, not both")
        if int(fm_mumax_nz) <= 0:
            raise ValueError("fm_mumax_nz must be > 0")

        self._fm_mumax_nz = int(fm_mumax_nz)
        self._fm_export_mode = "full"
        self._fm_export_layers = tuple(range(self.fm_layer_count))
        self._fm_height = None

        if fm_nz is not None:
            self._fm_export_mode = "layer"
            self._fm_export_layers = parse_fm_nz_spec(
                fm_nz,
                self.fm_layer_count,
                self._fm_mumax_nz,
            )
            return

        if fm_height is not None:
            if float(fm_height) <= 0.0:
                raise ValueError("fm_height must be > 0")
            self._fm_export_mode = "height"
            self._fm_height = float(fm_height)
            return

        self._fm_mumax_nz = self.fm_layer_count

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
    def fm_layer_count(self) -> int:
        """Number of FM layers in the Poisson world (before export selection)."""

        return int(self._impl.buffer_shape[0])

    @property
    def fm_export_mode(self) -> str:
        """Configured export mode: ``full``, ``layer``, or ``height``."""

        return self._fm_export_mode

    @property
    def fm_mumax_nz(self) -> int:
        """Number of z cells in returned mumax+ current frames."""

        return self._fm_mumax_nz

    @property
    def fm_export_layers(self) -> Tuple[int, ...]:
        """Poisson source layer index/indices used in layer/full export."""

        return self._fm_export_layers

    @property
    def fm_height(self) -> Optional[float]:
        """Height [m] used in height export, or ``None`` otherwise."""

        return self._fm_height

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
        return (3, self._fm_mumax_nz, ny, nx)

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

    @property
    def num_contacts(self) -> int:
        """Number of contact-potential channels expected by this Poisson world."""

        return int(self._impl.num_contacts)

    @property
    def transport_enabled(self) -> bool:
        """Whether AMR and/or AHE transport is enabled."""

        return self._transport_enabled

    @property
    def amr_enabled(self) -> bool:
        return self._amr_enabled

    @property
    def ahe_enabled(self) -> bool:
        return self._ahe_enabled

    @property
    def amr_ratio(self) -> float:
        return self._amr_ratio

    @property
    def ahe_ratio(self) -> float:
        return self._ahe_ratio

    @property
    def picard_sweeps(self) -> int:
        return self._picard_sweeps

    @property
    def solver(self) -> str:
        """Native linear solver backend: ``"pcg"`` or ``"gmres_cusparse"``."""

        return self._solver

    def reset(self) -> None:
        """Restart from contact frame zero and clear the CUDA warm start."""

        self._impl.reset()

    def _world_spec_for_hall(self) -> WorldSpec:
        if self._world_spec is not None:
            return self._world_spec
        if self._manifest_path is None:
            raise RuntimeError("cannot resolve Hall geometry without a world or manifest")
        self._world_spec = world_spec_from_manifest(self._manifest_path)
        return self._world_spec

    def _resolve_hall_geometry(
        self,
        geometry: Any = "auto",
        *,
        z_mode: Any = "contact",
        high_y_masks: Any = None,
        low_y_masks: Any = None,
    ) -> HallContactGeometry:
        if isinstance(geometry, HallContactGeometry):
            return geometry
        if high_y_masks is not None or low_y_masks is not None:
            if high_y_masks is None or low_y_masks is None:
                raise ValueError("provide both high_y_masks and low_y_masks")
            return hall_geometry_from_masks(
                self._world_spec_for_hall(), high_y_masks, low_y_masks
            )
        if geometry is None or geometry == "auto":
            return resolve_hall_contact_geometry(
                self._world_spec_for_hall(), z_mode=z_mode
            )
        raise ValueError(
            "geometry must be 'auto', a HallContactGeometry, or custom masks"
        )

    def _ensure_hall_geometry_configured(
        self,
        geometry: Any = "auto",
        *,
        z_mode: Any = "contact",
        high_y_masks: Any = None,
        low_y_masks: Any = None,
        force: bool = False,
    ) -> HallContactGeometry:
        custom = (
            isinstance(geometry, HallContactGeometry)
            or high_y_masks is not None
            or low_y_masks is not None
            or (geometry not in (None, "auto"))
            or z_mode != getattr(self, "_hall_z_mode", "contact")
        )
        if self._hall_geometry is not None and not force and not custom:
            return self._hall_geometry

        geom = self._resolve_hall_geometry(
            geometry,
            z_mode=z_mode,
            high_y_masks=high_y_masks,
            low_y_masks=low_y_masks,
        )
        if geom.num_contacts != self.num_contacts:
            raise ValueError(
                f"Hall geometry has {geom.num_contacts} contacts but solver expects "
                f"{self.num_contacts}"
            )
        set_fn = getattr(self._impl, "set_hall_probe_indices", None)
        if set_fn is None:
            raise RuntimeError(
                "native Poisson solver does not expose set_hall_probe_indices; rebuild mumaxplus"
            )
        set_fn(list(geom.high_y_indices), list(geom.low_y_indices))
        self._hall_geometry = geom
        self._hall_z_mode = z_mode
        return geom

    def hall_potentials(
        self,
        geometry: Any = "auto",
        *,
        z_mode: Any = "contact",
        high_y_masks: Any = None,
        low_y_masks: Any = None,
        return_components: bool = False,
    ) -> Union[np.ndarray, HallPotentialResult]:
        """Return transverse Hall voltages for each applied-potential contact.

        Requires at least one prior ``iterate()`` call. Geometry defaults to an
        automatic Hall-bar layout: virtual probes mirror applied contact size
        and spacing, rotated onto the low-y and high-y edges. Values are in
        volts with sign ``mean(phi_high_y) - mean(phi_low_y)``.

        Parameters
        ----------
        geometry : ``"auto"`` or :class:`HallContactGeometry`, optional
            Probe layout. Default resolves from the Poisson world.
        z_mode : str or sequence, optional
            Z selection for auto geometry: ``"contact"`` (default), ``"pt"``,
            ``"fm"``, or an explicit layer index sequence.
        high_y_masks, low_y_masks : array_like, optional
            Custom boolean masks shaped ``(nz, ny, nx)`` or
            ``(num_contacts, nz, ny, nx)``.
        return_components : bool, optional
            If True, return a :class:`HallPotentialResult` with means/counts.

        Returns
        -------
        numpy.ndarray or HallPotentialResult
            ``float64`` array of shape ``(num_contacts,)`` by default.
        """

        geom = self._ensure_hall_geometry_configured(
            geometry,
            z_mode=z_mode,
            high_y_masks=high_y_masks,
            low_y_masks=low_y_masks,
        )
        if return_components:
            comps_fn = getattr(self._impl, "hall_potential_components", None)
            if comps_fn is None:
                raise RuntimeError(
                    "native Poisson solver does not expose hall_potential_components; "
                    "rebuild mumaxplus"
                )
            raw = comps_fn()
            return HallPotentialResult(
                voltages=np.ascontiguousarray(raw["voltages"], dtype=np.float64),
                high_y_means=np.ascontiguousarray(raw["high_y_means"], dtype=np.float64),
                low_y_means=np.ascontiguousarray(raw["low_y_means"], dtype=np.float64),
                high_y_counts=tuple(int(v) for v in raw["high_y_counts"]),
                low_y_counts=tuple(int(v) for v in raw["low_y_counts"]),
                geometry=geom,
            )
        voltages_fn = getattr(self._impl, "hall_potentials", None)
        if voltages_fn is None:
            raise RuntimeError(
                "native Poisson solver does not expose hall_potentials; rebuild mumaxplus"
            )
        return np.ascontiguousarray(voltages_fn(), dtype=np.float64)

    def _average_magnetization_over_z(self, m: np.ndarray) -> np.ndarray:
        """Average mumax ``(3, nz, ny, nx)`` over z and renormalize to unit vectors.

        Near-zero columns (voids) remain zero. The result is shaped ``(3, ny, nx)``.
        """

        arr = _as_mumax_vector_frame(m)
        m_avg = np.mean(arr, axis=1, dtype=np.float64).astype(np.float32)
        norm = np.linalg.norm(m_avg, axis=0, keepdims=True)
        valid = norm > 1e-12
        m_avg = np.where(valid, m_avg / np.maximum(norm, 1e-12), 0.0)
        return np.ascontiguousarray(m_avg, dtype=np.float32)

    def _broadcast_magnetization_to_fm_stack(self, m_xy: np.ndarray) -> np.ndarray:
        """Broadcast in-plane ``(3, ny, nx)`` magnetization to all Poisson FM layers."""

        arr = np.asarray(m_xy, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(
                f"expected in-plane magnetization (3, ny, nx), got shape {arr.shape}"
            )
        n_fm = self.fm_layer_count
        _, ny, nx = arr.shape
        out = np.broadcast_to(arr[:, np.newaxis, :, :], (3, n_fm, ny, nx))
        return np.ascontiguousarray(out.copy(), dtype=np.float32)

    def _map_magnetization_to_poisson_fm_stack(self, m: np.ndarray) -> np.ndarray:
        """Map mumax export-shaped magnetization to the full Poisson FM stack.

        If mumax has fewer z cells than the Poisson FM stack, average those
        mumax layers and broadcast the result to every Poisson FM layer
        (uniform-z magnetization). The electrical solve still uses the full
        Poisson FM stack; only the magnetization input is z-extended.
        """

        arr = _as_mumax_vector_frame(m)
        n_fm = self.fm_layer_count
        nz_m, ny, nx = arr.shape[1:]

        if nz_m == n_fm:
            return np.ascontiguousarray(arr, dtype=np.float32)

        if nz_m < n_fm:
            return self._broadcast_magnetization_to_fm_stack(
                self._average_magnetization_over_z(arr)
            )

        # mumax nz > Poisson FM layers: mode-specific downsample into the stack.
        if self._fm_export_mode == "full":
            raise ValueError(
                f"magnetization nz={nz_m} does not match Poisson FM layers {n_fm}"
            )

        if self._fm_export_mode == "layer":
            out = np.zeros((3, n_fm, ny, nx), dtype=np.float32)
            layers = self._fm_export_layers
            if nz_m != len(layers):
                raise ValueError(
                    f"magnetization nz={nz_m} does not match selected layers {layers}"
                )
            for i, layer in enumerate(layers):
                out[:, layer, :, :] = arr[:, i, :, :]
            selected = sorted(set(layers))
            for layer in range(n_fm):
                if layer in selected:
                    continue
                nearest = min(selected, key=lambda s: abs(s - layer))
                out[:, layer, :, :] = out[:, nearest, :, :]
            return np.ascontiguousarray(out, dtype=np.float32)

        if self._fm_height is None:
            raise RuntimeError("height export is missing fm_height")
        out = np.empty((3, n_fm, ny, nx), dtype=np.float32)
        fm_cz = self._fm_height / max(nz_m, 1)
        poisson_cz = self.cellsize[2]
        for iz in range(n_fm):
            z_mid = (iz + 0.5) * poisson_cz
            pos = z_mid / max(fm_cz, 1e-30) - 0.5
            if pos <= 0.0:
                out[:, iz, ...] = arr[:, 0, ...]
            elif pos >= nz_m - 1:
                out[:, iz, ...] = arr[:, -1, ...]
            else:
                lo = int(np.floor(pos))
                hi = lo + 1
                weight = np.float32(pos - lo)
                out[:, iz, ...] = (1.0 - weight) * arr[:, lo, ...] + weight * arr[:, hi, ...]
        return np.ascontiguousarray(out, dtype=np.float32)

    def _device_magnetization_mapping(
        self,
        source_nz: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return per-Poisson-layer device mapping arrays for native Variable input."""

        n_fm = self.fm_layer_count
        src_nz = int(source_nz)
        cache_key = (
            src_nz,
            n_fm,
            self._fm_export_mode,
            self._fm_height,
            tuple(self._fm_export_layers) if self._fm_export_layers is not None else None,
        )
        cached = getattr(self, "_device_mag_mapping_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        if src_nz <= 0:
            raise ValueError("magnetization source nz must be > 0")

        if src_nz < n_fm:
            zeros_i = np.zeros(n_fm, dtype=np.int32)
            zeros_f = np.zeros(n_fm, dtype=np.float32)
            result = (zeros_i, zeros_i, zeros_f, True)
        elif src_nz == n_fm:
            idx = np.arange(n_fm, dtype=np.int32)
            weight = np.zeros(n_fm, dtype=np.float32)
            result = (idx, idx.copy(), weight, False)
        elif self._fm_export_mode == "full":
            raise ValueError(
                f"magnetization nz={src_nz} does not match Poisson FM layers {n_fm}"
            )
        elif self._fm_export_mode == "layer":
            layers = self._fm_export_layers
            if src_nz != len(layers):
                raise ValueError(
                    f"magnetization nz={src_nz} does not match selected layers {layers}"
                )
            selected = sorted(set(layers))
            lo = np.empty(n_fm, dtype=np.int32)
            for layer in range(n_fm):
                nearest = min(selected, key=lambda s: abs(s - layer))
                lo[layer] = layers.index(nearest)
            weight = np.zeros(n_fm, dtype=np.float32)
            result = (lo, lo.copy(), weight, False)
        else:
            if self._fm_height is None:
                raise RuntimeError("height export is missing fm_height")
            lo = np.empty(n_fm, dtype=np.int32)
            hi = np.empty(n_fm, dtype=np.int32)
            weight = np.empty(n_fm, dtype=np.float32)
            fm_cz = self._fm_height / max(src_nz, 1)
            poisson_cz = self.cellsize[2]
            for iz in range(n_fm):
                z_mid = (iz + 0.5) * poisson_cz
                pos = z_mid / max(fm_cz, 1e-30) - 0.5
                if pos <= 0.0:
                    lo[iz] = hi[iz] = 0
                    weight[iz] = 0.0
                elif pos >= src_nz - 1:
                    lo[iz] = hi[iz] = src_nz - 1
                    weight[iz] = 0.0
                else:
                    lo_i = int(np.floor(pos))
                    lo[iz] = lo_i
                    hi[iz] = lo_i + 1
                    weight[iz] = np.float32(pos - lo_i)
            result = (lo, hi, weight, False)

        self._device_mag_mapping_cache = (cache_key, result)
        return result

    def _map_fm_export(self, frame: np.ndarray) -> np.ndarray:
        arr = _as_mumax_vector_frame(frame)
        _, ny, nx = arr.shape[1:]

        if self._fm_export_mode == "full":
            return arr

        if self._fm_export_mode == "layer":
            layers = self._fm_export_layers
            if len(layers) == 1:
                slab = arr[:, layers[0] : layers[0] + 1, ...]
                return np.ascontiguousarray(
                    np.broadcast_to(slab, (3, self._fm_mumax_nz, ny, nx)).copy(),
                    dtype=np.float32,
                )
            return np.ascontiguousarray(arr[:, list(layers), ...], dtype=np.float32)

        if self._fm_height is None:
            raise RuntimeError("height export is missing fm_height")
        out = np.empty((3, self._fm_mumax_nz, ny, nx), dtype=np.float32)
        fm_cz = self._fm_height / self._fm_mumax_nz
        poisson_cz = self.cellsize[2]
        for iz in range(self._fm_mumax_nz):
            z_mid = (iz + 0.5) * fm_cz
            out[:, iz, ...] = _interp_poisson_stack_at_z(arr, z_mid, poisson_cz)
        return out

    def iterate(self, magnetization: Optional[Any] = None) -> PoissonStepResult:
        """Solve the next contact-potential frame.

        Parameters
        ----------
        magnetization : array_like, optional
            Magnetization in mumax layout ``(3, nz_export, ny, nx)`` or
            ``(3, ny, nx)``. Required when AMR/AHE transport is enabled
            (except for skipped near-zero contact frames). Ignored on the
            scalar path. If ``nz_export`` is smaller than the Poisson FM
            stack, the solver averages mumax z-layers and broadcasts that
            uniform-z magnetization across all Poisson FM layers.

        Returns
        -------
        PoissonStepResult
            ``jmod`` and ``jcur`` are independent ``float32`` NumPy arrays already
            mapped to ``self.output_shape``. The next call to ``iterate`` will not
            mutate previously returned arrays.
        """

        timing_python_magnetization_s = 0.0
        timing_native_call_s = 0.0
        if self._transport_enabled:
            if magnetization is not None:
                from mumaxplus.variable import Variable

                t_mag0 = time.perf_counter()
                if isinstance(magnetization, Variable) and hasattr(
                    self._impl, "iterate_with_magnetization_variable"
                ):
                    shape = tuple(int(v) for v in getattr(magnetization, "shape"))
                    if len(shape) != 4 or shape[0] != 3:
                        raise ValueError(
                            f"magnetization shape {shape} is invalid; expected "
                            "(3, nz, ny, nx)"
                        )
                    if shape[2:] != self.output_shape[2:]:
                        raise ValueError(
                            f"magnetization xy shape {shape[2:]} does not match Poisson "
                            f"xy shape {self.output_shape[2:]}"
                        )
                    src_lo, src_hi, weight_hi, average_z = self._device_magnetization_mapping(
                        shape[1]
                    )
                    timing_python_magnetization_s = time.perf_counter() - t_mag0
                    t_native0 = time.perf_counter()
                    raw = self._impl.iterate_with_magnetization_variable(
                        magnetization._impl,
                        src_lo,
                        src_hi,
                        weight_hi,
                        average_z,
                    )
                    timing_native_call_s = time.perf_counter() - t_native0
                else:
                    m_export = _normalize_magnetization_frame(
                        magnetization, self.output_shape[1:]
                    )
                    m_stack = self._map_magnetization_to_poisson_fm_stack(m_export)
                    timing_python_magnetization_s = time.perf_counter() - t_mag0
                    t_native0 = time.perf_counter()
                    raw = self._impl.iterate_with_magnetization(m_stack)
                    timing_native_call_s = time.perf_counter() - t_native0
            else:
                # Allow skipped near-zero frames without magnetization; C++ raises
                # if the frame is not skipped and transport is enabled.
                t_native0 = time.perf_counter()
                raw = self._impl.iterate()
                timing_native_call_s = time.perf_counter() - t_native0
        else:
            t_native0 = time.perf_counter()
            raw = self._impl.iterate()
            timing_native_call_s = time.perf_counter() - t_native0
        t_out0 = time.perf_counter()
        jmod = self._map_fm_export(raw["jmod"])
        jcur = self._map_fm_export(raw["jcur"])
        timing_python_output_map_s = time.perf_counter() - t_out0
        stats = _stats_from_dict(raw["stats"])
        stats = replace(
            stats,
            timing_python_magnetization_s=timing_python_magnetization_s,
            timing_native_call_s=timing_native_call_s,
            timing_python_output_map_s=timing_python_output_map_s,
        )
        return PoissonStepResult(
            jmod=jmod,
            jcur=jcur,
            stats=stats,
        )

    def check_compatible(
        self,
        grid_shape: Any,
        cellsize: Optional[Tuple[float, float, float]] = None,
        *,
        check_cellsize_z: bool = True,
        rtol: float = 1e-6,
    ) -> Dict[str, Any]:
        """Check Poisson xy grid and cell size against a mumax+ ferromagnet grid.

        Compares the mumax grid shape to ``self.output_shape``. Set
        ``check_cellsize_z=False`` when mumax+ ``cz`` intentionally differs from
        Poisson ``cz``.

        This method only inspects metadata and does not advance the Poisson
        contact-potential series.
        """

        expected = self.output_shape[1:]
        actual = _spatial_shape(grid_shape)
        if actual != expected:
            raise ValueError(
                f"Poisson export shape {expected} is incompatible with mumax grid "
                f"shape {actual} (mode={self.fm_export_mode})"
            )

        if cellsize is not None:
            cellsize_actual = tuple(float(v) for v in cellsize)
            if len(cellsize_actual) != 3:
                raise ValueError("cellsize must have length 3")
            poisson_cs = self.cellsize
            axes = (0, 1, 2) if check_cellsize_z else (0, 1)
            if not np.allclose(
                [cellsize_actual[i] for i in axes],
                [poisson_cs[i] for i in axes],
                rtol=rtol,
                atol=0.0,
            ):
                raise ValueError(
                    f"Poisson cellsize {poisson_cs} is incompatible with {cellsize_actual}"
                )

        return {
            "shape": self.output_shape,
            "cellsize": self.cellsize,
            "n_steps": self.n_steps,
            "world_shape": self.world_shape,
            "fm_layer_count": self.fm_layer_count,
            "first_fm_layer": self.first_fm_layer,
            "fm_export_mode": self.fm_export_mode,
            "fm_export_layers": self.fm_export_layers,
            "fm_height": self.fm_height,
        }

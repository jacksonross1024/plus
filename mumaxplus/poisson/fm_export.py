"""Map Poisson FM current frames onto a micromagnetic FM stack.

The electrical solve stores one ``jmod`` and ``jcur`` slice per Poisson FM
cell. Those slices do not have to be copied one-to-one onto the ferromagnet
used by the LLG. :func:`map_fm_currents` resamples a chosen set of Poisson
layers onto any positive number of LLG layers.

``jcur`` is the raw FM current. Selected Poisson layers are stacked in order
and split evenly across the LLG thickness.

``jmod`` is the heavy-metal injection profile. Three profiles are available:

``sample``
    Keep the Poisson cell values and resample them the same way as ``jcur``.
``exponential``
    Rebuild the profile from the interface current and ``exp(-z/λ)``, using
    the real-space z interval of each LLG cell.
``average``
    Integrate that exponential from the Pt interface to a chosen depth, then
    give every LLG layer the same value. The thickness-integrated injection
    equals the exponential integral.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

JMOD_PROFILES = ("sample", "exponential", "average")


def fm_injection_decay_factor(fm_layer_index: int, cz: float, decay_length: float) -> float:
    """Native per-cell decay used inside the Poisson ``jmod`` postprocess.

    Arithmetic mean of ``exp(-z/λ)`` at the bottom, middle, and top of Poisson
    FM layer ``fm_layer_index`` (``0`` is the Pt interface). Returns ``1`` when
    ``decay_length`` or ``cz`` is not positive.
    """

    if not (decay_length > 0.0) or not (cz > 0.0):
        return 1.0
    z_bottom = float(fm_layer_index) * float(cz)
    z_mid = z_bottom + 0.5 * float(cz)
    z_top = z_bottom + float(cz)
    inv_lambda = 1.0 / float(decay_length)
    return (
        math.exp(-z_bottom * inv_lambda)
        + math.exp(-z_mid * inv_lambda)
        + math.exp(-z_top * inv_lambda)
    ) / 3.0


def exponential_integral(z0: float, z1: float, decay_length: float) -> float:
    """Integral of ``exp(-z/λ)`` from ``z0`` to ``z1``.

    Returns ``z1 - z0`` when ``decay_length`` is not positive.
    """

    if float(z1) < float(z0):
        raise ValueError(f"exponential interval requires z1 >= z0, got [{z0}, {z1}]")
    if not (decay_length > 0.0):
        return float(z1) - float(z0)
    return float(decay_length) * (
        math.exp(-float(z0) / float(decay_length)) - math.exp(-float(z1) / float(decay_length))
    )


def exponential_interval_mean(z0: float, z1: float, decay_length: float) -> float:
    """Thickness average of ``exp(-z/λ)`` on ``[z0, z1]``."""

    dz = float(z1) - float(z0)
    if dz <= 0.0:
        raise ValueError("exponential interval must have positive thickness")
    return exponential_integral(z0, z1, decay_length) / dz


def resolve_fm_cellsize_z(
    source_layers: Sequence[int],
    n_out: int,
    poisson_cz: float,
    *,
    fm_height: Optional[float] = None,
    fm_cellsize_z: Optional[float] = None,
) -> float:
    """Z cell size of the LLG stack that receives the exported currents.

    An explicit ``fm_cellsize_z`` or ``fm_height`` wins. Otherwise a one-to-one
    export, or a single Poisson layer broadcast onto many LLG layers, keeps
    the Poisson cell size. A different number of layers shares the thickness
    of the selected Poisson block, so two Poisson layers become one LLG layer
    of thickness ``2*cz`` or eight LLG layers of thickness ``2*cz/8``.
    """

    if fm_cellsize_z is not None and fm_height is not None:
        raise ValueError("provide either fm_cellsize_z or fm_height, not both")
    n_out = int(n_out)
    if n_out <= 0:
        raise ValueError("n_out must be > 0")
    if not (float(poisson_cz) > 0.0):
        raise ValueError("poisson_cz must be > 0")
    if fm_cellsize_z is not None:
        if not (float(fm_cellsize_z) > 0.0):
            raise ValueError("fm_cellsize_z must be > 0")
        return float(fm_cellsize_z)
    if fm_height is not None:
        if not (float(fm_height) > 0.0):
            raise ValueError("fm_height must be > 0")
        return float(fm_height) / n_out

    layers = tuple(int(v) for v in source_layers)
    if len(layers) > 1 and n_out != len(layers):
        span_layers = max(layers) - min(layers) + 1
        return span_layers * float(poisson_cz) / n_out
    return float(poisson_cz)


def default_jmod_depth(source_layers: Sequence[int], poisson_cz: float) -> float:
    """Distance from the Pt interface to the top of the last selected layer."""

    layers = tuple(int(v) for v in source_layers)
    if not layers:
        raise ValueError("source_layers must not be empty")
    if not (float(poisson_cz) > 0.0):
        raise ValueError("poisson_cz must be > 0")
    return (max(layers) + 1) * float(poisson_cz)


def _as_vector_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"expected current frame (3, nz, ny, nx), got {arr.shape}")
    return arr


def _overlap_units(out_index: int, src_index: int, n_out: int, n_src: int) -> int:
    """Overlap of two uniform partitions, in units of ``1/(n_out*n_src)``."""

    a0 = out_index * n_src
    a1 = (out_index + 1) * n_src
    b0 = src_index * n_out
    b1 = (src_index + 1) * n_out
    return max(0, min(a1, b1) - max(a0, b0))


def resample_stacked_layers(
    frame: np.ndarray,
    source_layers: Sequence[int],
    n_out: int,
) -> np.ndarray:
    """Stack ``source_layers`` in order and split them evenly over ``n_out`` layers.

    One source layer is broadcast. An equal number of layers is copied in the
    given order. Any other pair of counts uses a thickness-weighted average, so
    two Poisson layers collapsed onto one LLG layer are their mean, and two
    Poisson layers spread onto eight LLG layers fill the bottom half from the
    first source and the top half from the second.
    """

    arr = _as_vector_frame(frame)
    layers = tuple(int(v) for v in source_layers)
    n_out = int(n_out)
    n_src = len(layers)
    n_full = int(arr.shape[1])
    if n_out <= 0:
        raise ValueError("n_out must be > 0")
    if n_src == 0:
        raise ValueError("source_layers must not be empty")
    for layer in layers:
        if layer < 0 or layer >= n_full:
            raise ValueError(
                f"source layer {layer} is outside the Poisson FM stack of length {n_full}"
            )

    _, _, ny, nx = arr.shape
    if n_src == 1:
        slab = arr[:, layers[0] : layers[0] + 1, ...]
        return np.ascontiguousarray(
            np.broadcast_to(slab, (3, n_out, ny, nx)).copy(),
            dtype=np.float32,
        )
    if n_src == n_out:
        return np.ascontiguousarray(arr[:, list(layers), ...], dtype=np.float32)

    out = np.empty((3, n_out, ny, nx), dtype=np.float32)
    for i_out in range(n_out):
        acc = np.zeros((3, ny, nx), dtype=np.float64)
        weight = 0
        for i_src, layer in enumerate(layers):
            overlap = _overlap_units(i_out, i_src, n_out, n_src)
            if overlap == 0:
                continue
            acc += float(overlap) * arr[:, layer, ...]
            weight += overlap
        out[:, i_out, ...] = (acc / float(weight)).astype(np.float32)
    return out


def _interface_current(
    jmod: np.ndarray,
    source_layers: Sequence[int],
    poisson_cz: float,
    decay_length: float,
) -> np.ndarray:
    """Recover ``J_pt`` from a postprocessed ``jmod`` stack.

    Native ``jmod`` on Poisson layer ``k`` is ``J_pt * fm_injection_decay_factor(k)``.
    """

    if not (decay_length > 0.0):
        raise ValueError("exponential and average jmod profiles require decay_length > 0")
    layer = int(source_layers[0])
    factor = fm_injection_decay_factor(layer, poisson_cz, decay_length)
    if factor <= 0.0:
        raise ValueError("interface decay factor is zero; cannot recover J_pt")
    return np.asarray(jmod[:, layer, ...], dtype=np.float64) / factor


def map_fm_currents(
    jmod: np.ndarray,
    jcur: np.ndarray,
    *,
    source_layers: Sequence[int],
    n_out: int,
    poisson_cz: float,
    fm_cz: float,
    decay_length: float,
    jmod_profile: str = "sample",
    jmod_depth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample full-stack Poisson currents onto ``n_out`` LLG layers.

    Parameters
    ----------
    jmod, jcur :
        Full Poisson FM stacks, mumax layout ``(3, n_fm, ny, nx)``. ``jmod``
        must already include the native Pt-average and per-cell exponential
        postprocess when ``jmod_profile`` is ``exponential`` or ``average``.
    source_layers :
        Poisson FM indices to export, ``0`` at the Pt interface, in the order
        they should be applied.
    n_out :
        Number of LLG layers in the returned frames.
    poisson_cz, fm_cz :
        Poisson and LLG z cell sizes in meters. ``fm_cz`` is the thickness of
        each returned layer. Layer ``i`` occupies ``[i*fm_cz, (i+1)*fm_cz]``
        measured from the Pt interface.
    decay_length :
        Spin-current decay length in meters.
    jmod_profile :
        ``sample``, ``exponential``, or ``average``.
    jmod_depth :
        Upper limit, in meters from the interface, of the ``average`` integral.
        The default is the top of the last selected Poisson layer.

    Returns
    -------
    tuple of numpy.ndarray
        ``(jmod_out, jcur_out)``, each ``(3, n_out, ny, nx)`` and ``float32``.
    """

    profile = str(jmod_profile).strip().lower()
    if profile not in JMOD_PROFILES:
        raise ValueError(
            "jmod_profile must be 'sample', 'exponential', or 'average', "
            f"got {jmod_profile!r}"
        )
    if not (float(fm_cz) > 0.0):
        raise ValueError("fm_cz must be > 0")
    if not (float(poisson_cz) > 0.0):
        raise ValueError("poisson_cz must be > 0")

    layers = tuple(int(v) for v in source_layers)
    n_out = int(n_out)
    jmod_arr = _as_vector_frame(jmod)
    jcur_arr = _as_vector_frame(jcur)
    if jmod_arr.shape != jcur_arr.shape:
        raise ValueError(
            f"jmod shape {jmod_arr.shape} does not match jcur shape {jcur_arr.shape}"
        )

    jcur_out = resample_stacked_layers(jcur_arr, layers, n_out)
    if profile == "sample":
        return resample_stacked_layers(jmod_arr, layers, n_out), jcur_out

    interface = _interface_current(jmod_arr, layers, float(poisson_cz), float(decay_length))
    _, _, ny, nx = jmod_arr.shape
    jmod_out = np.empty((3, n_out, ny, nx), dtype=np.float32)
    if profile == "exponential":
        for iz in range(n_out):
            z0 = iz * float(fm_cz)
            factor = exponential_interval_mean(z0, z0 + float(fm_cz), float(decay_length))
            jmod_out[:, iz, ...] = (interface * factor).astype(np.float32)
        return jmod_out, jcur_out

    depth = (
        default_jmod_depth(layers, float(poisson_cz))
        if jmod_depth is None
        else float(jmod_depth)
    )
    if not (depth > 0.0):
        raise ValueError("jmod_depth must be > 0")
    thickness = n_out * float(fm_cz)
    factor = exponential_integral(0.0, depth, float(decay_length)) / thickness
    uniform = (interface * factor).astype(np.float32)
    for iz in range(n_out):
        jmod_out[:, iz, ...] = uniform
    return jmod_out, jcur_out

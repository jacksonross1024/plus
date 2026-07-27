#include "poisson_gmres_cuda.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "poisson_world.hpp"

namespace {

constexpr double kAtol = 1e-30;

bool residual_meets_tolerance(double r_inf, double rhs_inf, double rtol) {
  return r_inf <= kAtol + rtol * rhs_inf;
}

void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

void check_cublas(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cublas error " + std::to_string(s));
  }
}

void check_cusparse(cusparseStatus_t s, const char* what) {
  if (s != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cusparse error " + std::to_string(s));
  }
}

__global__ void k_residual(double* r, const double* rhs, const double* ax, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    r[i] = rhs[i] - ax[i];
  }
}

__global__ void k_jacobi(double* z, const double* r, const double* inv_diag, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = r[i] * inv_diag[i];
  }
}

struct SymTensor6D {
  float xx = 0.0f;
  float yy = 0.0f;
  float zz = 0.0f;
  float xy = 0.0f;
  float xz = 0.0f;
  float yz = 0.0f;
};

struct SkewTensor3D {
  float xy = 0.0f;
  float xz = 0.0f;
  float yz = 0.0f;
};

__device__ bool d_in_bounds(int value, int limit) {
  return value >= 0 && value < limit;
}

__device__ int d_flat_index(int iz, int iy, int ix, int ny, int nx) {
  return (iz * ny + iy) * nx + ix;
}

__device__ double d_face_area(int axis, double cx, double cy, double cz) {
  if (axis == 0) {
    return cy * cz;
  }
  if (axis == 1) {
    return cx * cz;
  }
  return cx * cy;
}

__device__ double d_axis_spacing(int axis, double cx, double cy, double cz) {
  if (axis == 0) {
    return cx;
  }
  if (axis == 1) {
    return cy;
  }
  return cz;
}

__device__ bool d_is_conducting(const float* sigma, int cell) {
  return sigma[cell] > 1e-20f;
}

__device__ bool d_uses_magnetization(const signed char* region, const float* sigma, int cell) {
  return d_is_conducting(sigma, cell) && region[cell] == 2;
}

__device__ void d_load_magnetization_fm_stack(float& mx,
                                              float& my,
                                              float& mz,
                                              int fm_layer,
                                              int xy,
                                              int n_fm,
                                              int plane,
                                              const float* magnetization) {
  const std::size_t n_xy = static_cast<std::size_t>(n_fm) * static_cast<std::size_t>(plane);
  const std::size_t base =
      static_cast<std::size_t>(fm_layer) * static_cast<std::size_t>(plane) +
      static_cast<std::size_t>(xy);
  mx = magnetization[base];
  my = magnetization[n_xy + base];
  mz = magnetization[2u * n_xy + base];
  const float norm = sqrtf(mx * mx + my * my + mz * mz);
  if (norm > 1e-12f) {
    mx /= norm;
    my /= norm;
    mz /= norm;
  } else {
    mx = 0.0f;
    my = 0.0f;
    mz = 0.0f;
  }
}

// region==2 cells are validated at prepare_transport_update to lie in FM layers;
// no per-cell fm_layer range check on the hot path.
__device__ SymTensor6D d_sym_tensor_for_cell(int cell,
                                            const signed char* region,
                                            const float* sigma,
                                            int nx,
                                            int ny,
                                            int first_r2_layer,
                                            int n_fm,
                                            bool amr_enabled,
                                            double amr_ratio,
                                            const float* magnetization) {
  const float s = sigma[cell];
  if (!(s > 1e-20f)) {
    return {};
  }
  if (!amr_enabled || region[cell] != 2) {
    return {s, s, s, 0.0f, 0.0f, 0.0f};
  }

  const int plane = nx * ny;
  const int iz = cell / plane;
  const int rem = cell - iz * plane;
  const int iy = rem / nx;
  const int ix = rem - iy * nx;
  const int fm_layer = iz - first_r2_layer;

  float mx = 0.0f;
  float my = 0.0f;
  float mz = 0.0f;
  d_load_magnetization_fm_stack(mx, my, mz, fm_layer, iy * nx + ix, n_fm, plane, magnetization);
  const double q = 6.0 * amr_ratio / (6.0 + amr_ratio);
  const double base_s = static_cast<double>(s);
  return {static_cast<float>(base_s * (1.0 - q * (static_cast<double>(mx) * mx - 1.0 / 3.0))),
          static_cast<float>(base_s * (1.0 - q * (static_cast<double>(my) * my - 1.0 / 3.0))),
          static_cast<float>(base_s * (1.0 - q * (static_cast<double>(mz) * mz - 1.0 / 3.0))),
          static_cast<float>(-base_s * q * static_cast<double>(mx) * my),
          static_cast<float>(-base_s * q * static_cast<double>(mx) * mz),
          static_cast<float>(-base_s * q * static_cast<double>(my) * mz)};
}

__device__ SkewTensor3D d_skew_tensor_for_cell(int cell,
                                              const signed char* region,
                                              const float* sigma,
                                              int nx,
                                              int ny,
                                              int first_r2_layer,
                                              int n_fm,
                                              bool ahe_enabled,
                                              double ahe_ratio,
                                              const float* magnetization) {
  if (!ahe_enabled || !d_uses_magnetization(region, sigma, cell)) {
    return {};
  }
  const int plane = nx * ny;
  const int iz = cell / plane;
  const int rem = cell - iz * plane;
  const int iy = rem / nx;
  const int ix = rem - iy * nx;
  const int fm_layer = iz - first_r2_layer;
  float mx = 0.0f;
  float my = 0.0f;
  float mz = 0.0f;
  d_load_magnetization_fm_stack(mx, my, mz, fm_layer, iy * nx + ix, n_fm, plane, magnetization);
  if (mx == 0.0f && my == 0.0f && mz == 0.0f) {
    return {};
  }
  const float sigma_ahe = static_cast<float>(ahe_ratio * static_cast<double>(sigma[cell]));
  return {-sigma_ahe * mz, sigma_ahe * my, -sigma_ahe * mx};
}

__device__ float d_avg_diag(float a, float b) {
  if (!(a > 0.0f) || !(b > 0.0f)) {
    return 0.0f;
  }
  return static_cast<float>(sqrt(static_cast<double>(a) * static_cast<double>(b)));
}

__device__ float d_avg_signed(float a, float b) {
  return 0.5f * (a + b);
}

__device__ void d_add_matrix_coeff(int row,
                                   int row_cell,
                                   int col_cell,
                                   double coeff,
                                   const int* unknown_index,
                                   const signed char* contact_id,
                                   const double* potentials,
                                   const int* row_off,
                                   const int* col_idx,
                                   double* val,
                                   double& rhs,
                                   double& diag,
                                   int* fail,
                                   bool skew) {
  if (col_cell < 0 || coeff == 0.0) {
    return;
  }
  // Host stores SPD diagonal separately and never uploads skew diagonal into the
  // merged GMRES CSR. Mirror that: SPD diag -> Jacobi + CSR diag; drop skew diag.
  if (row_cell == col_cell) {
    if (skew) {
      return;
    }
    const float c = static_cast<float>(coeff);
    diag += static_cast<double>(c);
    const int col = unknown_index[col_cell];
    if (col < 0) {
      atomicExch(fail, 1);
      return;
    }
    const int start = row_off[row];
    const int end = row_off[row + 1];
    for (int p = start; p < end; ++p) {
      if (col_idx[p] == col) {
        val[p] += static_cast<double>(c);
        return;
      }
    }
    atomicExch(fail, 1);
    return;
  }
  const int col = unknown_index[col_cell];
  if (col >= 0) {
    // Host offdiag stores float(-coeff); GMRES merge does val -= offd (SPD) and
    // val += skew_val (skew) => merged A_ij += coeff (SPD) or A_ij += -coeff (skew).
    const double matrix_coeff =
        skew ? static_cast<double>(static_cast<float>(-coeff))
             : static_cast<double>(static_cast<float>(coeff));
    const int start = row_off[row];
    const int end = row_off[row + 1];
    for (int p = start; p < end; ++p) {
      if (col_idx[p] == col) {
        val[p] += matrix_coeff;
        return;
      }
    }
    atomicExch(fail, 1);
    return;
  }
  const int cid = static_cast<int>(contact_id[col_cell]);
  if (cid != 0) {
    // contact_id channels validated at world construction; abs(cid)-1 is in range.
    const int channel = (cid > 0 ? cid : -cid) - 1;
    const double sign = cid > 0 ? 1.0 : -1.0;
    const double rhs_weight = static_cast<double>(static_cast<float>(-coeff * sign));
    rhs += rhs_weight * potentials[channel];
  }
}

__device__ void d_add_diff(int row,
                           int row_cell,
                           int c1,
                           int c2,
                           double coeff,
                           const int* unknown_index,
                           const signed char* contact_id,
                           const double* potentials,
                           const int* row_off,
                           const int* col_idx,
                           double* val,
                           double& rhs,
                           double& diag,
                           int* fail,
                           bool skew) {
  if (c1 < 0 || c2 < 0 || coeff == 0.0) {
    return;
  }
  d_add_matrix_coeff(row, row_cell, c1, coeff, unknown_index, contact_id, potentials, row_off,
                     col_idx, val, rhs, diag, fail, skew);
  d_add_matrix_coeff(row, row_cell, c2, -coeff, unknown_index, contact_id, potentials, row_off,
                     col_idx, val, rhs, diag, fail, skew);
}

__device__ int d_cell_at(int iz,
                         int iy,
                         int ix,
                         int nx,
                         int ny,
                         int nz,
                         const float* sigma) {
  if (!d_in_bounds(ix, nx) || !d_in_bounds(iy, ny) || !d_in_bounds(iz, nz)) {
    return -1;
  }
  const int cell = d_flat_index(iz, iy, ix, ny, nx);
  return d_is_conducting(sigma, cell) ? cell : -1;
}

__device__ void d_add_cross_terms(int row,
                                  int cell,
                                  int nbr,
                                  int axis,
                                  float s_xy,
                                  float s_xz,
                                  float s_yz,
                                  int nx,
                                  int ny,
                                  int nz,
                                  double cx,
                                  double cy,
                                  double cz,
                                  const float* sigma,
                                  const int* unknown_index,
                                  const signed char* contact_id,
                                  const double* potentials,
                                  const int* row_off,
                                  const int* col_idx,
                                  double* val,
                                  double& rhs,
                                  double& diag,
                                  int* fail,
                                  bool skew) {
  const int plane = nx * ny;
  const int iz = cell / plane;
  const int rem = cell - iz * plane;
  const int iy = rem / nx;
  const int ix = rem - iy * nx;
  const int niz = nbr / plane;
  const int nrem = nbr - niz * plane;
  const int niy = nrem / nx;
  const int nix = nrem - niy * nx;
  const int sx = (nix > ix) ? 1 : ((nix < ix) ? -1 : 0);
  const int sy = (niy > iy) ? 1 : ((niy < iy) ? -1 : 0);
  const int sz = (niz > iz) ? 1 : ((niz < iz) ? -1 : 0);
  const double area = d_face_area(axis, cx, cy, cz);

  if (axis == 0) {
    if (s_xy != 0.0f) {
      const double fac = -static_cast<double>(sx) * area * static_cast<double>(s_xy) / (4.0 * cy);
      d_add_diff(row, cell, cell, d_cell_at(iz, iy - 1, ix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, nbr, d_cell_at(iz, iy - 1, nix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz, iy + 1, ix, nx, ny, nz, sigma), cell, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz, iy + 1, nix, nx, ny, nz, sigma), nbr, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
    }
    if (s_xz != 0.0f) {
      const double fac = -static_cast<double>(sx) * area * static_cast<double>(s_xz) / (4.0 * cz);
      d_add_diff(row, cell, cell, d_cell_at(iz - 1, iy, ix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, nbr, d_cell_at(iz - 1, iy, nix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz + 1, iy, ix, nx, ny, nz, sigma), cell, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz + 1, iy, nix, nx, ny, nz, sigma), nbr, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
    }
  } else if (axis == 1) {
    if (s_xy != 0.0f) {
      const double fac = -static_cast<double>(sy) * area * static_cast<double>(s_xy) / (4.0 * cx);
      d_add_diff(row, cell, cell, d_cell_at(iz, iy, ix - 1, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, nbr, d_cell_at(iz, niy, ix - 1, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz, iy, ix + 1, nx, ny, nz, sigma), cell, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz, niy, ix + 1, nx, ny, nz, sigma), nbr, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
    }
    if (s_yz != 0.0f) {
      const double fac = -static_cast<double>(sy) * area * static_cast<double>(s_yz) / (4.0 * cz);
      d_add_diff(row, cell, cell, d_cell_at(iz - 1, iy, ix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, nbr, d_cell_at(iz - 1, niy, ix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz + 1, iy, ix, nx, ny, nz, sigma), cell, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz + 1, niy, ix, nx, ny, nz, sigma), nbr, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
    }
  } else {
    if (s_xz != 0.0f) {
      const double fac = -static_cast<double>(sz) * area * static_cast<double>(s_xz) / (4.0 * cx);
      d_add_diff(row, cell, cell, d_cell_at(iz, iy, ix - 1, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, nbr, d_cell_at(niz, iy, ix - 1, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz, iy, ix + 1, nx, ny, nz, sigma), cell, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(niz, iy, ix + 1, nx, ny, nz, sigma), nbr, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
    }
    if (s_yz != 0.0f) {
      const double fac = -static_cast<double>(sz) * area * static_cast<double>(s_yz) / (4.0 * cy);
      d_add_diff(row, cell, cell, d_cell_at(iz, iy - 1, ix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, nbr, d_cell_at(niz, iy - 1, ix, nx, ny, nz, sigma), fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(iz, iy + 1, ix, nx, ny, nz, sigma), cell, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
      d_add_diff(row, cell, d_cell_at(niz, iy + 1, ix, nx, ny, nz, sigma), nbr, fac,
                 unknown_index, contact_id, potentials, row_off, col_idx, val, rhs, diag, fail,
                 skew);
    }
  }
}

__global__ void k_update_transport_matrix_and_rhs(int n,
                                                  int nx,
                                                  int ny,
                                                  int nz,
                                                  int first_r2_layer,
                                                  int n_fm,
                                                  double cx,
                                                  double cy,
                                                  double cz,
                                                  bool amr_enabled,
                                                  double amr_ratio,
                                                  bool ahe_enabled,
                                                  double ahe_ratio,
                                                  const float* magnetization,
                                                  const signed char* region,
                                                  const signed char* contact_id,
                                                  const float* sigma,
                                                  const int* unknown_index,
                                                  const int* unknown_to_cell,
                                                  const int* row_off,
                                                  const int* col_idx,
                                                  const double* potentials,
                                                  double* val,
                                                  double* diag_out,
                                                  double* inv_diag_out,
                                                  double* rhs_out,
                                                  int* fail) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  for (int p = row_off[row]; p < row_off[row + 1]; ++p) {
    val[p] = 0.0;
  }

  const int cell = unknown_to_cell[row];
  const int plane = nx * ny;
  const int iz = cell / plane;
  const int rem = cell - iz * plane;
  const int iy = rem / nx;
  const int ix = rem - iy * nx;
  const SymTensor6D s0 = d_sym_tensor_for_cell(cell, region, sigma, nx, ny, first_r2_layer, n_fm,
                                               amr_enabled, amr_ratio, magnetization);
  const SkewTensor3D k0 = d_skew_tensor_for_cell(cell, region, sigma, nx, ny, first_r2_layer, n_fm,
                                                 ahe_enabled, ahe_ratio, magnetization);

  double rhs = 0.0;
  double diag = 0.0;
  constexpr int offsets[6][4] = {{-1, 0, 0, 0}, {1, 0, 0, 0}, {0, -1, 0, 1},
                                 {0, 1, 0, 1},  {0, 0, -1, 2}, {0, 0, 1, 2}};
  for (int i = 0; i < 6; ++i) {
    const int nix = ix + offsets[i][0];
    const int niy = iy + offsets[i][1];
    const int niz = iz + offsets[i][2];
    const int axis = offsets[i][3];
    // Geometric domain edges only; grid sizes are fixed after prepare_transport_update.
    if (!d_in_bounds(nix, nx) || !d_in_bounds(niy, ny) || !d_in_bounds(niz, nz)) {
      continue;
    }
    const int nbr = d_flat_index(niz, niy, nix, ny, nx);
    if (!d_is_conducting(sigma, nbr)) {
      continue;
    }
    const SymTensor6D s1 = d_sym_tensor_for_cell(nbr, region, sigma, nx, ny, first_r2_layer, n_fm,
                                                 amr_enabled, amr_ratio, magnetization);
    const SkewTensor3D k1 = d_skew_tensor_for_cell(nbr, region, sigma, nx, ny, first_r2_layer, n_fm,
                                                   ahe_enabled, ahe_ratio, magnetization);

    float face_diag = 0.0f;
    if (axis == 0) {
      face_diag = d_avg_diag(s0.xx, s1.xx);
    } else if (axis == 1) {
      face_diag = d_avg_diag(s0.yy, s1.yy);
    } else {
      face_diag = d_avg_diag(s0.zz, s1.zz);
    }
    if (face_diag > 0.0f) {
      const double g = d_face_area(axis, cx, cy, cz) * static_cast<double>(face_diag) /
                       d_axis_spacing(axis, cx, cy, cz);
      d_add_matrix_coeff(row, cell, cell, g, unknown_index, contact_id, potentials, row_off,
                         col_idx, val, rhs, diag, fail, false);
      d_add_matrix_coeff(row, cell, nbr, -g, unknown_index, contact_id, potentials, row_off,
                         col_idx, val, rhs, diag, fail, false);
    }

    const float sym_xy = d_avg_signed(s0.xy, s1.xy);
    const float sym_xz = d_avg_signed(s0.xz, s1.xz);
    const float sym_yz = d_avg_signed(s0.yz, s1.yz);
    d_add_cross_terms(row, cell, nbr, axis, sym_xy, sym_xz, sym_yz, nx, ny, nz, cx, cy, cz,
                      sigma, unknown_index, contact_id, potentials, row_off, col_idx, val, rhs,
                      diag, fail, false);
    if (ahe_enabled) {
      const float skew_xy = d_avg_signed(k0.xy, k1.xy);
      const float skew_xz = d_avg_signed(k0.xz, k1.xz);
      const float skew_yz = d_avg_signed(k0.yz, k1.yz);
      d_add_cross_terms(row, cell, nbr, axis, skew_xy, skew_xz, skew_yz, nx, ny, nz, cx, cy, cz,
                        sigma, unknown_index, contact_id, potentials, row_off, col_idx, val, rhs,
                        diag, fail, true);
    }
  }
  diag_out[row] = diag;
  inv_diag_out[row] = fabs(diag) > DBL_MIN ? 1.0 / diag : 1.0;
  rhs_out[row] = rhs;
}

int threads_per_block() {
  return 256;
}

double device_max_abs(cublasHandle_t h, int n, const double* d_v) {
  if (n <= 0) {
    return 0.0;
  }
  int idx = 0;
  check_cublas(cublasIdamax(h, n, d_v, 1, &idx), "cublasIdamax");
  if (idx <= 0) {
    return 0.0;
  }
  double value = 0.0;
  check_cuda(cudaMemcpy(&value, d_v + (idx - 1), sizeof(double), cudaMemcpyDeviceToHost),
             "cudaMemcpy max abs value");
  return std::fabs(value);
}

std::vector<double> solve_dense(std::vector<double> a, std::vector<double> b, int n) {
  std::vector<double> y(static_cast<std::size_t>(n), 0.0);
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    double pivot_abs = std::fabs(a[static_cast<std::size_t>(k * n + k)]);
    for (int row = k + 1; row < n; ++row) {
      const double v = std::fabs(a[static_cast<std::size_t>(row * n + k)]);
      if (v > pivot_abs) {
        pivot = row;
        pivot_abs = v;
      }
    }
    if (pivot_abs <= DBL_MIN) {
      return y;
    }
    if (pivot != k) {
      for (int col = k; col < n; ++col) {
        std::swap(a[static_cast<std::size_t>(k * n + col)],
                  a[static_cast<std::size_t>(pivot * n + col)]);
      }
      std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot)]);
    }
    for (int row = k + 1; row < n; ++row) {
      const double factor = a[static_cast<std::size_t>(row * n + k)] /
                            a[static_cast<std::size_t>(k * n + k)];
      a[static_cast<std::size_t>(row * n + k)] = 0.0;
      for (int col = k + 1; col < n; ++col) {
        a[static_cast<std::size_t>(row * n + col)] -=
            factor * a[static_cast<std::size_t>(k * n + col)];
      }
      b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(k)];
    }
  }
  for (int row = n - 1; row >= 0; --row) {
    double sum = b[static_cast<std::size_t>(row)];
    for (int col = row + 1; col < n; ++col) {
      sum -= a[static_cast<std::size_t>(row * n + col)] * y[static_cast<std::size_t>(col)];
    }
    const double diag = a[static_cast<std::size_t>(row * n + row)];
    if (std::fabs(diag) > DBL_MIN) {
      y[static_cast<std::size_t>(row)] = sum / diag;
    }
  }
  return y;
}

std::vector<double> solve_least_squares(const std::vector<double>& h,
                                        double beta,
                                        int rows,
                                        int cols) {
  std::vector<double> normal(static_cast<std::size_t>(cols * cols), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(cols), 0.0);
  for (int j = 0; j < cols; ++j) {
    rhs[static_cast<std::size_t>(j)] = beta * h[static_cast<std::size_t>(0 + rows * j)];
    for (int k = 0; k < cols; ++k) {
      double acc = 0.0;
      for (int i = 0; i < rows; ++i) {
        acc += h[static_cast<std::size_t>(i + rows * j)] *
               h[static_cast<std::size_t>(i + rows * k)];
      }
      normal[static_cast<std::size_t>(j * cols + k)] = acc;
    }
  }
  return solve_dense(std::move(normal), std::move(rhs), cols);
}

}  // namespace

PoissonGmresCuda::PoissonGmresCuda(const PoissonWorld& world) : world_(&world) {
  n_ = world.unknown_count();
  cublasHandle_t cublas{};
  check_cublas(cublasCreate(&cublas), "cublasCreate");
  check_cublas(cublasSetStream(cublas, nullptr), "cublasSetStream");
  check_cublas(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST),
               "cublasSetPointerMode HOST");
  handle_cublas_ = cublas;

  cusparseHandle_t cusparse{};
  check_cusparse(cusparseCreate(&cusparse), "cusparseCreate");
  check_cusparse(cusparseSetStream(cusparse, nullptr), "cusparseSetStream");
  handle_cusparse_ = cusparse;
}

void PoissonGmresCuda::destroy_spmv_descriptors() const {
  if (dnvec_x_) {
    cusparseDestroyDnVec(static_cast<cusparseDnVecDescr_t>(dnvec_x_));
    dnvec_x_ = nullptr;
  }
  if (dnvec_y_) {
    cusparseDestroyDnVec(static_cast<cusparseDnVecDescr_t>(dnvec_y_));
    dnvec_y_ = nullptr;
  }
  cudaFree(spmv_buffer_);
  spmv_buffer_ = nullptr;
  spmv_buffer_size_ = 0;
}

PoissonGmresCuda::~PoissonGmresCuda() {
  if (spmat_) {
    cusparseDestroySpMat(static_cast<cusparseSpMatDescr_t>(spmat_));
  }
  destroy_spmv_descriptors();
  cudaFree(d_row_off_);
  cudaFree(d_col_idx_);
  cudaFree(d_val_);
  cudaFree(d_diag_);
  cudaFree(d_inv_diag_);
  cudaFree(d_unknown_index_);
  cudaFree(d_unknown_to_cell_);
  cudaFree(d_region_);
  cudaFree(d_contact_id_);
  cudaFree(d_sigma_);
  cudaFree(d_magnetization_);
  cudaFree(d_contact_potentials_);
  cudaFree(d_update_fail_);
  cudaFree(d_x_);
  cudaFree(d_rhs_);
  cudaFree(d_r_);
  cudaFree(d_z_);
  cudaFree(d_w_);
  cudaFree(d_aw_);
  cudaFree(d_basis_);
  if (handle_cusparse_) {
    cusparseDestroy(static_cast<cusparseHandle_t>(handle_cusparse_));
  }
  if (handle_cublas_) {
    cublasDestroy(static_cast<cublasHandle_t>(handle_cublas_));
  }
}

void PoissonGmresCuda::set_restart(int restart) {
  if (restart < 2) {
    throw std::invalid_argument("GMRES restart must be >= 2");
  }
  if (restart == restart_) {
    return;
  }
  restart_ = restart;
  cudaFree(d_basis_);
  d_basis_ = nullptr;
}

void PoissonGmresCuda::reset_solution() {
  if (d_x_ && n_ > 0) {
    check_cuda(cudaMemset(d_x_, 0, static_cast<std::size_t>(n_) * sizeof(double)),
               "cudaMemset gmres x");
  }
}

std::size_t PoissonGmresCuda::magnetization_device_bytes() const {
  return 3u * static_cast<std::size_t>(fm_layer_count_) * static_cast<std::size_t>(nx_ * ny_) *
         sizeof(float);
}

void PoissonGmresCuda::copy_magnetization_device_to_host(std::vector<float>& out) const {
  if (!d_magnetization_ || !transport_update_ready_) {
    throw std::runtime_error("GMRES magnetization device buffer is not ready");
  }
  const std::size_t n_vals =
      3u * static_cast<std::size_t>(fm_layer_count_) * static_cast<std::size_t>(nx_ * ny_);
  out.resize(n_vals);
  check_cuda(cudaMemcpy(out.data(), d_magnetization_, n_vals * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy gmres magnetization D2H");
}

void PoissonGmresCuda::upload_transport_operator(const PoissonWorld& world) {
  world_ = &world;
  n_ = world.unknown_count();

  const auto& row_off = world.row_offsets();
  const auto& col_idx = world.col_indices();
  const auto& offd = world.offdiag_conductance();
  const auto& diag = world.diagonal();
  const auto& skew_row_off = world.skew_row_offsets();
  const auto& skew_col_idx = world.skew_col_indices();
  const auto& skew_val = world.skew_values();

  std::vector<std::map<int, double>> rows(static_cast<std::size_t>(n_));
  for (int row = 0; row < n_; ++row) {
    rows[static_cast<std::size_t>(row)][row] += static_cast<double>(diag[static_cast<std::size_t>(row)]);
    for (int p = row_off[static_cast<std::size_t>(row)];
         p < row_off[static_cast<std::size_t>(row + 1)]; ++p) {
      rows[static_cast<std::size_t>(row)][col_idx[static_cast<std::size_t>(p)]] -=
          static_cast<double>(offd[static_cast<std::size_t>(p)]);
    }
    if (!skew_row_off.empty()) {
      for (int p = skew_row_off[static_cast<std::size_t>(row)];
           p < skew_row_off[static_cast<std::size_t>(row + 1)]; ++p) {
        rows[static_cast<std::size_t>(row)][skew_col_idx[static_cast<std::size_t>(p)]] +=
            static_cast<double>(skew_val[static_cast<std::size_t>(p)]);
      }
    }
  }

  std::vector<int> h_row(static_cast<std::size_t>(n_ + 1), 0);
  std::vector<int> h_col;
  std::vector<double> h_val;
  for (int row = 0; row < n_; ++row) {
    for (const auto& entry : rows[static_cast<std::size_t>(row)]) {
      if (entry.second == 0.0) {
        continue;
      }
      h_col.push_back(entry.first);
      h_val.push_back(entry.second);
    }
    h_row[static_cast<std::size_t>(row + 1)] = static_cast<int>(h_col.size());
  }
  nnz_ = static_cast<int>(h_col.size());

  std::vector<double> h_diag(static_cast<std::size_t>(n_), 1.0);
  std::vector<double> h_inv_diag(static_cast<std::size_t>(n_), 1.0);
  for (int row = 0; row < n_; ++row) {
    h_diag[static_cast<std::size_t>(row)] = static_cast<double>(diag[static_cast<std::size_t>(row)]);
    h_inv_diag[static_cast<std::size_t>(row)] =
        std::fabs(h_diag[static_cast<std::size_t>(row)]) > DBL_MIN
            ? 1.0 / h_diag[static_cast<std::size_t>(row)]
            : 1.0;
  }

  cudaFree(d_row_off_);
  cudaFree(d_col_idx_);
  cudaFree(d_val_);
  cudaFree(d_diag_);
  cudaFree(d_inv_diag_);
  d_row_off_ = nullptr;
  d_col_idx_ = nullptr;
  d_val_ = nullptr;
  d_diag_ = nullptr;
  d_inv_diag_ = nullptr;

  check_cuda(cudaMalloc(&d_row_off_, (static_cast<std::size_t>(n_) + 1u) * sizeof(int)),
             "cudaMalloc gmres row offsets");
  check_cuda(cudaMalloc(&d_col_idx_, static_cast<std::size_t>(nnz_) * sizeof(int)),
             "cudaMalloc gmres col indices");
  check_cuda(cudaMalloc(&d_val_, static_cast<std::size_t>(nnz_) * sizeof(double)),
             "cudaMalloc gmres values");
  check_cuda(cudaMalloc(&d_diag_, static_cast<std::size_t>(n_) * sizeof(double)),
             "cudaMalloc gmres diag");
  check_cuda(cudaMalloc(&d_inv_diag_, static_cast<std::size_t>(n_) * sizeof(double)),
             "cudaMalloc gmres inv diag");
  check_cuda(cudaMemcpy(d_row_off_, h_row.data(),
                        (static_cast<std::size_t>(n_) + 1u) * sizeof(int),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres row offsets");
  check_cuda(cudaMemcpy(d_col_idx_, h_col.data(), static_cast<std::size_t>(nnz_) * sizeof(int),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres col indices");
  check_cuda(cudaMemcpy(d_val_, h_val.data(), static_cast<std::size_t>(nnz_) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres values");
  check_cuda(cudaMemcpy(d_diag_, h_diag.data(), static_cast<std::size_t>(n_) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres diag");
  check_cuda(cudaMemcpy(d_inv_diag_, h_inv_diag.data(),
                        static_cast<std::size_t>(n_) * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy gmres inv diag");

  if (spmat_) {
    check_cusparse(cusparseDestroySpMat(static_cast<cusparseSpMatDescr_t>(spmat_)),
                   "cusparseDestroySpMat");
    spmat_ = nullptr;
  }
  destroy_spmv_descriptors();
  cusparseSpMatDescr_t mat{};
  check_cusparse(cusparseCreateCsr(&mat, n_, n_, nnz_, d_row_off_, d_col_idx_, d_val_,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                 "cusparseCreateCsr");
  spmat_ = mat;
}

void PoissonGmresCuda::prepare_transport_update(const PoissonWorld& world) {
  world_ = &world;
  if (!d_row_off_ || !d_col_idx_ || !d_val_) {
    throw std::runtime_error("prepare_transport_update requires uploaded GMRES operator pattern");
  }
  n_ = world.unknown_count();
  cell_count_ = world.cell_count();
  nx_ = world.nx();
  ny_ = world.ny();
  nz_ = world.nz();
  first_r2_layer_ = world.first_r2_layer();
  fm_layer_count_ = world.fm_layer_count();
  num_contacts_ = world.num_contacts();
  cx_ = world.cx();
  cy_ = world.cy();
  cz_ = world.cz();
  amr_enabled_ = world.amr_enabled();
  ahe_enabled_ = world.ahe_enabled();
  const TransportConfig& config = world.transport_config();
  amr_ratio_ = config.amr_ratio;
  ahe_ratio_ = config.ahe_ratio;

  // One-time geometry check: region==2 cells must map into the FM magnetization stack.
  const int plane = nx_ * ny_;
  const auto& region = world.region();
  for (int cell = 0; cell < cell_count_; ++cell) {
    if (region[static_cast<std::size_t>(cell)] != 2) {
      continue;
    }
    const int iz = cell / plane;
    const int fm_layer = iz - first_r2_layer_;
    if (fm_layer < 0 || fm_layer >= fm_layer_count_) {
      throw std::runtime_error(
          "prepare_transport_update: region==2 cell outside FM layer range "
          "(first_r2_layer / fm_layer_count inconsistent with region map)");
    }
  }

  cudaFree(d_unknown_index_);
  cudaFree(d_unknown_to_cell_);
  cudaFree(d_region_);
  cudaFree(d_contact_id_);
  cudaFree(d_sigma_);
  cudaFree(d_magnetization_);
  cudaFree(d_contact_potentials_);
  cudaFree(d_update_fail_);
  d_unknown_index_ = nullptr;
  d_unknown_to_cell_ = nullptr;
  d_region_ = nullptr;
  d_contact_id_ = nullptr;
  d_sigma_ = nullptr;
  d_magnetization_ = nullptr;
  d_contact_potentials_ = nullptr;
  d_update_fail_ = nullptr;

  const std::size_t mag_values =
      3u * static_cast<std::size_t>(fm_layer_count_) * static_cast<std::size_t>(nx_ * ny_);

  check_cuda(cudaMalloc(&d_unknown_index_, static_cast<std::size_t>(cell_count_) * sizeof(int)),
             "cudaMalloc gmres unknown index");
  check_cuda(cudaMalloc(&d_unknown_to_cell_, static_cast<std::size_t>(n_) * sizeof(int)),
             "cudaMalloc gmres unknown to cell");
  check_cuda(cudaMalloc(&d_region_, static_cast<std::size_t>(cell_count_) * sizeof(signed char)),
             "cudaMalloc gmres region");
  check_cuda(cudaMalloc(&d_contact_id_, static_cast<std::size_t>(cell_count_) * sizeof(signed char)),
             "cudaMalloc gmres contact id");
  check_cuda(cudaMalloc(&d_sigma_, static_cast<std::size_t>(cell_count_) * sizeof(float)),
             "cudaMalloc gmres sigma");
  check_cuda(cudaMalloc(&d_magnetization_, mag_values * sizeof(float)),
             "cudaMalloc gmres magnetization");
  check_cuda(
      cudaMalloc(&d_contact_potentials_, static_cast<std::size_t>(num_contacts_) * sizeof(double)),
      "cudaMalloc gmres contact potentials");
  check_cuda(cudaMalloc(&d_update_fail_, sizeof(int)), "cudaMalloc gmres update fail");

  check_cuda(cudaMemcpy(d_unknown_index_, world.unknown_index().data(),
                        static_cast<std::size_t>(cell_count_) * sizeof(int),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres unknown index");
  check_cuda(cudaMemcpy(d_unknown_to_cell_, world.unknown_to_cell().data(),
                        static_cast<std::size_t>(n_) * sizeof(int), cudaMemcpyHostToDevice),
             "cudaMemcpy gmres unknown to cell");
  check_cuda(cudaMemcpy(d_region_, world.region().data(),
                        static_cast<std::size_t>(cell_count_) * sizeof(signed char),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres region");
  check_cuda(cudaMemcpy(d_contact_id_, world.contact_id().data(),
                        static_cast<std::size_t>(cell_count_) * sizeof(signed char),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres contact id");
  check_cuda(cudaMemcpy(d_sigma_, world.sigma().data(),
                        static_cast<std::size_t>(cell_count_) * sizeof(float),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres sigma");
  transport_update_ready_ = true;
}

void PoissonGmresCuda::update_transport_operator_and_rhs_device(
    const PoissonWorld& world,
    const std::vector<double>& potentials) {
  if (!transport_update_ready_) {
    throw std::runtime_error("GMRES transport update pattern has not been prepared");
  }
  (void)world;
  ensure_vectors();
  check_cuda(cudaMemcpy(d_contact_potentials_, potentials.data(),
                        static_cast<std::size_t>(num_contacts_) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres contact potentials");
  check_cuda(cudaMemset(d_update_fail_, 0, sizeof(int)), "cudaMemset gmres update fail");

  const int blocks = (n_ + threads_per_block() - 1) / threads_per_block();
  k_update_transport_matrix_and_rhs<<<blocks, threads_per_block()>>>(
      n_, nx_, ny_, nz_, first_r2_layer_, fm_layer_count_, cx_, cy_, cz_, amr_enabled_,
      amr_ratio_, ahe_enabled_, ahe_ratio_, d_magnetization_, d_region_, d_contact_id_, d_sigma_,
      d_unknown_index_, d_unknown_to_cell_, d_row_off_, d_col_idx_, d_contact_potentials_, d_val_,
      d_diag_, d_inv_diag_, d_rhs_, d_update_fail_);
  check_cuda(cudaGetLastError(), "k_update_transport_matrix_and_rhs launch");
  int failed = 0;
  check_cuda(cudaMemcpy(&failed, d_update_fail_, sizeof(int), cudaMemcpyDeviceToHost),
             "cudaMemcpy gmres update fail");
  if (failed != 0) {
    throw std::runtime_error(
        "GMRES transport update encountered an entry outside the fixed CSR pattern");
  }
}

void PoissonGmresCuda::update_transport_operator_and_rhs_device(
    const PoissonWorld& world,
    const std::vector<float>& magnetization_fm_stack,
    const std::vector<double>& potentials) {
  if (!transport_update_ready_) {
    throw std::runtime_error("GMRES transport update pattern has not been prepared");
  }
  const std::size_t expected = 3u * static_cast<std::size_t>(fm_layer_count_) *
                               static_cast<std::size_t>(nx_ * ny_);
  if (magnetization_fm_stack.size() != expected) {
    throw std::runtime_error("GMRES transport update magnetization stack size mismatch");
  }
  check_cuda(cudaMemcpy(d_magnetization_, magnetization_fm_stack.data(),
                        expected * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy gmres magnetization");
  update_transport_operator_and_rhs_device(world, potentials);
}

void PoissonGmresCuda::ensure_vectors() const {
  if (n_ <= 0) {
    return;
  }
  const std::size_t nbytes = static_cast<std::size_t>(n_) * sizeof(double);
  if (!d_x_) {
    check_cuda(cudaMalloc(&d_x_, nbytes), "cudaMalloc gmres x");
    check_cuda(cudaMemset(d_x_, 0, nbytes), "cudaMemset gmres x");
  }
  if (!d_rhs_) {
    check_cuda(cudaMalloc(&d_rhs_, nbytes), "cudaMalloc gmres rhs");
    check_cuda(cudaMalloc(&d_r_, nbytes), "cudaMalloc gmres r");
    check_cuda(cudaMalloc(&d_z_, nbytes), "cudaMalloc gmres z");
    check_cuda(cudaMalloc(&d_w_, nbytes), "cudaMalloc gmres w");
    check_cuda(cudaMalloc(&d_aw_, nbytes), "cudaMalloc gmres aw");
  }
  if (!d_basis_) {
    check_cuda(cudaMalloc(&d_basis_,
                          static_cast<std::size_t>(restart_ + 1) * static_cast<std::size_t>(n_) *
                              sizeof(double)),
               "cudaMalloc gmres basis");
  }
}

void PoissonGmresCuda::ensure_spmv_descriptors() const {
  ensure_vectors();
  if (!dnvec_x_) {
    cusparseDnVecDescr_t xvec{};
    cusparseDnVecDescr_t yvec{};
    check_cusparse(cusparseCreateDnVec(&xvec, n_, d_x_, CUDA_R_64F), "cusparseCreateDnVec x");
    check_cusparse(cusparseCreateDnVec(&yvec, n_, d_aw_, CUDA_R_64F), "cusparseCreateDnVec y");
    dnvec_x_ = xvec;
    dnvec_y_ = yvec;
  }
  if (spmv_buffer_) {
    return;
  }
  const double alpha = 1.0;
  const double beta = 0.0;
  check_cusparse(
      cusparseSpMV_bufferSize(static_cast<cusparseHandle_t>(handle_cusparse_),
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              static_cast<cusparseSpMatDescr_t>(spmat_),
                              static_cast<cusparseDnVecDescr_t>(dnvec_x_), &beta,
                              static_cast<cusparseDnVecDescr_t>(dnvec_y_), CUDA_R_64F,
                              CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size_),
      "cusparseSpMV_bufferSize");
  check_cuda(cudaMalloc(&spmv_buffer_, spmv_buffer_size_), "cudaMalloc cusparse SpMV buffer");
}

void PoissonGmresCuda::spmv(const double* d_x, double* d_y) const {
  ensure_spmv_descriptors();
  check_cusparse(
      cusparseDnVecSetValues(static_cast<cusparseDnVecDescr_t>(dnvec_x_),
                             const_cast<double*>(d_x)),
      "cusparseDnVecSetValues x");
  check_cusparse(cusparseDnVecSetValues(static_cast<cusparseDnVecDescr_t>(dnvec_y_), d_y),
                 "cusparseDnVecSetValues y");
  const double alpha = 1.0;
  const double beta = 0.0;
  check_cusparse(cusparseSpMV(static_cast<cusparseHandle_t>(handle_cusparse_),
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              static_cast<cusparseSpMatDescr_t>(spmat_),
                              static_cast<cusparseDnVecDescr_t>(dnvec_x_), &beta,
                              static_cast<cusparseDnVecDescr_t>(dnvec_y_), CUDA_R_64F,
                              CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_),
                 "cusparseSpMV");
}

void PoissonGmresCuda::copy_solution_to_host(std::vector<double>& x) const {
  if (static_cast<int>(x.size()) != n_) {
    x.assign(static_cast<std::size_t>(n_), 0.0);
  }
  check_cuda(cudaMemcpy(x.data(), d_x_, static_cast<std::size_t>(n_) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy gmres x out");
}

double* PoissonGmresCuda::basis_vector(int index) const {
  return d_basis_ + static_cast<std::size_t>(index) * static_cast<std::size_t>(n_);
}

PcgResult PoissonGmresCuda::solve_device_rhs(std::vector<double>& x) const {
  PcgResult result;
  if (n_ == 0) {
    result.converged = true;
    return result;
  }
  ensure_vectors();
  if (x.size() != static_cast<std::size_t>(n_)) {
    x.assign(static_cast<std::size_t>(n_), 0.0);
    check_cuda(cudaMemset(d_x_, 0, static_cast<std::size_t>(n_) * sizeof(double)),
               "cudaMemset gmres x");
  }

  cublasHandle_t h = static_cast<cublasHandle_t>(handle_cublas_);
  const int blocks = (n_ + threads_per_block() - 1) / threads_per_block();

  // Default-stream cuBLAS host-mode reductions order kernels; no explicit sync needed.
  spmv(d_x_, d_aw_);
  k_residual<<<blocks, threads_per_block()>>>(d_r_, d_rhs_, d_aw_, n_);

  const double rhs_linf = device_max_abs(h, n_, d_rhs_);
  const double r0_linf = device_max_abs(h, n_, d_r_);
  result.rhs_inf_norm = rhs_linf;
  result.initial_residual_max_norm = r0_linf;
  result.residual_max_norm = r0_linf;
  result.residual_relative = (rhs_linf > 0.0) ? (r0_linf / rhs_linf) : 0.0;
  if (residual_meets_tolerance(result.residual_max_norm, rhs_linf, tolerance_)) {
    result.converged = true;
    copy_solution_to_host(x);
    return result;
  }

  const int rows_h = restart_ + 1;
  std::vector<double> hessenberg(static_cast<std::size_t>(rows_h * restart_), 0.0);

  int total_iters = 0;
  while (total_iters < max_iterations_) {
    k_jacobi<<<blocks, threads_per_block()>>>(d_z_, d_r_, d_inv_diag_, n_);
    double beta = 0.0;
    check_cublas(cublasDnrm2(h, n_, d_z_, 1, &beta), "cublasDnrm2 gmres beta");
    if (!(beta > 0.0) || !std::isfinite(beta)) {
      result.numerical_failure = true;
      break;
    }

    check_cublas(cublasDcopy(h, n_, d_z_, 1, basis_vector(0), 1), "cublasDcopy gmres v0");
    const double inv_beta = 1.0 / beta;
    check_cublas(cublasDscal(h, n_, &inv_beta, basis_vector(0), 1), "cublasDscal gmres v0");

    std::fill(hessenberg.begin(), hessenberg.end(), 0.0);

    int inner_used = 0;
    for (int j = 0; j < restart_ && total_iters < max_iterations_; ++j) {
      spmv(basis_vector(j), d_aw_);
      k_jacobi<<<blocks, threads_per_block()>>>(d_w_, d_aw_, d_inv_diag_, n_);

      for (int i = 0; i <= j; ++i) {
        double hij = 0.0;
        check_cublas(cublasDdot(h, n_, d_w_, 1, basis_vector(i), 1, &hij),
                     "cublasDdot gmres h");
        hessenberg[static_cast<std::size_t>(i + rows_h * j)] = hij;
        const double neg_hij = -hij;
        check_cublas(cublasDaxpy(h, n_, &neg_hij, basis_vector(i), 1, d_w_, 1),
                     "cublasDaxpy gmres orthogonalize");
      }

      double hnext = 0.0;
      check_cublas(cublasDnrm2(h, n_, d_w_, 1, &hnext), "cublasDnrm2 gmres hnext");
      hessenberg[static_cast<std::size_t>((j + 1) + rows_h * j)] = hnext;
      if (hnext > DBL_MIN) {
        check_cublas(cublasDcopy(h, n_, d_w_, 1, basis_vector(j + 1), 1),
                     "cublasDcopy gmres vnext");
        const double inv_hnext = 1.0 / hnext;
        check_cublas(cublasDscal(h, n_, &inv_hnext, basis_vector(j + 1), 1),
                     "cublasDscal gmres vnext");
      }

      ++total_iters;
      ++inner_used;
      result.iterations = total_iters;

      if (hnext <= DBL_MIN) {
        break;
      }
    }

    const std::vector<double> y = solve_least_squares(hessenberg, beta, rows_h, inner_used);
    for (int i = 0; i < inner_used; ++i) {
      const double yi = y[static_cast<std::size_t>(i)];
      check_cublas(cublasDaxpy(h, n_, &yi, basis_vector(i), 1, d_x_, 1),
                   "cublasDaxpy gmres update x");
    }

    spmv(d_x_, d_aw_);
    k_residual<<<blocks, threads_per_block()>>>(d_r_, d_rhs_, d_aw_, n_);
    result.residual_max_norm = device_max_abs(h, n_, d_r_);
    result.residual_relative =
        (rhs_linf > 0.0) ? (result.residual_max_norm / rhs_linf) : 0.0;
    if (residual_meets_tolerance(result.residual_max_norm, rhs_linf, tolerance_)) {
      result.converged = true;
      break;
    }
  }

  result.converged = residual_meets_tolerance(result.residual_max_norm, rhs_linf, tolerance_);
  if (!result.converged && result.iterations >= max_iterations_) {
    result.numerical_failure = false;
  }
  copy_solution_to_host(x);
  return result;
}

PcgResult PoissonGmresCuda::solve(const std::vector<double>& rhs, std::vector<double>& x) const {
  if (static_cast<int>(rhs.size()) != n_) {
    throw std::runtime_error("GMRES rhs size does not match operator size");
  }
  if (n_ == 0) {
    PcgResult result;
    result.converged = true;
    return result;
  }
  ensure_vectors();
  check_cuda(cudaMemcpy(d_rhs_, rhs.data(), static_cast<std::size_t>(n_) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy gmres rhs");
  return solve_device_rhs(x);
}

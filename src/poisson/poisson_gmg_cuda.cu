#include "poisson_gmg_cuda.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "poisson_world.hpp"

namespace {

constexpr double kAtol = 1e-30;
constexpr int kMinXySide = 8;
constexpr int kDefaultCoarsestRestart = 200;
constexpr int kCoarseJacobiSweeps = 8;
constexpr double kCoarseTol = 1e-2;

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

int threads_per_block() { return 256; }

int blocks_for(int n) { return (n + threads_per_block() - 1) / threads_per_block(); }

int flat_index(int iz, int iy, int ix, int ny, int nx) { return (iz * ny + iy) * nx + ix; }

double iso_face_area(double cx, double cy, double cz, int axis) {
  if (axis == 0) {
    return cy * cz;
  }
  if (axis == 1) {
    return cx * cz;
  }
  return cx * cy;
}

double iso_axis_spacing(double cx, double cy, double cz, int axis) {
  if (axis == 0) {
    return cx;
  }
  if (axis == 1) {
    return cy;
  }
  return cz;
}

struct HostLevel {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int n = 0;
  int nnz = 0;
  std::vector<int> unknown_index;
  std::vector<int> unknown_to_cell;
  std::vector<signed char> cell_kind;  // 0 void, 1 Dirichlet, 2 unknown
  std::vector<int> row_off;
  std::vector<int> col_idx;
  std::vector<double> val;
  std::vector<int> parent;
  std::vector<int> scatter_to;
};

constexpr signed char kKindVoid = 0;
constexpr signed char kKindDirichlet = 1;
constexpr signed char kKindUnknown = 2;

bool residual_meets(double r_inf, double rhs_inf, double rtol) {
  return r_inf <= kAtol + rtol * rhs_inf;
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
      const double factor =
          a[static_cast<std::size_t>(row * n + k)] / a[static_cast<std::size_t>(k * n + k)];
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

double device_max_abs(cublasHandle_t h, int n, const double* d_v) {
  if (n <= 0) {
    return 0.0;
  }
  int idx = 0;
  check_cublas(cublasIdamax(h, n, d_v, 1, &idx), "cublasIdamax gmg");
  if (idx <= 0) {
    return 0.0;
  }
  double value = 0.0;
  check_cuda(cudaMemcpy(&value, d_v + (idx - 1), sizeof(double), cudaMemcpyDeviceToHost),
             "cudaMemcpy gmg max abs");
  return std::fabs(value);
}

__global__ void k_csr_spmv(int n,
                           const int* row_off,
                           const int* col_idx,
                           const double* val,
                           const double* x,
                           double* y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  double acc = 0.0;
#pragma unroll 8
  for (int p = row_off[row]; p < row_off[row + 1]; ++p) {
    acc += val[p] * x[col_idx[p]];
  }
  y[row] = acc;
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

__global__ void k_scale_jacobi(double* z,
                               const double* r,
                               const double* inv_diag,
                               double omega,
                               int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = omega * inv_diag[i] * r[i];
  }
}

__global__ void k_damped_jacobi(double* z,
                                const double* r,
                                const double* az,
                                const double* inv_diag,
                                double omega,
                                int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] += omega * inv_diag[i] * (r[i] - az[i]);
  }
}

__global__ void k_restrict(int n_fine, const int* parent, const double* r_fine, double* r_coarse) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_fine) {
    return;
  }
  const int p = parent[i];
  if (p >= 0) {
    atomicAdd(r_coarse + p, r_fine[i]);
  }
}

__global__ void k_prolong(int n_fine, const int* parent, const double* e_coarse, double* z_fine) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_fine) {
    return;
  }
  const int p = parent[i];
  if (p >= 0) {
    z_fine[i] += e_coarse[p];
  }
}

__global__ void k_scatter_add(int nnz, const int* scatter_to, const double* fine, double* coarse) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nnz) {
    return;
  }
  const int s = scatter_to[k];
  if (s >= 0) {
    atomicAdd(coarse + s, fine[k]);
  }
}

__global__ void k_extract_diag(int n,
                               const int* row_off,
                               const int* col_idx,
                               const double* val,
                               double* diag,
                               double* inv_diag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  double d = 0.0;
  for (int p = row_off[row]; p < row_off[row + 1]; ++p) {
    if (col_idx[p] == row) {
      d = val[p];
      break;
    }
  }
  diag[row] = d;
  inv_diag[row] = fabs(d) > DBL_MIN ? 1.0 / d : 1.0;
}

void flatten_sorted_row(std::vector<std::pair<int, double>>& entries,
                        std::vector<int>& col_idx,
                        std::vector<double>& val) {
  const std::size_t start = col_idx.size();
  std::sort(entries.begin(), entries.end(),
            [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
              return a.first < b.first;
            });
  for (const auto& e : entries) {
    if (col_idx.size() > start && col_idx.back() == e.first) {
      val.back() += e.second;
      continue;
    }
    col_idx.push_back(e.first);
    val.push_back(e.second);
  }
}

void build_iso_seven_point(const PoissonWorld& world, HostLevel& fine) {
  fine.nx = world.nx();
  fine.ny = world.ny();
  fine.nz = world.nz();
  fine.unknown_index = world.unknown_index();
  fine.unknown_to_cell = world.unknown_to_cell();
  fine.n = world.unknown_count();
  fine.cell_kind.assign(static_cast<std::size_t>(world.cell_count()), kKindVoid);
  const auto& contact_id = world.contact_id();
  for (int cell = 0; cell < world.cell_count(); ++cell) {
    if (!world.is_conducting(cell)) {
      continue;
    }
    if (contact_id[static_cast<std::size_t>(cell)] != 0) {
      fine.cell_kind[static_cast<std::size_t>(cell)] = kKindDirichlet;
    } else {
      fine.cell_kind[static_cast<std::size_t>(cell)] = kKindUnknown;
    }
  }

  const double cx = world.cx();
  const double cy = world.cy();
  const double cz = world.cz();
  const auto& sigma = world.sigma();
  const int plane = fine.nx * fine.ny;
  constexpr std::array<std::array<int, 4>, 6> kNeighbors = {{
      {{-1, 0, 0, 0}}, {{1, 0, 0, 0}}, {{0, -1, 0, 1}},
      {{0, 1, 0, 1}},  {{0, 0, -1, 2}}, {{0, 0, 1, 2}},
  }};

  fine.row_off.assign(static_cast<std::size_t>(fine.n) + 1u, 0);
  fine.col_idx.clear();
  fine.val.clear();
  fine.col_idx.reserve(static_cast<std::size_t>(std::max(fine.n, 0)) * 7u);
  fine.val.reserve(static_cast<std::size_t>(std::max(fine.n, 0)) * 7u);

  for (int row = 0; row < fine.n; ++row) {
    const int cell = fine.unknown_to_cell[static_cast<std::size_t>(row)];
    const int iz = cell / plane;
    const int rem = cell - iz * plane;
    const int iy = rem / fine.nx;
    const int ix = rem - iy * fine.nx;
    const float s0 = sigma[static_cast<std::size_t>(cell)];
    double diag = 0.0;
    std::vector<std::pair<int, double>> entries;
    entries.reserve(7);
    for (const auto& neighbor : kNeighbors) {
      const int nix = ix + neighbor[0];
      const int niy = iy + neighbor[1];
      const int niz = iz + neighbor[2];
      const int axis = neighbor[3];
      if (nix < 0 || nix >= fine.nx || niy < 0 || niy >= fine.ny || niz < 0 || niz >= fine.nz) {
        continue;
      }
      const int nbr = flat_index(niz, niy, nix, fine.ny, fine.nx);
      if (!world.is_conducting(nbr)) {
        continue;
      }
      const float s1 = sigma[static_cast<std::size_t>(nbr)];
      if (!(s0 > 0.0f) || !(s1 > 0.0f)) {
        continue;
      }
      const double g = iso_face_area(cx, cy, cz, axis) *
                       std::sqrt(static_cast<double>(s0) * static_cast<double>(s1)) /
                       iso_axis_spacing(cx, cy, cz, axis);
      if (!(g > 0.0)) {
        continue;
      }
      diag += g;
      const int col = fine.unknown_index[static_cast<std::size_t>(nbr)];
      if (col >= 0) {
        entries.emplace_back(col, -g);
      }
    }
    entries.emplace_back(row, diag);
    flatten_sorted_row(entries, fine.col_idx, fine.val);
    fine.row_off[static_cast<std::size_t>(row + 1)] = static_cast<int>(fine.col_idx.size());
  }
  fine.nnz = static_cast<int>(fine.col_idx.size());
}

void classify_xy_level(const HostLevel& fine, HostLevel& coarse) {
  coarse.nx = (fine.nx + 1) / 2;
  coarse.ny = (fine.ny + 1) / 2;
  coarse.nz = fine.nz;
  const int cell_count = coarse.nx * coarse.ny * coarse.nz;
  coarse.unknown_index.assign(static_cast<std::size_t>(cell_count), -1);
  coarse.cell_kind.assign(static_cast<std::size_t>(cell_count), kKindVoid);
  coarse.unknown_to_cell.clear();
  coarse.unknown_to_cell.reserve(static_cast<std::size_t>(cell_count));

  auto child_cell = [&](int iz, int iy_c, int ix_c, int dy, int dx) -> int {
    const int iy = 2 * iy_c + dy;
    const int ix = 2 * ix_c + dx;
    if (ix < 0 || ix >= fine.nx || iy < 0 || iy >= fine.ny || iz < 0 || iz >= fine.nz) {
      return -1;
    }
    return flat_index(iz, iy, ix, fine.ny, fine.nx);
  };

  for (int iz = 0; iz < coarse.nz; ++iz) {
    for (int iy = 0; iy < coarse.ny; ++iy) {
      for (int ix = 0; ix < coarse.nx; ++ix) {
        const int ccell = flat_index(iz, iy, ix, coarse.ny, coarse.nx);
        int n_child = 0;
        int n_void = 0;
        int n_dirichlet = 0;
        int n_unknown = 0;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const int fcell = child_cell(iz, iy, ix, dy, dx);
            if (fcell < 0) {
              continue;
            }
            ++n_child;
            const signed char kind = fine.cell_kind[static_cast<std::size_t>(fcell)];
            if (kind == kKindVoid) {
              ++n_void;
            } else if (kind == kKindDirichlet) {
              ++n_dirichlet;
            } else {
              ++n_unknown;
            }
          }
        }
        if (n_child == 0 || n_void == n_child) {
          continue;
        }
        if (n_dirichlet > 0) {
          coarse.cell_kind[static_cast<std::size_t>(ccell)] = kKindDirichlet;
          continue;
        }
        if (n_unknown == 0) {
          continue;
        }
        coarse.cell_kind[static_cast<std::size_t>(ccell)] = kKindUnknown;
        coarse.unknown_index[static_cast<std::size_t>(ccell)] =
            static_cast<int>(coarse.unknown_to_cell.size());
        coarse.unknown_to_cell.push_back(ccell);
      }
    }
  }
  coarse.n = static_cast<int>(coarse.unknown_to_cell.size());
  coarse.parent.assign(static_cast<std::size_t>(fine.n), -1);
  for (int i = 0; i < fine.n; ++i) {
    const int fcell = fine.unknown_to_cell[static_cast<std::size_t>(i)];
    const int plane = fine.nx * fine.ny;
    const int iz = fcell / plane;
    const int rem = fcell - iz * plane;
    const int iy = rem / fine.nx;
    const int ix = rem - iy * fine.nx;
    const int ccell = flat_index(iz, iy / 2, ix / 2, coarse.ny, coarse.nx);
    coarse.parent[static_cast<std::size_t>(i)] =
        coarse.unknown_index[static_cast<std::size_t>(ccell)];
  }
}

void classify_z_level(const HostLevel& fine, HostLevel& coarse) {
  coarse.nx = fine.nx;
  coarse.ny = fine.ny;
  coarse.nz = (fine.nz + 1) / 2;
  const int cell_count = coarse.nx * coarse.ny * coarse.nz;
  coarse.unknown_index.assign(static_cast<std::size_t>(cell_count), -1);
  coarse.cell_kind.assign(static_cast<std::size_t>(cell_count), kKindVoid);
  coarse.unknown_to_cell.clear();
  coarse.unknown_to_cell.reserve(static_cast<std::size_t>(cell_count));

  auto child_cell = [&](int iz_c, int iy, int ix, int dz) -> int {
    const int iz = 2 * iz_c + dz;
    if (ix < 0 || ix >= fine.nx || iy < 0 || iy >= fine.ny || iz < 0 || iz >= fine.nz) {
      return -1;
    }
    return flat_index(iz, iy, ix, fine.ny, fine.nx);
  };

  for (int iz = 0; iz < coarse.nz; ++iz) {
    for (int iy = 0; iy < coarse.ny; ++iy) {
      for (int ix = 0; ix < coarse.nx; ++ix) {
        const int ccell = flat_index(iz, iy, ix, coarse.ny, coarse.nx);
        int n_child = 0;
        int n_void = 0;
        int n_dirichlet = 0;
        int n_unknown = 0;
        for (int dz = 0; dz < 2; ++dz) {
          const int fcell = child_cell(iz, iy, ix, dz);
          if (fcell < 0) {
            continue;
          }
          ++n_child;
          const signed char kind = fine.cell_kind[static_cast<std::size_t>(fcell)];
          if (kind == kKindVoid) {
            ++n_void;
          } else if (kind == kKindDirichlet) {
            ++n_dirichlet;
          } else {
            ++n_unknown;
          }
        }
        if (n_child == 0 || n_void == n_child) {
          continue;
        }
        if (n_dirichlet > 0) {
          coarse.cell_kind[static_cast<std::size_t>(ccell)] = kKindDirichlet;
          continue;
        }
        if (n_unknown == 0) {
          continue;
        }
        coarse.cell_kind[static_cast<std::size_t>(ccell)] = kKindUnknown;
        coarse.unknown_index[static_cast<std::size_t>(ccell)] =
            static_cast<int>(coarse.unknown_to_cell.size());
        coarse.unknown_to_cell.push_back(ccell);
      }
    }
  }
  coarse.n = static_cast<int>(coarse.unknown_to_cell.size());
  coarse.parent.assign(static_cast<std::size_t>(fine.n), -1);
  for (int i = 0; i < fine.n; ++i) {
    const int fcell = fine.unknown_to_cell[static_cast<std::size_t>(i)];
    const int plane = fine.nx * fine.ny;
    const int iz = fcell / plane;
    const int rem = fcell - iz * plane;
    const int iy = rem / fine.nx;
    const int ix = rem - iy * fine.nx;
    const int ccell = flat_index(iz / 2, iy, ix, coarse.ny, coarse.nx);
    coarse.parent[static_cast<std::size_t>(i)] =
        coarse.unknown_index[static_cast<std::size_t>(ccell)];
  }
}

bool build_galerkin_pattern(const HostLevel& fine, HostLevel& coarse, bool& void_ok) {
  if (fine.n <= 0 || coarse.n <= 0) {
    coarse.row_off.assign(1, 0);
    coarse.col_idx.clear();
    coarse.scatter_to.assign(static_cast<std::size_t>(fine.nnz), -1);
    coarse.nnz = 0;
    return true;
  }

  std::vector<std::uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(fine.nnz) + static_cast<std::size_t>(coarse.n));
  auto pack = [](int r, int c) -> std::uint64_t {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(r)) << 32u) |
           static_cast<std::uint32_t>(c);
  };
  auto unpack_r = [](std::uint64_t k) -> int { return static_cast<int>(k >> 32u); };
  auto unpack_c = [](std::uint64_t k) -> int { return static_cast<int>(k & 0xffffffffu); };

  for (int i = 0; i < coarse.n; ++i) {
    keys.push_back(pack(i, i));
  }
  coarse.scatter_to.assign(static_cast<std::size_t>(fine.nnz), -1);
  for (int row = 0; row < fine.n; ++row) {
    const int pi = fine.parent[static_cast<std::size_t>(row)];
    for (int p = fine.row_off[static_cast<std::size_t>(row)];
         p < fine.row_off[static_cast<std::size_t>(row + 1)]; ++p) {
      if (pi < 0) {
        continue;
      }
      const int col = fine.col_idx[static_cast<std::size_t>(p)];
      const int pj = fine.parent[static_cast<std::size_t>(col)];
      if (pj < 0) {
        continue;
      }
      keys.push_back(pack(pi, pj));
    }
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  coarse.nnz = static_cast<int>(keys.size());
  coarse.row_off.assign(static_cast<std::size_t>(coarse.n) + 1u, 0);
  coarse.col_idx.resize(keys.size());
  int row_cursor = 0;
  for (int k = 0; k < coarse.nnz; ++k) {
    const int r = unpack_r(keys[static_cast<std::size_t>(k)]);
    const int c = unpack_c(keys[static_cast<std::size_t>(k)]);
    while (row_cursor < r) {
      ++row_cursor;
      coarse.row_off[static_cast<std::size_t>(row_cursor)] = k;
    }
    coarse.col_idx[static_cast<std::size_t>(k)] = c;
  }
  while (row_cursor < coarse.n) {
    ++row_cursor;
    coarse.row_off[static_cast<std::size_t>(row_cursor)] = coarse.nnz;
  }

  auto slot_of = [&](int r, int c) -> int {
    const std::uint64_t key = pack(r, c);
    const auto it = std::lower_bound(keys.begin(), keys.end(), key);
    if (it == keys.end() || *it != key) {
      return -1;
    }
    return static_cast<int>(it - keys.begin());
  };

  std::vector<char> coarse_has_fine(static_cast<std::size_t>(coarse.nnz), 0);
  for (int row = 0; row < fine.n; ++row) {
    const int pi = fine.parent[static_cast<std::size_t>(row)];
    for (int p = fine.row_off[static_cast<std::size_t>(row)];
         p < fine.row_off[static_cast<std::size_t>(row + 1)]; ++p) {
      if (pi < 0) {
        continue;
      }
      const int col = fine.col_idx[static_cast<std::size_t>(p)];
      const int pj = fine.parent[static_cast<std::size_t>(col)];
      if (pj < 0) {
        continue;
      }
      const int slot = slot_of(pi, pj);
      coarse.scatter_to[static_cast<std::size_t>(p)] = slot;
      if (slot >= 0) {
        coarse_has_fine[static_cast<std::size_t>(slot)] = 1;
      }
    }
  }

  void_ok = true;
  for (int row = 0; row < coarse.n && void_ok; ++row) {
    for (int p = coarse.row_off[static_cast<std::size_t>(row)];
         p < coarse.row_off[static_cast<std::size_t>(row + 1)]; ++p) {
      const int col = coarse.col_idx[static_cast<std::size_t>(p)];
      if (col == row) {
        continue;
      }
      if (!coarse_has_fine[static_cast<std::size_t>(p)]) {
        void_ok = false;
        break;
      }
    }
  }
  return void_ok;
}

bool should_coarsen_xy(const HostLevel& level) {
  return std::min(level.nx, level.ny) > kMinXySide && level.n > 64;
}

bool should_coarsen_z(const HostLevel& level) {
  return level.nz > 2 && level.n > 64;
}

}  // namespace

PoissonGmgCuda::PoissonGmgCuda() {
  cublasHandle_t h{};
  check_cublas(cublasCreate(&h), "cublasCreate gmg");
  check_cublas(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode gmg");
  handle_cublas_ = h;
  check_cuda(cudaMalloc(&d_scalar_, sizeof(double)), "cudaMalloc gmg scalar");
}

void PoissonGmgCuda::destroy_level(Level& level) {
  if (level.owns_pattern) {
    cudaFree(level.d_row_off);
    cudaFree(level.d_col_idx);
  }
  cudaFree(level.d_val);
  cudaFree(level.d_diag);
  cudaFree(level.d_inv_diag);
  cudaFree(level.d_parent);
  cudaFree(level.d_scatter_to);
  cudaFree(level.d_r);
  cudaFree(level.d_z);
  cudaFree(level.d_az);
  cudaFree(level.d_tmp);
  cudaFree(level.d_basis);
  cudaFree(level.d_w);
  level = Level{};
}

PoissonGmgCuda::~PoissonGmgCuda() {
  for (Level& level : levels_) {
    destroy_level(level);
  }
  cudaFree(d_scalar_);
  if (handle_cublas_) {
    cublasDestroy(static_cast<cublasHandle_t>(handle_cublas_));
  }
}

void PoissonGmgCuda::build(const PoissonWorld& world) {
  for (Level& level : levels_) {
    destroy_level(level);
  }
  levels_.clear();
  void_sparsity_ok_ = false;

  HostLevel fine;
  build_iso_seven_point(world, fine);

  std::vector<HostLevel> host_levels;
  host_levels.push_back(std::move(fine));

  while (host_levels.size() < 16u) {
    const HostLevel& current = host_levels.back();
    HostLevel next;
    if (should_coarsen_xy(current)) {
      classify_xy_level(current, next);
    } else if (should_coarsen_z(current)) {
      classify_z_level(current, next);
    } else {
      break;
    }
    if (next.n <= 0 || next.n >= current.n) {
      break;
    }
    bool ok = true;
    HostLevel& finer = host_levels.back();
    finer.parent = next.parent;
    if (!build_galerkin_pattern(finer, next, ok)) {
      throw std::runtime_error(
          "PoissonGmgCuda: Galerkin coarse entry without a fine conducting path (void fill)");
    }
    void_sparsity_ok_ = ok;
    finer.scatter_to = std::move(next.scatter_to);
    next.parent.clear();
    host_levels.push_back(std::move(next));
  }
  if (host_levels.size() == 1u) {
    void_sparsity_ok_ = true;
  }

  levels_.resize(host_levels.size());
  for (std::size_t ell = 0; ell < host_levels.size(); ++ell) {
    const HostLevel& host = host_levels[ell];
    Level& level = levels_[ell];
    level.n = host.n;
    level.nnz = host.nnz;
    level.nx = host.nx;
    level.ny = host.ny;
    level.nz = host.nz;
    level.owns_pattern = true;
    const std::size_t n_bytes = static_cast<std::size_t>(level.n) * sizeof(double);
    const std::size_t nnz_bytes = static_cast<std::size_t>(level.nnz) * sizeof(double);
    check_cuda(cudaMalloc(&level.d_row_off, (static_cast<std::size_t>(level.n) + 1u) * sizeof(int)),
               "cudaMalloc gmg row offsets");
    check_cuda(cudaMalloc(&level.d_col_idx, static_cast<std::size_t>(std::max(level.nnz, 1)) * sizeof(int)),
               "cudaMalloc gmg col indices");
    check_cuda(cudaMemcpy(level.d_row_off, host.row_off.data(),
                          (static_cast<std::size_t>(level.n) + 1u) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy gmg row offsets");
    if (level.nnz > 0) {
      check_cuda(cudaMemcpy(level.d_col_idx, host.col_idx.data(),
                            static_cast<std::size_t>(level.nnz) * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy gmg col indices");
    }
    check_cuda(cudaMalloc(&level.d_val, nnz_bytes > 0 ? nnz_bytes : sizeof(double)),
               "cudaMalloc gmg val");
    check_cuda(cudaMalloc(&level.d_diag, n_bytes > 0 ? n_bytes : sizeof(double)),
               "cudaMalloc gmg diag");
    check_cuda(cudaMalloc(&level.d_inv_diag, n_bytes > 0 ? n_bytes : sizeof(double)),
               "cudaMalloc gmg inv diag");
    if (ell == 0 && level.nnz > 0) {
      check_cuda(cudaMemcpy(level.d_val, host.val.data(), nnz_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy gmg fine A_iso");
    } else if (level.nnz > 0) {
      check_cuda(cudaMemset(level.d_val, 0, nnz_bytes), "cudaMemset gmg coarse val");
    }
    if (ell + 1u < host_levels.size()) {
      check_cuda(cudaMalloc(&level.d_parent, static_cast<std::size_t>(level.n) * sizeof(int)),
                 "cudaMalloc gmg parent");
      check_cuda(cudaMemcpy(level.d_parent, host.parent.data(),
                            static_cast<std::size_t>(level.n) * sizeof(int), cudaMemcpyHostToDevice),
                 "cudaMemcpy gmg parent");
      check_cuda(
          cudaMalloc(&level.d_scatter_to, static_cast<std::size_t>(std::max(level.nnz, 1)) * sizeof(int)),
          "cudaMalloc gmg scatter");
      if (level.nnz > 0) {
        check_cuda(cudaMemcpy(level.d_scatter_to, host.scatter_to.data(),
                              static_cast<std::size_t>(level.nnz) * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy gmg scatter");
      }
    }
    if (ell > 0 && level.n > 0) {
      check_cuda(cudaMalloc(&level.d_r, n_bytes), "cudaMalloc gmg r");
      check_cuda(cudaMalloc(&level.d_z, n_bytes), "cudaMalloc gmg z");
    }
    if (level.n > 0) {
      check_cuda(cudaMalloc(&level.d_az, n_bytes), "cudaMalloc gmg az");
      check_cuda(cudaMalloc(&level.d_tmp, n_bytes), "cudaMalloc gmg tmp");
    }
  }

  if (!levels_.empty() && levels_.front().n > 0) {
    extract_diag(levels_.front());
    scatter_coarse_values();
  }

  std::vector<int> defaults(levels_.size(), 0);
  if (!defaults.empty()) {
    defaults.front() = 50;
    if (defaults.size() > 1u) {
      defaults.back() = kDefaultCoarsestRestart;
    }
  }
  set_restart_schedule(defaults);
}

void PoissonGmgCuda::scatter_coarse_values() const {
  for (std::size_t ell = 0; ell + 1u < levels_.size(); ++ell) {
    const Level& lo = levels_[ell];
    const Level& hi = levels_[ell + 1u];
    if (hi.n <= 0) {
      continue;
    }
    check_cuda(cudaMemset(hi.d_val, 0, static_cast<std::size_t>(std::max(hi.nnz, 1)) * sizeof(double)),
               "cudaMemset gmg coarse val");
    if (lo.nnz > 0 && hi.nnz > 0) {
      k_scatter_add<<<blocks_for(lo.nnz), threads_per_block()>>>(lo.nnz, lo.d_scatter_to, lo.d_val,
                                                                hi.d_val);
      check_cuda(cudaGetLastError(), "k_scatter_add launch");
    }
    extract_diag(hi);
  }
}

void PoissonGmgCuda::extract_diag(const Level& level) const {
  if (level.n <= 0) {
    return;
  }
  k_extract_diag<<<blocks_for(level.n), threads_per_block()>>>(
      level.n, level.d_row_off, level.d_col_idx, level.d_val, level.d_diag, level.d_inv_diag);
  check_cuda(cudaGetLastError(), "k_extract_diag launch");
}

void PoissonGmgCuda::set_restart_schedule(const std::vector<int>& restarts) {
  if (levels_.empty()) {
    return;
  }
  const int n_levels = static_cast<int>(levels_.size());
  std::vector<int> schedule(static_cast<std::size_t>(n_levels), 0);
  if (restarts.size() == 1u) {
    schedule.front() = restarts.front();
    if (n_levels > 1) {
      schedule.back() = kDefaultCoarsestRestart;
    }
  } else if (!restarts.empty()) {
    for (std::size_t i = 0; i < restarts.size() && i < schedule.size(); ++i) {
      schedule[i] = restarts[i];
    }
    if (static_cast<int>(restarts.size()) < n_levels) {
      schedule.back() = kDefaultCoarsestRestart;
    }
  } else {
    schedule.front() = 50;
    if (n_levels > 1) {
      schedule.back() = kDefaultCoarsestRestart;
    }
  }
  if (schedule.front() < 2) {
    throw std::invalid_argument("finest GMRES restart must be >= 2");
  }
  for (int v : schedule) {
    if (v < 0 || v == 1) {
      throw std::invalid_argument("GMRES restart entries must be 0 or >= 2");
    }
  }
  for (int ell = 0; ell < n_levels; ++ell) {
    Level& level = levels_[static_cast<std::size_t>(ell)];
    int rst = schedule[static_cast<std::size_t>(ell)];
    if (ell == n_levels - 1 && ell > 0 && rst > 0) {
      rst = std::min(rst, std::max(level.n, 2));
    }
    if (level.d_basis && rst != level.restart) {
      cudaFree(level.d_basis);
      cudaFree(level.d_w);
      level.d_basis = nullptr;
      level.d_w = nullptr;
    }
    level.restart = rst;
    if (ell == n_levels - 1 && ell > 0 && rst >= 2 && level.n > 0 && !level.d_basis) {
      check_cuda(cudaMalloc(&level.d_basis,
                            static_cast<std::size_t>(rst + 1) * static_cast<std::size_t>(level.n) *
                                sizeof(double)),
                 "cudaMalloc gmg coarsest basis");
      check_cuda(cudaMalloc(&level.d_w, static_cast<std::size_t>(level.n) * sizeof(double)),
                 "cudaMalloc gmg coarsest w");
    }
  }
}

std::vector<int> PoissonGmgCuda::unknown_counts() const {
  std::vector<int> out;
  out.reserve(levels_.size());
  for (const Level& level : levels_) {
    out.push_back(level.n);
  }
  return out;
}

std::vector<int> PoissonGmgCuda::restart_schedule() const {
  std::vector<int> out;
  out.reserve(levels_.size());
  for (const Level& level : levels_) {
    out.push_back(level.restart);
  }
  return out;
}

void PoissonGmgCuda::spmv(const Level& level, const double* d_x, double* d_y) const {
  if (level.n <= 0) {
    return;
  }
  k_csr_spmv<<<blocks_for(level.n), threads_per_block()>>>(level.n, level.d_row_off, level.d_col_idx,
                                                           level.d_val, d_x, d_y);
  check_cuda(cudaGetLastError(), "k_csr_spmv gmg launch");
}

void PoissonGmgCuda::smooth(const Level& level,
                            const double* d_r,
                            double* d_z,
                            int nu,
                            bool z_is_zero) const {
  if (level.n <= 0 || nu <= 0) {
    return;
  }
  for (int s = 0; s < nu; ++s) {
    if (z_is_zero && s == 0) {
      k_scale_jacobi<<<blocks_for(level.n), threads_per_block()>>>(d_z, d_r, level.d_inv_diag, omega_,
                                                                   level.n);
      z_is_zero = false;
    } else {
      spmv(level, d_z, level.d_az);
      k_damped_jacobi<<<blocks_for(level.n), threads_per_block()>>>(
          d_z, d_r, level.d_az, level.d_inv_diag, omega_, level.n);
    }
  }
  check_cuda(cudaGetLastError(), "gmg smoother launch");
}

void PoissonGmgCuda::coarsest_solve(const Level& level, const double* d_r, double* d_z) const {
  if (level.n <= 0) {
    return;
  }
  if (level.restart < 2 || level.d_basis == nullptr) {
    check_cuda(cudaMemset(d_z, 0, static_cast<std::size_t>(level.n) * sizeof(double)),
               "cudaMemset gmg coarsest z");
    k_scale_jacobi<<<blocks_for(level.n), threads_per_block()>>>(d_z, d_r, level.d_inv_diag, omega_,
                                                                 level.n);
    for (int s = 1; s < kCoarseJacobiSweeps; ++s) {
      spmv(level, d_z, level.d_az);
      k_damped_jacobi<<<blocks_for(level.n), threads_per_block()>>>(
          d_z, d_r, level.d_az, level.d_inv_diag, omega_, level.n);
    }
    return;
  }

  cublasHandle_t h = static_cast<cublasHandle_t>(handle_cublas_);
  const int n = level.n;
  const int restart = std::min(level.restart, 16);
  auto basis = [&](int index) { return level.d_basis + static_cast<std::size_t>(index) * static_cast<std::size_t>(n); };

  check_cuda(cudaMemset(d_z, 0, static_cast<std::size_t>(n) * sizeof(double)),
             "cudaMemset gmg coarsest z");
  spmv(level, d_z, level.d_az);
  k_residual<<<blocks_for(n), threads_per_block()>>>(level.d_tmp, d_r, level.d_az, n);
  const double rhs_linf = device_max_abs(h, n, d_r);
  double r_linf = device_max_abs(h, n, level.d_tmp);
  if (residual_meets(r_linf, rhs_linf, kCoarseTol)) {
    return;
  }

  const int rows_h = restart + 1;
  std::vector<double> hessenberg(static_cast<std::size_t>(rows_h * restart), 0.0);
  int total = 0;
  const int max_iters = restart;
  while (total < max_iters) {
    k_jacobi<<<blocks_for(n), threads_per_block()>>>(level.d_w, level.d_tmp, level.d_inv_diag, n);
    double beta = 0.0;
    check_cublas(cublasDnrm2(h, n, level.d_w, 1, &beta), "cublasDnrm2 gmg coarse beta");
    if (!(beta > 0.0) || !std::isfinite(beta)) {
      break;
    }
    check_cublas(cublasDcopy(h, n, level.d_w, 1, basis(0), 1), "cublasDcopy gmg coarse v0");
    const double inv_beta = 1.0 / beta;
    check_cublas(cublasDscal(h, n, &inv_beta, basis(0), 1), "cublasDscal gmg coarse v0");
    std::fill(hessenberg.begin(), hessenberg.end(), 0.0);
    int inner = 0;
    for (int j = 0; j < restart && total < max_iters; ++j) {
      spmv(level, basis(j), level.d_az);
      k_jacobi<<<blocks_for(n), threads_per_block()>>>(level.d_w, level.d_az, level.d_inv_diag, n);
      for (int i = 0; i <= j; ++i) {
        double hij = 0.0;
        check_cublas(cublasDdot(h, n, level.d_w, 1, basis(i), 1, &hij), "cublasDdot gmg coarse h");
        hessenberg[static_cast<std::size_t>(i + rows_h * j)] = hij;
        const double neg = -hij;
        check_cublas(cublasDaxpy(h, n, &neg, basis(i), 1, level.d_w, 1),
                     "cublasDaxpy gmg coarse gs");
      }
      double hnext = 0.0;
      check_cublas(cublasDnrm2(h, n, level.d_w, 1, &hnext), "cublasDnrm2 gmg coarse hnext");
      hessenberg[static_cast<std::size_t>((j + 1) + rows_h * j)] = hnext;
      if (hnext > DBL_MIN) {
        check_cublas(cublasDcopy(h, n, level.d_w, 1, basis(j + 1), 1), "cublasDcopy gmg coarse vnext");
        const double inv = 1.0 / hnext;
        check_cublas(cublasDscal(h, n, &inv, basis(j + 1), 1), "cublasDscal gmg coarse vnext");
      }
      ++total;
      ++inner;
      if (hnext <= DBL_MIN) {
        break;
      }
    }
    const std::vector<double> y = solve_least_squares(hessenberg, beta, rows_h, inner);
    for (int i = 0; i < inner; ++i) {
      const double yi = y[static_cast<std::size_t>(i)];
      check_cublas(cublasDaxpy(h, n, &yi, basis(i), 1, d_z, 1), "cublasDaxpy gmg coarse x");
    }
    spmv(level, d_z, level.d_az);
    k_residual<<<blocks_for(n), threads_per_block()>>>(level.d_tmp, d_r, level.d_az, n);
    r_linf = device_max_abs(h, n, level.d_tmp);
    if (residual_meets(r_linf, rhs_linf, kCoarseTol)) {
      break;
    }
  }
}

void PoissonGmgCuda::vcycle(int ell, const double* d_r, double* d_z) const {
  const Level& level = levels_[static_cast<std::size_t>(ell)];
  if (ell == static_cast<int>(levels_.size()) - 1) {
    coarsest_solve(level, d_r, d_z);
    return;
  }
  const Level& coarse = levels_[static_cast<std::size_t>(ell) + 1u];
  check_cuda(cudaMemset(d_z, 0, static_cast<std::size_t>(level.n) * sizeof(double)),
             "cudaMemset gmg z");
  smooth(level, d_r, d_z, pre_smooth_, true);
  spmv(level, d_z, level.d_az);
  k_residual<<<blocks_for(level.n), threads_per_block()>>>(level.d_tmp, d_r, level.d_az, level.n);
  if (coarse.n > 0) {
    check_cuda(cudaMemset(coarse.d_r, 0, static_cast<std::size_t>(coarse.n) * sizeof(double)),
               "cudaMemset gmg coarse r");
    k_restrict<<<blocks_for(level.n), threads_per_block()>>>(level.n, level.d_parent, level.d_tmp,
                                                             coarse.d_r);
    vcycle(ell + 1, coarse.d_r, coarse.d_z);
    k_prolong<<<blocks_for(level.n), threads_per_block()>>>(level.n, level.d_parent, coarse.d_z, d_z);
  }
  smooth(level, d_r, d_z, post_smooth_, false);
}

void PoissonGmgCuda::apply(const double* d_r, double* d_z) const {
  if (levels_.empty() || levels_.front().n <= 0) {
    return;
  }
  vcycle(0, d_r, d_z);
}

#include "poisson_pcg_cuda.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "poisson_world.hpp"

namespace {

constexpr double kAtol = 1e-30;

bool residual_meets_tolerance(double r_inf, double rhs_inf, double rtol) {
  return r_inf <= kAtol + rtol * rhs_inf;
}

__global__ void k_spmv(int n,
                       const int* row_off,
                       const int* col_idx,
                       const double* offd,
                       const double* diag,
                       const double* xvec,
                       double* y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  double v = diag[row] * xvec[row];
  for (int k = row_off[row]; k < row_off[row + 1]; ++k) {
    v -= offd[k] * xvec[col_idx[k]];
  }
  y[row] = v;
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

__global__ void k_compute_alpha(double* d_alpha,
                                const double* d_rz_old,
                                const double* d_denom,
                                int* d_fail) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const double denom = *d_denom;
  if (fabs(denom) <= DBL_MIN) {
    *d_fail = 1;
    *d_alpha = 0.0;
    return;
  }
  *d_alpha = (*d_rz_old) / denom;
}

__global__ void k_xpay(double* x, const double* p, const double* d_alpha, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += (*d_alpha) * p[i];
  }
}

__global__ void k_rsub(double* r, const double* ap, const double* d_alpha, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    r[i] -= (*d_alpha) * ap[i];
  }
}

__global__ void k_beta(double* d_rz_old,
                       const double* d_rz_new,
                       double* d_beta,
                       int* d_fail) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const double rzn = *d_rz_new;
  const double rzo = *d_rz_old;
  if (!(rzn > 0.0)) {
    *d_fail = 1;
    *d_beta = 0.0;
    return;
  }
  *d_beta = rzn / rzo;
  *d_rz_old = rzn;
}

__global__ void k_p_combine(double* p, const double* z, const double* d_beta, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = z[i] + (*d_beta) * p[i];
  }
}

void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

void check_cublas(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cublas error " +
                             std::to_string(static_cast<int>(s)));
  }
}

int threads_per_block() {
  return 256;
}

void launch_spmv(int n,
                 const int* row_off,
                 const int* col_idx,
                 const double* offd,
                 const double* diag,
                 const double* xvec,
                 double* y,
                 cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  const int blocks = (n + threads_per_block() - 1) / threads_per_block();
  k_spmv<<<blocks, threads_per_block(), 0, stream>>>(n, row_off, col_idx, offd, diag, xvec, y);
}

void launch_residual(double* r, const double* rhs, const double* ax, int n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  const int blocks = (n + threads_per_block() - 1) / threads_per_block();
  k_residual<<<blocks, threads_per_block(), 0, stream>>>(r, rhs, ax, n);
}

double device_max_abs(cublasHandle_t h, int n, const double* d_v) {
  if (n <= 0) {
    return 0.0;
  }
  cublasPointerMode_t prev_mode;
  check_cublas(cublasGetPointerMode(h, &prev_mode), "cublasGetPointerMode");
  check_cublas(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode HOST");
  int idx = 0;
  check_cublas(cublasIdamax(h, n, d_v, 1, &idx), "cublasIdamax");
  check_cublas(cublasSetPointerMode(h, prev_mode), "cublasSetPointerMode restore");
  double v = 0.0;
  check_cuda(cudaMemcpy(&v, d_v + (idx - 1), sizeof(double), cudaMemcpyDeviceToHost),
             "cudaMemcpy max abs");
  return std::fabs(v);
}

}  // namespace

PoissonPcgCuda::PoissonPcgCuda(const PoissonWorld& world) : world_(&world) {
  n_ = world_->unknown_count();
  cublasHandle_t h{};
  check_cublas(cublasCreate(&h), "cublasCreate");
  check_cublas(cublasSetStream(h, nullptr), "cublasSetStream");
  handle_cublas_ = h;

  if (n_ <= 0) {
    return;
  }

  const auto& row_off_host = world_->row_offsets();
  const auto& col_idx_host = world_->col_indices();
  const auto& off_host = world_->offdiag_conductance();
  const auto& diag_host = world_->diagonal();
  const std::size_t nnz = off_host.size();

  std::vector<double> inv_diag(static_cast<std::size_t>(n_));
  std::vector<double> offd_d(nnz);
  std::vector<double> diag_d(static_cast<std::size_t>(n_));
  for (int i = 0; i < n_; ++i) {
    const double d = static_cast<double>(diag_host[static_cast<std::size_t>(i)]);
    inv_diag[static_cast<std::size_t>(i)] = 1.0 / std::max(d, 1e-30);
    diag_d[static_cast<std::size_t>(i)] = d;
  }
  for (std::size_t k = 0; k < nnz; ++k) {
    offd_d[k] = static_cast<double>(off_host[k]);
  }

  check_cuda(cudaMalloc(&d_row_off_, (static_cast<std::size_t>(n_) + 1u) * sizeof(int)),
             "cudaMalloc row_off");
  check_cuda(cudaMalloc(&d_col_idx_, nnz * sizeof(int)), "cudaMalloc col_idx");
  check_cuda(cudaMalloc(&d_offd_, nnz * sizeof(double)), "cudaMalloc offd");
  check_cuda(cudaMalloc(&d_diag_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc diag");
  check_cuda(cudaMalloc(&d_inv_diag_, static_cast<std::size_t>(n_) * sizeof(double)),
             "cudaMalloc inv_diag");
  check_cuda(cudaMemcpy(d_row_off_, row_off_host.data(),
                        (static_cast<std::size_t>(n_) + 1u) * sizeof(int),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy row_off");
  check_cuda(cudaMemcpy(d_col_idx_, col_idx_host.data(), nnz * sizeof(int), cudaMemcpyHostToDevice),
             "cudaMemcpy col_idx");
  check_cuda(cudaMemcpy(d_offd_, offd_d.data(), nnz * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy offd");
  check_cuda(cudaMemcpy(d_diag_, diag_d.data(), static_cast<std::size_t>(n_) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy diag");
  check_cuda(cudaMemcpy(d_inv_diag_, inv_diag.data(), static_cast<std::size_t>(n_) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy inv_diag");

  check_cuda(cudaMalloc(&d_x_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc x");
  check_cuda(cudaMalloc(&d_rhs_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc rhs");
  check_cuda(cudaMalloc(&d_r_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc r");
  check_cuda(cudaMalloc(&d_z_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc z");
  check_cuda(cudaMalloc(&d_p_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc p");
  check_cuda(cudaMalloc(&d_ap_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc ap");
  check_cuda(cudaMalloc(&d_ax_, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMalloc ax");
  check_cuda(cudaMalloc(&d_work_, sizeof(double)), "cudaMalloc work");
  check_cuda(cudaMalloc(&d_rz_old_, sizeof(double)), "cudaMalloc rz_old");
  check_cuda(cudaMalloc(&d_alpha_, sizeof(double)), "cudaMalloc alpha");
  check_cuda(cudaMalloc(&d_beta_, sizeof(double)), "cudaMalloc beta");
  check_cuda(cudaMalloc(&d_pcg_fail_, sizeof(int)), "cudaMalloc pcg_fail");
  reset_solution();
}

void PoissonPcgCuda::set_tolerance_check_batches(int first_batch, int subsequent_batch) {
  if (first_batch < 1 || subsequent_batch < 1) {
    throw std::invalid_argument("set_tolerance_check_batches: batch sizes must be >= 1");
  }
  tolerance_batch_first_ = first_batch;
  tolerance_batch_next_ = subsequent_batch;
}

void PoissonPcgCuda::reset_solution() {
  if (n_ > 0) {
    check_cuda(cudaMemset(d_x_, 0, static_cast<std::size_t>(n_) * sizeof(double)), "cudaMemset x");
  }
}

PoissonPcgCuda::~PoissonPcgCuda() {
  cudaFree(d_row_off_);
  cudaFree(d_col_idx_);
  cudaFree(d_offd_);
  cudaFree(d_diag_);
  cudaFree(d_inv_diag_);
  cudaFree(d_x_);
  cudaFree(d_rhs_);
  cudaFree(d_r_);
  cudaFree(d_z_);
  cudaFree(d_p_);
  cudaFree(d_ap_);
  cudaFree(d_ax_);
  cudaFree(d_work_);
  cudaFree(d_rz_old_);
  cudaFree(d_alpha_);
  cudaFree(d_beta_);
  cudaFree(d_pcg_fail_);
  if (handle_cublas_) {
    cublasDestroy(static_cast<cublasHandle_t>(handle_cublas_));
  }
}

PcgResult PoissonPcgCuda::solve(const std::vector<double>& rhs, std::vector<double>& x) const {
  PcgResult result;
  const int n = n_;
  if (static_cast<int>(rhs.size()) != n) {
    throw std::runtime_error("PCG rhs size does not match operator size");
  }
  if (n == 0) {
    result.converged = true;
    return result;
  }
  if (x.size() != rhs.size()) {
    x.assign(rhs.size(), 0.0);
    check_cuda(cudaMemset(d_x_, 0, static_cast<std::size_t>(n) * sizeof(double)), "cudaMemset x");
  }

  cublasHandle_t h = static_cast<cublasHandle_t>(handle_cublas_);
  const cudaStream_t stream = 0;
  check_cuda(cudaMemcpy(d_rhs_, rhs.data(), static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy rhs");
  check_cuda(cudaMemset(d_pcg_fail_, 0, sizeof(int)), "cudaMemset pcg_fail");

  launch_spmv(n, d_row_off_, d_col_idx_, d_offd_, d_diag_, d_x_, d_ax_, stream);
  launch_residual(d_r_, d_rhs_, d_ax_, n, stream);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after init residual");

  const double rhs_linf = device_max_abs(h, n, d_rhs_);
  const double r0_linf = device_max_abs(h, n, d_r_);
  result.rhs_inf_norm = rhs_linf;
  result.initial_residual_max_norm = r0_linf;
  result.residual_max_norm = r0_linf;
  result.residual_relative = (rhs_linf > 0.0) ? (r0_linf / rhs_linf) : 0.0;
  if (residual_meets_tolerance(result.residual_max_norm, rhs_linf, tolerance_)) {
    result.converged = true;
    check_cuda(cudaMemcpy(x.data(), d_x_, static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy x out");
    return result;
  }

  const int blocks = (n + threads_per_block() - 1) / threads_per_block();
  k_jacobi<<<blocks, threads_per_block(), 0, stream>>>(d_z_, d_r_, d_inv_diag_, n);
  check_cuda(cudaMemcpy(d_p_, d_z_, static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "cudaMemcpy p=z");
  check_cublas(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE), "cublas pointer DEVICE");
  check_cublas(cublasDdot(h, n, d_r_, 1, d_z_, 1, d_rz_old_), "cublasDdot initial rz");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after initial ddot");

  double rz_init_host = 0.0;
  check_cuda(cudaMemcpy(&rz_init_host, d_rz_old_, sizeof(double), cudaMemcpyDeviceToHost),
             "cudaMemcpy rz_init");
  if (!(rz_init_host > 0.0)) {
    check_cublas(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublas pointer HOST");
    result.numerical_failure = true;
    result.converged = residual_meets_tolerance(result.residual_max_norm, rhs_linf, tolerance_);
    check_cuda(cudaMemcpy(x.data(), d_x_, static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy x out");
    return result;
  }

  int total_iters = 0;
  while (total_iters < max_iterations_) {
    const int batch_cap = (total_iters == 0) ? tolerance_batch_first_ : tolerance_batch_next_;
    const int batch = std::min(batch_cap, max_iterations_ - total_iters);
    for (int k = 0; k < batch; ++k) {
      launch_spmv(n, d_row_off_, d_col_idx_, d_offd_, d_diag_, d_p_, d_ap_, stream);
      check_cublas(cublasDdot(h, n, d_p_, 1, d_ap_, 1, d_work_), "cublasDdot denom");
      k_compute_alpha<<<1, 1, 0, stream>>>(d_alpha_, d_rz_old_, d_work_, d_pcg_fail_);
      k_xpay<<<blocks, threads_per_block(), 0, stream>>>(d_x_, d_p_, d_alpha_, n);
      k_rsub<<<blocks, threads_per_block(), 0, stream>>>(d_r_, d_ap_, d_alpha_, n);
      k_jacobi<<<blocks, threads_per_block(), 0, stream>>>(d_z_, d_r_, d_inv_diag_, n);
      check_cublas(cublasDdot(h, n, d_r_, 1, d_z_, 1, d_work_), "cublasDdot rz_new");
      k_beta<<<1, 1, 0, stream>>>(d_rz_old_, d_work_, d_beta_, d_pcg_fail_);
      k_p_combine<<<blocks, threads_per_block(), 0, stream>>>(d_p_, d_z_, d_beta_, n);
      ++total_iters;
      result.iterations = total_iters;
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after PCG batch");
    result.residual_max_norm = device_max_abs(h, n, d_r_);
    result.residual_relative =
        (rhs_linf > 0.0) ? (result.residual_max_norm / rhs_linf) : 0.0;
    if (residual_meets_tolerance(result.residual_max_norm, rhs_linf, tolerance_)) {
      result.converged = true;
      break;
    }
  }

  int fail_h = 0;
  check_cuda(cudaMemcpy(&fail_h, d_pcg_fail_, sizeof(int), cudaMemcpyDeviceToHost),
             "cudaMemcpy pcg_fail");
  result.numerical_failure = fail_h != 0;
  result.converged = result.converged ||
                     residual_meets_tolerance(result.residual_max_norm, rhs_linf, tolerance_);
  check_cublas(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublas pointer HOST final");
  check_cuda(cudaMemcpy(x.data(), d_x_, static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy x out");
  return result;
}

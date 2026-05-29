#pragma once

#include <vector>

#include "pcg_result.hpp"

class PoissonWorld;

class PoissonPcgCuda {
 public:
  explicit PoissonPcgCuda(const PoissonWorld& world);
  ~PoissonPcgCuda();

  PoissonPcgCuda(const PoissonPcgCuda&) = delete;
  PoissonPcgCuda& operator=(const PoissonPcgCuda&) = delete;
  PoissonPcgCuda(PoissonPcgCuda&&) noexcept = delete;
  PoissonPcgCuda& operator=(PoissonPcgCuda&&) noexcept = delete;

  void set_tolerance(double tolerance) { tolerance_ = tolerance; }
  void set_max_iterations(int max_iterations) { max_iterations_ = max_iterations; }
  void set_tolerance_check_batches(int first_batch, int subsequent_batch);
  void reset_solution();

  PcgResult solve(const std::vector<double>& rhs, std::vector<double>& x) const;

 private:
  const PoissonWorld* world_ = nullptr;
  int n_ = 0;
  double tolerance_ = 1e-5;
  int max_iterations_ = 2000;
  int tolerance_batch_first_ = 1000;
  int tolerance_batch_next_ = 500;

  void* handle_cublas_ = nullptr;

  int* d_row_off_ = nullptr;
  int* d_col_idx_ = nullptr;
  double* d_offd_ = nullptr;
  double* d_diag_ = nullptr;
  double* d_inv_diag_ = nullptr;

  mutable double* d_x_ = nullptr;
  double* d_rhs_ = nullptr;
  mutable double* d_r_ = nullptr;
  mutable double* d_z_ = nullptr;
  mutable double* d_p_ = nullptr;
  mutable double* d_ap_ = nullptr;
  double* d_ax_ = nullptr;

  double* d_work_ = nullptr;
  double* d_rz_old_ = nullptr;
  double* d_alpha_ = nullptr;
  double* d_beta_ = nullptr;
  int* d_pcg_fail_ = nullptr;
};

#pragma once

#include <string>
#include <vector>

#include "pcg_result.hpp"

class PoissonWorld;

struct PicardResult {
  PcgResult final_pcg;
  int picard_sweeps_used = 0;
  int total_pcg_iterations = 0;
  /// Infinity-norm change in the unknown vector between the last two Picard
  /// iterates (0 if only one sweep / no Picard).
  double picard_error = 0.0;
  /// Relative PCG residual of the last inner solve (``residual / rhs_inf``).
  double pcg_error = 0.0;
  bool pcg_converged = false;
  std::string note;
};

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

  void upload_spd_operator(const PoissonWorld& world);
  void upload_skew_operator(const PoissonWorld& world);
  void clear_skew_operator();

  PcgResult solve(const std::vector<double>& rhs, std::vector<double>& x) const;
  PicardResult solve_picard(const std::vector<double>& rhs_s,
                            const std::vector<double>& rhs_k,
                            int picard_sweeps,
                            double picard_tolerance,
                            std::vector<double>& x) const;

 private:
  PcgResult solve_device_rhs(double* d_rhs) const;
  void ensure_picard_buffers() const;
  void free_skew_device();
  void free_picard_device();

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
  int nnz_ = 0;

  int* d_skew_row_off_ = nullptr;
  int* d_skew_col_idx_ = nullptr;
  double* d_skew_val_ = nullptr;
  int skew_nnz_ = 0;
  bool has_skew_ = false;

  mutable double* d_x_ = nullptr;
  double* d_rhs_ = nullptr;
  mutable double* d_r_ = nullptr;
  mutable double* d_z_ = nullptr;
  mutable double* d_p_ = nullptr;
  mutable double* d_ap_ = nullptr;
  double* d_ax_ = nullptr;

  mutable double* d_rhs_s_ = nullptr;
  mutable double* d_rhs_k_ = nullptr;
  mutable double* d_rhs_picard_ = nullptr;
  mutable double* d_kx_ = nullptr;
  mutable double* d_prev_rhs_ = nullptr;
  mutable double* d_prev_x_ = nullptr;

  double* d_work_ = nullptr;
  double* d_rz_old_ = nullptr;
  double* d_alpha_ = nullptr;
  double* d_beta_ = nullptr;
  int* d_pcg_fail_ = nullptr;
};

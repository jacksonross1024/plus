#pragma once

#include <vector>

class PoissonWorld;

// Geometric multigrid V-cycle on a frozen scalar-conductivity operator (7-point
// faces only; no AMR/AHE/THE/OHE). Hierarchy, injection maps, and Galerkin
// scatter maps are built once from geometry and stay device-resident.
class PoissonGmgCuda {
 public:
  PoissonGmgCuda();
  ~PoissonGmgCuda();

  PoissonGmgCuda(const PoissonGmgCuda&) = delete;
  PoissonGmgCuda& operator=(const PoissonGmgCuda&) = delete;
  PoissonGmgCuda(PoissonGmgCuda&&) noexcept = delete;
  PoissonGmgCuda& operator=(PoissonGmgCuda&&) noexcept = delete;

  void build(const PoissonWorld& world);

  void set_restart_schedule(const std::vector<int>& restarts);

  // Flexible preconditioner: z ≈ A_iso^{-1} r on the 7-point scalar operator.
  void apply(const double* d_r, double* d_z) const;

  int n_levels() const { return static_cast<int>(levels_.size()); }
  std::vector<int> unknown_counts() const;
  std::vector<int> restart_schedule() const;
  bool void_sparsity_ok() const { return void_sparsity_ok_; }
  bool ready() const { return !levels_.empty(); }

 private:
  struct Level {
    int n = 0;
    int nnz = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int restart = 0;
    bool owns_pattern = true;

    int* d_row_off = nullptr;
    int* d_col_idx = nullptr;
    double* d_val = nullptr;
    double* d_diag = nullptr;
    double* d_inv_diag = nullptr;
    int* d_parent = nullptr;      // size n, maps to next coarser unknown or -1
    int* d_scatter_to = nullptr;  // size nnz, maps to next coarser nnz or -1
    double* d_r = nullptr;
    double* d_z = nullptr;
    double* d_az = nullptr;
    double* d_tmp = nullptr;
    double* d_basis = nullptr;
    double* d_w = nullptr;
  };

  void destroy_level(Level& level);
  void vcycle(int ell, const double* d_r, double* d_z) const;
  void smooth(const Level& level, const double* d_r, double* d_z, int nu, bool z_is_zero) const;
  void spmv(const Level& level, const double* d_x, double* d_y) const;
  void coarsest_solve(const Level& level, const double* d_r, double* d_z) const;
  void extract_diag(const Level& level) const;
  void scatter_coarse_values() const;

  std::vector<Level> levels_;
  void* handle_cublas_ = nullptr;
  double* d_scalar_ = nullptr;
  bool void_sparsity_ok_ = false;
  int pre_smooth_ = 2;
  int post_smooth_ = 2;
  double omega_ = 0.7;
};

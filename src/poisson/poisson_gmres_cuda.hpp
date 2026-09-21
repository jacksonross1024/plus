#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pcg_result.hpp"

class PoissonWorld;
class PoissonGmgCuda;

enum class PoissonPreconditionerKind {
  kJacobi,
  kGmg,
};

class PoissonGmresCuda {
 public:
  explicit PoissonGmresCuda(const PoissonWorld& world);
  ~PoissonGmresCuda();

  PoissonGmresCuda(const PoissonGmresCuda&) = delete;
  PoissonGmresCuda& operator=(const PoissonGmresCuda&) = delete;
  PoissonGmresCuda(PoissonGmresCuda&&) noexcept = delete;
  PoissonGmresCuda& operator=(PoissonGmresCuda&&) noexcept = delete;

  void set_tolerance(double tolerance) { tolerance_ = tolerance; }
  void set_max_iterations(int max_iterations) { max_iterations_ = max_iterations; }
  void set_restart(int restart);
  void set_restart_schedule(const std::vector<int>& restarts);
  void set_preconditioner(PoissonPreconditionerKind kind);
  void reset_solution();
  // Device-only: d_x := alpha * d_x. Used for contact-voltage warm-start scaling.
  void scale_solution(double alpha);

  void upload_transport_operator(const PoissonWorld& world);
  void prepare_transport_update(const PoissonWorld& world);
  void build_gmg(const PoissonWorld& world);
  // Host magnetization path (H2D then device update).
  void update_transport_operator_and_rhs_device(const PoissonWorld& world,
                                                const std::vector<float>& magnetization_fm_stack,
                                                const std::vector<double>& potentials);
  // Device magnetization already in device_magnetization(); only potentials are uploaded.
  void update_transport_operator_and_rhs_device(const PoissonWorld& world,
                                                const std::vector<double>& potentials);

  void set_applied_field_uniform(float bx, float by, float bz);
  void set_applied_field_grid(const float* h_b_ext, std::size_t n_values);

  float* device_magnetization() { return d_magnetization_; }
  const float* device_magnetization() const { return d_magnetization_; }
  std::size_t magnetization_device_bytes() const;
  void copy_magnetization_device_to_host(std::vector<float>& out) const;

  PcgResult solve(const std::vector<double>& rhs, std::vector<double>& x) const;
  PcgResult solve_device_rhs(std::vector<double>& x) const;

  PoissonPreconditionerKind preconditioner() const { return preconditioner_; }
  bool gmg_ready() const;
  int gmg_n_levels() const;
  std::vector<int> gmg_unknown_counts() const;
  std::vector<int> restart_schedule() const;
  bool gmg_void_sparsity_ok() const;

 private:
  void ensure_vectors() const;
  void ensure_spmv_descriptors() const;
  void destroy_spmv_descriptors() const;
  void spmv(const double* d_x, double* d_y) const;
  void copy_solution_to_host(std::vector<double>& x) const;
  double* basis_vector(int index) const;
  double* zbasis_vector(int index) const;
  void launch_transport_update(bool amr, bool ahe, bool the, bool ohe) const;
  PcgResult solve_device_rhs_jacobi(std::vector<double>& x) const;
  PcgResult solve_device_rhs_fgmres(std::vector<double>& x) const;
  double nrm2_d2h(const double* d_x) const;
  double dot_d2h(const double* d_x, const double* d_y) const;

  const PoissonWorld* world_ = nullptr;
  int n_ = 0;
  int nnz_ = 0;
  double tolerance_ = 1e-6;
  int max_iterations_ = 2000;
  int restart_ = 200;
  std::vector<int> restart_schedule_{200};
  PoissonPreconditionerKind preconditioner_ = PoissonPreconditionerKind::kJacobi;

  void* handle_cublas_ = nullptr;
  void* handle_cusparse_ = nullptr;
  void* spmat_ = nullptr;
  mutable void* spmv_buffer_ = nullptr;
  mutable std::size_t spmv_buffer_size_ = 0;
  mutable void* dnvec_x_ = nullptr;
  mutable void* dnvec_y_ = nullptr;

  int* d_row_off_ = nullptr;
  int* d_col_idx_ = nullptr;
  double* d_val_ = nullptr;
  double* d_diag_ = nullptr;
  double* d_inv_diag_ = nullptr;
  int* d_unknown_index_ = nullptr;
  int* d_unknown_to_cell_ = nullptr;
  signed char* d_region_ = nullptr;
  signed char* d_contact_id_ = nullptr;
  float* d_sigma_ = nullptr;
  float* d_magnetization_ = nullptr;
  float* d_h_ = nullptr;
  double* d_contact_potentials_ = nullptr;
  int* d_update_fail_ = nullptr;
  mutable double* d_reduce_ = nullptr;
  int cell_count_ = 0;
  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;
  int first_r2_layer_ = 0;
  int fm_layer_count_ = 0;
  int num_contacts_ = 0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  double cz_ = 0.0;
  bool amr_enabled_ = false;
  bool ahe_enabled_ = false;
  bool the_enabled_ = false;
  bool ohe_enabled_ = false;
  bool resistivity_invert_ = false;
  double amr_ratio_ = 0.0;
  double ahe_ratio_ = 0.0;
  double the_ratio_ = 0.0;
  double hall_coefficient_pt_ = -2.44e-11;
  double hall_coefficient_fm_ = 3.09e-10;
  float applied_bx_ = 0.0f;
  float applied_by_ = 0.0f;
  float applied_bz_ = 0.0f;
  bool applied_field_uniform_ = true;
  float* d_b_ext_ = nullptr;
  bool transport_update_ready_ = false;

  mutable double* d_x_ = nullptr;
  mutable double* d_rhs_ = nullptr;
  mutable double* d_r_ = nullptr;
  mutable double* d_z_ = nullptr;
  mutable double* d_w_ = nullptr;
  mutable double* d_aw_ = nullptr;
  mutable double* d_basis_ = nullptr;
  mutable double* d_zbasis_ = nullptr;

  std::unique_ptr<PoissonGmgCuda> gmg_;
};

#pragma once

#include <string>
#include <vector>

#include "pcg_result.hpp"

class PoissonWorld;

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
  void reset_solution();

  void upload_transport_operator(const PoissonWorld& world);
  void prepare_transport_update(const PoissonWorld& world);
  // Host magnetization path (H2D then device update).
  void update_transport_operator_and_rhs_device(const PoissonWorld& world,
                                                const std::vector<float>& magnetization_fm_stack,
                                                const std::vector<double>& potentials);
  // Device magnetization already in device_magnetization(); only potentials are uploaded.
  void update_transport_operator_and_rhs_device(const PoissonWorld& world,
                                                const std::vector<double>& potentials);

  float* device_magnetization() { return d_magnetization_; }
  const float* device_magnetization() const { return d_magnetization_; }
  std::size_t magnetization_device_bytes() const;
  void copy_magnetization_device_to_host(std::vector<float>& out) const;

  PcgResult solve(const std::vector<double>& rhs, std::vector<double>& x) const;
  PcgResult solve_device_rhs(std::vector<double>& x) const;

 private:
  void ensure_vectors() const;
  void ensure_spmv_descriptors() const;
  void destroy_spmv_descriptors() const;
  void spmv(const double* d_x, double* d_y) const;
  void copy_solution_to_host(std::vector<double>& x) const;
  double* basis_vector(int index) const;

  const PoissonWorld* world_ = nullptr;
  int n_ = 0;
  int nnz_ = 0;
  double tolerance_ = 1e-5;
  int max_iterations_ = 2000;
  int restart_ = 50;

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
  double* d_contact_potentials_ = nullptr;
  int* d_update_fail_ = nullptr;
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
  double amr_ratio_ = 0.0;
  double ahe_ratio_ = 0.0;
  bool transport_update_ready_ = false;

  mutable double* d_x_ = nullptr;
  mutable double* d_rhs_ = nullptr;
  mutable double* d_r_ = nullptr;
  mutable double* d_z_ = nullptr;
  mutable double* d_w_ = nullptr;
  mutable double* d_aw_ = nullptr;
  mutable double* d_basis_ = nullptr;
};

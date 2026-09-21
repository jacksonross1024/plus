#pragma once

#include <string>
#include <vector>

#include "poisson_current.hpp"
#include "poisson_gmres_cuda.hpp"
#include "poisson_hall.hpp"
#include "poisson_pcg_cuda.hpp"
#include "poisson_world.hpp"
#include "signal_loader.hpp"

enum class PoissonLinearSolverKind {
  kPcg,
  kGmresCusparse,
};

class PoissonCudaSession {
 public:
  PoissonCudaSession(PoissonWorld world,
                     ContactPotentials potentials,
                     double tolerance,
                     int max_iterations,
                     double skip_threshold,
                     const std::string& slice_x,
                     const std::string& slice_y,
                     const std::string& slice_z,
                     int cuda_tol_batch_first,
                     int cuda_tol_batch_next,
                     TransportConfig transport = TransportConfig{},
                     PoissonLinearSolverKind solver_kind = PoissonLinearSolverKind::kPcg,
                     std::vector<int> gmres_restart = {200},
                     bool voltage_scale_guess = false,
                     PoissonPreconditionerKind preconditioner = PoissonPreconditionerKind::kJacobi);

  ~PoissonCudaSession();

  PoissonCudaSession(const PoissonCudaSession&) = delete;
  PoissonCudaSession& operator=(const PoissonCudaSession&) = delete;
  PoissonCudaSession(PoissonCudaSession&&) noexcept = delete;
  PoissonCudaSession& operator=(PoissonCudaSession&&) noexcept = delete;

  StepStats iterate();
  StepStats iterate_with_magnetization(const std::vector<float>& magnetization_fm_stack);
  StepStats iterate_with_magnetization_device(const float* d_mx,
                                              const float* d_my,
                                              const float* d_mz,
                                              int src_nz,
                                              int src_ny,
                                              int src_nx,
                                              const std::vector<int>& src_lo,
                                              const std::vector<int>& src_hi,
                                              const std::vector<float>& weight_hi,
                                              bool average_z);
  void reset();

  int current_step() const { return step_; }
  int n_steps() const { return static_cast<int>(potentials_.size()); }
  bool exhausted() const { return step_ >= n_steps(); }

  int nx() const { return world_.nx(); }
  int ny() const { return world_.ny(); }
  int nz() const { return world_.nz(); }
  double cx() const { return world_.cx(); }
  double cy() const { return world_.cy(); }
  double cz() const { return world_.cz(); }
  int first_r2_layer() const { return world_.first_r2_layer(); }
  double theta_sh() const { return world_.theta_sh(); }
  double decay_length() const { return world_.decay_length(); }
  int unknown_count() const { return world_.unknown_count(); }
  int num_contacts() const { return world_.num_contacts(); }
  int fm_layer_count() const { return world_.fm_layer_count(); }

  bool transport_enabled() const { return world_.transport_enabled(); }
  bool amr_enabled() const { return world_.amr_enabled(); }
  bool ahe_enabled() const { return world_.ahe_enabled(); }
  bool the_enabled() const { return world_.the_enabled(); }
  bool ohe_enabled() const { return world_.ohe_enabled(); }
  bool resistivity_invert() const { return transport_config_.resistivity_invert; }
  bool magnetization_required() const { return world_.magnetization_required(); }
  double amr_ratio() const { return transport_config_.amr_ratio; }
  double ahe_ratio() const { return transport_config_.ahe_ratio; }
  double the_ratio() const { return transport_config_.the_ratio; }
  double hall_coefficient_pt() const { return transport_config_.hall_coefficient_pt; }
  double hall_coefficient_fm() const { return transport_config_.hall_coefficient_fm; }
  int picard_sweeps() const { return transport_config_.picard_sweeps; }
  PoissonLinearSolverKind solver_kind() const { return solver_kind_; }
  PoissonPreconditionerKind preconditioner() const { return preconditioner_; }
  bool voltage_scale_guess() const { return voltage_scale_guess_; }
  int gmg_n_levels() const { return gmres_solver_.gmg_n_levels(); }
  std::vector<int> gmg_unknown_counts() const { return gmres_solver_.gmg_unknown_counts(); }
  std::vector<int> gmres_restart_schedule() const { return gmres_solver_.restart_schedule(); }
  bool gmg_void_sparsity_ok() const { return gmres_solver_.gmg_void_sparsity_ok(); }

  void set_applied_field_uniform(float bx, float by, float bz);
  void set_applied_field_grid(const std::vector<float>& b_poisson);

  int out_nx() const { return output_spec_.out_nx(); }
  int out_ny() const { return output_spec_.out_ny(); }
  int out_nz() const { return output_spec_.out_nz(); }

  const std::vector<float>& jmod_frame() const { return jmod_out_; }
  const std::vector<float>& jcur_frame() const { return jcur_out_; }

  void set_hall_probe_indices(HallProbeIndices probes);
  bool hall_probes_configured() const { return hall_configured_; }
  bool hall_frame_available() const { return hall_frame_available_; }
  bool last_frame_skipped() const { return last_frame_skipped_; }
  const std::vector<double>& hall_potentials() const;
  HallPotentialComponents hall_potential_components() const;
  const std::vector<float>& winding_fm_stack() const;
  std::vector<float> the_hall_vector_fm_stack() const;
  void winding_stats(float& max_abs, double& sum_hz) const;

 private:
  static void validate_contact_potentials(const PoissonWorld& world,
                                          const ContactPotentials& potentials);
  static int initial_max_iterations(int max_iterations);
  void maybe_scale_voltage_guess();
  void remember_solved_contact_voltages();
  StepStats iterate_impl(const std::vector<float>* magnetization_fm_stack,
                         double timing_device_magnetization_s = 0.0);
  StepStats finish_iterate_after_solve(StepStats stats,
                                       const PcgResult& result,
                                       double picard_error,
                                       int picard_sweeps_used,
                                       double elapsed_s);

  PoissonWorld world_;
  ContactPotentials potentials_;
  JmodOutputSpec output_spec_;
  PoissonPcgCuda pcg_solver_;
  PoissonGmresCuda gmres_solver_;
  TransportConfig transport_config_;
  PoissonLinearSolverKind solver_kind_ = PoissonLinearSolverKind::kPcg;
  PoissonPreconditionerKind preconditioner_ = PoissonPreconditionerKind::kJacobi;

  double tolerance_ = 1e-6;
  int max_iterations_ = 2000;
  double skip_threshold_ = 1e-5;
  int step_ = 0;
  bool first_solve_ = true;
  bool voltage_scale_guess_ = false;
  bool last_voltage_scale_applied_ = false;
  double last_voltage_scale_alpha_ = 1.0;
  std::vector<double> last_solved_applied_;

  std::vector<double> x_;
  std::vector<double> applied_;
  std::vector<double> rhs_;
  std::vector<double> rhs_s_;
  std::vector<double> rhs_k_;
  std::vector<float> phi_;
  std::vector<float> j_frame_;
  std::vector<float> jcur_full_;
  std::vector<float> jmod_out_;
  std::vector<float> jcur_out_;
  std::vector<float> pt_avg_xy_;

  HallProbeIndices hall_probes_;
  bool hall_configured_ = false;
  bool hall_frame_available_ = false;
  bool last_frame_skipped_ = false;
  std::vector<double> hall_voltages_;
  HallPotentialComponents hall_components_;

  void update_hall_readout_from_phi();
  void clear_hall_readout_to_zeros();
  void ensure_device_magnetization_mapping(const std::vector<int>& src_lo,
                                           const std::vector<int>& src_hi,
                                           const std::vector<float>& weight_hi,
                                           bool average_z);

  // Cached device mapping tables for iterate_with_magnetization_device (static geometry).
  int* d_map_lo_ = nullptr;
  int* d_map_hi_ = nullptr;
  float* d_map_weight_ = nullptr;
  int map_dst_nz_ = 0;
  bool map_average_z_ = false;
  bool map_tables_ready_ = false;
};

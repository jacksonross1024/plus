#pragma once

#include <string>
#include <vector>

#include "poisson_current.hpp"
#include "poisson_hall.hpp"
#include "poisson_pcg_cuda.hpp"
#include "poisson_world.hpp"
#include "signal_loader.hpp"

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
                     TransportConfig transport = TransportConfig{});

  PoissonCudaSession(const PoissonCudaSession&) = delete;
  PoissonCudaSession& operator=(const PoissonCudaSession&) = delete;
  PoissonCudaSession(PoissonCudaSession&&) noexcept = delete;
  PoissonCudaSession& operator=(PoissonCudaSession&&) noexcept = delete;

  StepStats iterate();
  StepStats iterate_with_magnetization(const std::vector<float>& magnetization_fm_stack);
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
  double amr_ratio() const { return transport_config_.amr_ratio; }
  double ahe_ratio() const { return transport_config_.ahe_ratio; }
  int picard_sweeps() const { return transport_config_.picard_sweeps; }

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

 private:
  static void validate_contact_potentials(const PoissonWorld& world,
                                          const ContactPotentials& potentials);
  static int initial_max_iterations(int max_iterations);
  StepStats iterate_impl(const std::vector<float>* magnetization_fm_stack);

  PoissonWorld world_;
  ContactPotentials potentials_;
  JmodOutputSpec output_spec_;
  PoissonPcgCuda solver_;
  TransportConfig transport_config_;

  double tolerance_ = 1e-5;
  int max_iterations_ = 2000;
  double skip_threshold_ = 1e-5;
  int step_ = 0;
  bool first_solve_ = true;

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
};

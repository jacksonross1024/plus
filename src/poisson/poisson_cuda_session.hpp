#pragma once

#include <string>
#include <vector>

#include "poisson_current.hpp"
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
                     int cuda_tol_batch_next);

  PoissonCudaSession(const PoissonCudaSession&) = delete;
  PoissonCudaSession& operator=(const PoissonCudaSession&) = delete;
  PoissonCudaSession(PoissonCudaSession&&) noexcept = delete;
  PoissonCudaSession& operator=(PoissonCudaSession&&) noexcept = delete;

  StepStats iterate();
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

  int out_nx() const { return output_spec_.out_nx(); }
  int out_ny() const { return output_spec_.out_ny(); }
  int out_nz() const { return output_spec_.out_nz(); }

  const std::vector<float>& jmod_frame() const { return jmod_out_; }
  const std::vector<float>& jcur_frame() const { return jcur_out_; }

 private:
  static void validate_contact_potentials(const ContactPotentials& potentials);
  static int initial_max_iterations(int max_iterations);

  PoissonWorld world_;
  ContactPotentials potentials_;
  JmodOutputSpec output_spec_;
  PoissonPcgCuda solver_;

  double tolerance_ = 1e-5;
  int max_iterations_ = 2000;
  double skip_threshold_ = 1e-5;
  int step_ = 0;
  bool first_solve_ = true;

  std::vector<double> x_;
  std::vector<double> rhs_;
  std::vector<float> phi_;
  std::vector<float> j_frame_;
  std::vector<float> jmod_out_;
  std::vector<float> jcur_out_;
  std::vector<float> pt_avg_xy_;
};

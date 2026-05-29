#include "poisson_cuda_session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void assert_finite_active_poisson_data(const std::vector<double>& rhs,
                                       const std::vector<double>& x,
                                       const std::vector<float>& phi,
                                       int step) {
  for (double v : rhs) {
    if (!std::isfinite(v)) {
      throw std::runtime_error("non-finite PCG rhs at Poisson timestep " +
                               std::to_string(step));
    }
  }
  for (double v : x) {
    if (!std::isfinite(v)) {
      throw std::runtime_error("non-finite Poisson solution at timestep " +
                               std::to_string(step));
    }
  }
  for (float v : phi) {
    if (!std::isfinite(v)) {
      throw std::runtime_error("non-finite electrical potential at timestep " +
                               std::to_string(step));
    }
  }
}

}  // namespace

PoissonCudaSession::PoissonCudaSession(PoissonWorld world,
                                       ContactPotentials potentials,
                                       double tolerance,
                                       int max_iterations,
                                       double skip_threshold,
                                       const std::string& slice_x,
                                       const std::string& slice_y,
                                       const std::string& slice_z,
                                       int cuda_tol_batch_first,
                                       int cuda_tol_batch_next)
    : world_(std::move(world)),
      potentials_(std::move(potentials)),
      output_spec_(make_jmod_output_spec(world_, slice_x, slice_y, slice_z)),
      solver_(world_),
      tolerance_(tolerance),
      max_iterations_(max_iterations),
      skip_threshold_(skip_threshold),
      x_(static_cast<std::size_t>(world_.unknown_count()), 0.0),
      phi_(static_cast<std::size_t>(world_.cell_count()), 0.0f),
      j_frame_(world_.frame_elements(), 0.0f),
      jmod_out_(output_spec_.frame_elements(), 0.0f),
      jcur_out_(output_spec_.frame_elements(), 0.0f),
      pt_avg_xy_(static_cast<std::size_t>(world_.nx() * world_.ny()) * 3u, 0.0f) {
  validate_contact_potentials(potentials_);
  if (!(tolerance_ > 0.0)) {
    throw std::runtime_error("PoissonCudaSession: tolerance must be > 0");
  }
  if (max_iterations_ <= 0) {
    throw std::runtime_error("PoissonCudaSession: max_iterations must be > 0");
  }
  if (skip_threshold_ < 0.0) {
    throw std::runtime_error("PoissonCudaSession: skip_threshold must be >= 0");
  }
  if (cuda_tol_batch_first < 1 || cuda_tol_batch_next < 1) {
    throw std::runtime_error("PoissonCudaSession: CUDA tolerance batches must be >= 1");
  }
  solver_.set_tolerance(tolerance_);
  solver_.set_tolerance_check_batches(cuda_tol_batch_first, cuda_tol_batch_next);
}

void PoissonCudaSession::validate_contact_potentials(const ContactPotentials& potentials) {
  if (potentials.c0.empty()) {
    throw std::runtime_error("PoissonCudaSession: contact potentials cannot be empty");
  }
  if (!(potentials.c0.size() == potentials.c1.size() &&
        potentials.c0.size() == potentials.c2.size())) {
    throw std::runtime_error("PoissonCudaSession: contact-potential channels differ in length");
  }
  for (std::size_t i = 0; i < potentials.c0.size(); ++i) {
    if (!std::isfinite(potentials.c0[i]) || !std::isfinite(potentials.c1[i]) ||
        !std::isfinite(potentials.c2[i])) {
      throw std::runtime_error("PoissonCudaSession: contact potentials contain NaN or Inf");
    }
  }
}

int PoissonCudaSession::initial_max_iterations(int max_iterations) {
  if (max_iterations > std::numeric_limits<int>::max() / 2) {
    return std::numeric_limits<int>::max();
  }
  return max_iterations * 2;
}

void PoissonCudaSession::reset() {
  step_ = 0;
  first_solve_ = true;
  std::fill(x_.begin(), x_.end(), 0.0);
  solver_.reset_solution();
  std::fill(jmod_out_.begin(), jmod_out_.end(), 0.0f);
  std::fill(jcur_out_.begin(), jcur_out_.end(), 0.0f);
}

StepStats PoissonCudaSession::iterate() {
  if (exhausted()) {
    throw std::out_of_range("PoissonCudaSession: contact-potential series exhausted");
  }

  const std::array<double, 3> applied = potentials_.at(static_cast<std::size_t>(step_));
  const double vmax =
      std::max({std::abs(applied[0]), std::abs(applied[1]), std::abs(applied[2])});

  StepStats stats;
  stats.step = step_;

  if (skip_threshold_ > 0.0 && vmax < skip_threshold_) {
    stats.skipped = true;
    std::fill(jmod_out_.begin(), jmod_out_.end(), 0.0f);
    std::fill(jcur_out_.begin(), jcur_out_.end(), 0.0f);
    ++step_;
    return stats;
  }

  world_.build_rhs(applied, rhs_);
  solver_.set_max_iterations(first_solve_ ? initial_max_iterations(max_iterations_)
                                          : max_iterations_);
  first_solve_ = false;

  const auto t0 = std::chrono::steady_clock::now();
  const PcgResult result = solver_.solve(rhs_, x_);
  const auto t1 = std::chrono::steady_clock::now();
  if (result.numerical_failure) {
    throw std::runtime_error("PoissonCudaSession: CUDA PCG numerical failure");
  }

  stats.iterations = result.iterations;
  stats.residual_initial = result.initial_residual_max_norm;
  stats.residual = result.residual_max_norm;
  stats.rhs_inf = result.rhs_inf_norm;
  stats.residual_rel = result.residual_relative;
  stats.elapsed_s = std::chrono::duration<double>(t1 - t0).count();

  world_.fill_full_phi(x_, applied, phi_);
  assert_finite_active_poisson_data(rhs_, x_, phi_, step_);

  compute_j_raw_from_phi(world_, phi_, j_frame_);
  log_poisson_phi_j_region_stats(world_, phi_, j_frame_, step_, "after_compute_j");

  // Raw FM **J** for ``jcur`` export (must be before postprocess overwrites FM cells).
  extract_jmod_subframe(world_, j_frame_, output_spec_, jcur_out_);

  apply_jmod_postprocess(world_, world_.decay_length(), j_frame_, pt_avg_xy_);
  log_poisson_phi_j_region_stats(world_, phi_, j_frame_, step_, "after_jmod_postprocess");

  extract_jmod_subframe(world_, j_frame_, output_spec_, jmod_out_);

  ++step_;
  return stats;
}

#include "poisson_cuda_session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

bool poisson_debug_finite_enabled() {
  const char* v = std::getenv("POISSON_DEBUG_FINITE");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

void assert_finite_active_poisson_data(const std::vector<double>& rhs,
                                       const std::vector<double>& x,
                                       const std::vector<float>& phi,
                                       int step) {
  // Opt-in only: full-grid finite scans are O(N) and redundant with PCG failure flags.
  if (!poisson_debug_finite_enabled()) {
    return;
  }
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
                                       int cuda_tol_batch_next,
                                       TransportConfig transport)
    : world_(std::move(world)),
      potentials_(std::move(potentials)),
      output_spec_(make_jmod_output_spec(world_, slice_x, slice_y, slice_z)),
      solver_(world_),
      transport_config_(transport),
      tolerance_(tolerance),
      max_iterations_(max_iterations),
      skip_threshold_(skip_threshold),
      x_(static_cast<std::size_t>(world_.unknown_count()), 0.0),
      phi_(static_cast<std::size_t>(world_.cell_count()), 0.0f),
      j_frame_(world_.frame_elements(), 0.0f),
      jcur_full_(world_.frame_elements(), 0.0f),
      jmod_out_(output_spec_.frame_elements(), 0.0f),
      jcur_out_(output_spec_.frame_elements(), 0.0f),
      pt_avg_xy_(static_cast<std::size_t>(world_.nx() * world_.ny()) * 3u, 0.0f) {
  validate_contact_potentials(world_, potentials_);
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
  world_.set_transport_config(transport_config_);
  // Host operator may have been rebuilt by set_transport_config; keep GPU in sync.
  solver_.upload_spd_operator(world_);
  solver_.set_tolerance(tolerance_);
  solver_.set_tolerance_check_batches(cuda_tol_batch_first, cuda_tol_batch_next);
}

void PoissonCudaSession::validate_contact_potentials(const PoissonWorld& world,
                                                     const ContactPotentials& potentials) {
  if (potentials.channels.empty()) {
    throw std::runtime_error("PoissonCudaSession: contact potentials cannot be empty");
  }
  if (static_cast<int>(potentials.num_contacts()) != world.num_contacts()) {
    throw std::runtime_error(
        "PoissonCudaSession: contact-potential channel count does not match Poisson world");
  }
  const std::size_t nt = potentials.channels.front().size();
  if (nt == 0) {
    throw std::runtime_error("PoissonCudaSession: contact potentials cannot be empty");
  }
  for (const auto& channel : potentials.channels) {
    if (channel.size() != nt) {
      throw std::runtime_error("PoissonCudaSession: contact-potential channels differ in length");
    }
    for (float v : channel) {
      if (!std::isfinite(v)) {
        throw std::runtime_error("PoissonCudaSession: contact potentials contain NaN or Inf");
      }
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
  hall_frame_available_ = false;
  last_frame_skipped_ = false;
  hall_voltages_.clear();
  hall_components_ = HallPotentialComponents{};
}

StepStats PoissonCudaSession::iterate() {
  return iterate_impl(nullptr);
}

StepStats PoissonCudaSession::iterate_with_magnetization(
    const std::vector<float>& magnetization_fm_stack) {
  return iterate_impl(&magnetization_fm_stack);
}

StepStats PoissonCudaSession::iterate_impl(const std::vector<float>* magnetization_fm_stack) {
  if (exhausted()) {
    throw std::out_of_range("PoissonCudaSession: contact-potential series exhausted");
  }

  potentials_.fill_at(static_cast<std::size_t>(step_), applied_);
  double vmax = 0.0;
  for (double v : applied_) {
    vmax = std::max(vmax, std::abs(v));
  }

  StepStats stats;
  stats.step = step_;

  if (skip_threshold_ > 0.0 && vmax < skip_threshold_) {
    stats.skipped = true;
    std::fill(jmod_out_.begin(), jmod_out_.end(), 0.0f);
    std::fill(jcur_out_.begin(), jcur_out_.end(), 0.0f);
    last_frame_skipped_ = true;
    hall_frame_available_ = true;
    clear_hall_readout_to_zeros();
    ++step_;
    return stats;
  }

  if (world_.transport_enabled()) {
    if (magnetization_fm_stack == nullptr) {
      throw std::runtime_error(
          "PoissonCudaSession: magnetization is required when AMR/AHE transport is enabled");
    }
    world_.set_magnetization_fm_stack(*magnetization_fm_stack);
    world_.rebuild_transport_operators();
    // Unknown count is fixed by geometry; operator values change with m.
    solver_.upload_spd_operator(world_);
    if (world_.ahe_enabled()) {
      solver_.upload_skew_operator(world_);
    } else {
      solver_.clear_skew_operator();
    }
  } else if (magnetization_fm_stack != nullptr) {
    // Magnetization is accepted but ignored on the scalar path.
  }

  solver_.set_max_iterations(first_solve_ ? initial_max_iterations(max_iterations_)
                                          : max_iterations_);
  first_solve_ = false;

  const auto t0 = std::chrono::steady_clock::now();
  PcgResult result;
  double picard_error = 0.0;
  int picard_sweeps_used = 0;
  if (world_.transport_enabled() && world_.ahe_enabled()) {
    world_.build_rhs_spd(applied_, rhs_s_);
    world_.build_rhs_skew(applied_, rhs_k_);
    PicardResult picard = solver_.solve_picard(
        rhs_s_, rhs_k_, transport_config_.picard_sweeps, transport_config_.picard_tolerance, x_);
    result = picard.final_pcg;
    stats.stats_note = picard.note;
    picard_error = picard.picard_error;
    picard_sweeps_used = picard.picard_sweeps_used;
    rhs_ = rhs_s_;  // for optional finite checks of primary rhs
  } else {
    world_.build_rhs_spd(applied_, rhs_);
    result = solver_.solve(rhs_, x_);
    if (world_.amr_enabled()) {
      stats.stats_note = "amr pcg_err=" + std::to_string(result.residual_relative);
    } else {
      stats.stats_note = "pcg_err=" + std::to_string(result.residual_relative);
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (result.numerical_failure) {
    throw std::runtime_error("PoissonCudaSession: CUDA PCG numerical failure");
  }

  stats.iterations = result.iterations;
  stats.residual_initial = result.initial_residual_max_norm;
  stats.residual = result.residual_max_norm;
  stats.rhs_inf = result.rhs_inf_norm;
  stats.residual_rel = result.residual_relative;
  stats.pcg_error = result.residual_relative;
  stats.pcg_converged = result.converged;
  stats.picard_error = picard_error;
  stats.picard_sweeps_used = picard_sweeps_used;
  stats.elapsed_s = std::chrono::duration<double>(t1 - t0).count();

  world_.fill_full_phi(x_, applied_, phi_);
  assert_finite_active_poisson_data(rhs_, x_, phi_, step_);

  last_frame_skipped_ = false;
  hall_frame_available_ = true;
  update_hall_readout_from_phi();

  compute_j_raw_from_phi(world_, phi_, j_frame_);
  log_poisson_phi_j_region_stats(world_, phi_, j_frame_, step_, "after_compute_j");

  // FM-only torque current: zero Pt and scalar void/filler contributions.
  jcur_full_ = j_frame_;
  mask_jcur_fm_layers(world_, jcur_full_);
  extract_jmod_subframe(world_, jcur_full_, output_spec_, jcur_out_);

  apply_jmod_postprocess(world_, world_.decay_length(), j_frame_, pt_avg_xy_);
  log_poisson_phi_j_region_stats(world_, phi_, j_frame_, step_, "after_jmod_postprocess");

  extract_jmod_subframe(world_, j_frame_, output_spec_, jmod_out_);

  ++step_;
  return stats;
}

void PoissonCudaSession::set_hall_probe_indices(HallProbeIndices probes) {
  validate_hall_probe_indices(probes, world_.cell_count());
  hall_probes_ = std::move(probes);
  hall_configured_ = true;
  if (hall_frame_available_) {
    if (last_frame_skipped_) {
      clear_hall_readout_to_zeros();
    } else {
      update_hall_readout_from_phi();
    }
  }
}

void PoissonCudaSession::update_hall_readout_from_phi() {
  if (!hall_configured_) {
    return;
  }
  hall_components_ = compute_hall_potentials(phi_, hall_probes_);
  hall_voltages_ = hall_components_.voltages;
}

void PoissonCudaSession::clear_hall_readout_to_zeros() {
  if (!hall_configured_) {
    hall_voltages_.clear();
    hall_components_ = HallPotentialComponents{};
    return;
  }
  const std::size_t n = hall_probes_.high_y.size();
  hall_voltages_.assign(n, 0.0);
  hall_components_.voltages.assign(n, 0.0);
  hall_components_.high_y_means.assign(n, 0.0);
  hall_components_.low_y_means.assign(n, 0.0);
  hall_components_.high_y_counts.assign(n, 0);
  hall_components_.low_y_counts.assign(n, 0);
  for (std::size_t c = 0; c < n; ++c) {
    hall_components_.high_y_counts[c] = hall_probes_.high_y[c].size();
    hall_components_.low_y_counts[c] = hall_probes_.low_y[c].size();
  }
}

const std::vector<double>& PoissonCudaSession::hall_potentials() const {
  if (!hall_configured_) {
    throw std::runtime_error("PoissonCudaSession: Hall probe indices are not configured");
  }
  if (!hall_frame_available_) {
    throw std::runtime_error(
        "PoissonCudaSession: hall_potentials() requires at least one iterate() call");
  }
  return hall_voltages_;
}

HallPotentialComponents PoissonCudaSession::hall_potential_components() const {
  if (!hall_configured_) {
    throw std::runtime_error("PoissonCudaSession: Hall probe indices are not configured");
  }
  if (!hall_frame_available_) {
    throw std::runtime_error(
        "PoissonCudaSession: hall_potential_components() requires at least one iterate() call");
  }
  return hall_components_;
}

#include "poisson_cuda_session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "poisson_magnetization_cuda.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point t0, Clock::time_point t1) {
  return std::chrono::duration<double>(t1 - t0).count();
}

bool poisson_debug_finite_enabled() {
  const char* v = std::getenv("POISSON_DEBUG_FINITE");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool poisson_gmres_force_host_rebuild() {
  const char* v = std::getenv("POISSON_GMRES_FORCE_HOST_REBUILD");
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

void check_cuda_session(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
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
                                       TransportConfig transport,
                                       PoissonLinearSolverKind solver_kind,
                                       int gmres_restart)
    : world_(std::move(world)),
      potentials_(std::move(potentials)),
      output_spec_(make_jmod_output_spec(world_, slice_x, slice_y, slice_z)),
      pcg_solver_(world_),
      gmres_solver_(world_),
      transport_config_(transport),
      solver_kind_(solver_kind),
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
  if (world_.transport_enabled() && solver_kind_ == PoissonLinearSolverKind::kGmresCusparse) {
    // Fixed maximal CSR pattern from geometry; per-step device update fills values.
    world_.build_transport_pattern_operators();
    gmres_solver_.upload_transport_operator(world_);
    gmres_solver_.prepare_transport_update(world_);
  } else {
    // Host operator may have been rebuilt by set_transport_config; keep GPU in sync.
    pcg_solver_.upload_spd_operator(world_);
  }
  pcg_solver_.set_tolerance(tolerance_);
  pcg_solver_.set_tolerance_check_batches(cuda_tol_batch_first, cuda_tol_batch_next);
  gmres_solver_.set_tolerance(tolerance_);
  gmres_solver_.set_restart(gmres_restart);
}

PoissonCudaSession::~PoissonCudaSession() {
  cudaFree(d_map_lo_);
  cudaFree(d_map_hi_);
  cudaFree(d_map_weight_);
}

void PoissonCudaSession::ensure_device_magnetization_mapping(
    const std::vector<int>& src_lo,
    const std::vector<int>& src_hi,
    const std::vector<float>& weight_hi,
    bool average_z) {
  const int dst_nz = world_.fm_layer_count();
  if (map_tables_ready_ && map_dst_nz_ == dst_nz && map_average_z_ == average_z) {
    return;
  }
  cudaFree(d_map_lo_);
  cudaFree(d_map_hi_);
  cudaFree(d_map_weight_);
  d_map_lo_ = nullptr;
  d_map_hi_ = nullptr;
  d_map_weight_ = nullptr;
  map_tables_ready_ = false;
  map_dst_nz_ = dst_nz;
  map_average_z_ = average_z;
  if (!average_z) {
    if (static_cast<int>(src_lo.size()) != dst_nz ||
        static_cast<int>(src_hi.size()) != dst_nz ||
        static_cast<int>(weight_hi.size()) != dst_nz) {
      throw std::runtime_error("device magnetization mapping arrays must match FM layer count");
    }
    check_cuda_session(cudaMalloc(&d_map_lo_, static_cast<std::size_t>(dst_nz) * sizeof(int)),
                       "cudaMalloc magnetization map lo");
    check_cuda_session(cudaMalloc(&d_map_hi_, static_cast<std::size_t>(dst_nz) * sizeof(int)),
                       "cudaMalloc magnetization map hi");
    check_cuda_session(
        cudaMalloc(&d_map_weight_, static_cast<std::size_t>(dst_nz) * sizeof(float)),
        "cudaMalloc magnetization map weight");
    check_cuda_session(
        cudaMemcpy(d_map_lo_, src_lo.data(), static_cast<std::size_t>(dst_nz) * sizeof(int),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy magnetization map lo");
    check_cuda_session(
        cudaMemcpy(d_map_hi_, src_hi.data(), static_cast<std::size_t>(dst_nz) * sizeof(int),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy magnetization map hi");
    check_cuda_session(
        cudaMemcpy(d_map_weight_, weight_hi.data(),
                   static_cast<std::size_t>(dst_nz) * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy magnetization map weight");
  }
  map_tables_ready_ = true;
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
  pcg_solver_.reset_solution();
  gmres_solver_.reset_solution();
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

StepStats PoissonCudaSession::iterate_with_magnetization_device(
    const float* d_mx,
    const float* d_my,
    const float* d_mz,
    int src_nz,
    int src_ny,
    int src_nx,
    const std::vector<int>& src_lo,
    const std::vector<int>& src_hi,
    const std::vector<float>& weight_hi,
    bool average_z) {
  // xy shape is fixed by the mumax Variable + Poisson world; validate once here.
  if (src_ny != world_.ny() || src_nx != world_.nx()) {
    throw std::runtime_error("device magnetization xy shape does not match Poisson world");
  }
  if (src_nz <= 0) {
    throw std::runtime_error("device magnetization src_nz must be > 0");
  }

  if (world_.transport_enabled() && solver_kind_ == PoissonLinearSolverKind::kGmresCusparse) {
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
    const auto iter_t0 = Clock::now();

    if (skip_threshold_ > 0.0 && vmax < skip_threshold_) {
      stats.skipped = true;
      std::fill(jmod_out_.begin(), jmod_out_.end(), 0.0f);
      std::fill(jcur_out_.begin(), jcur_out_.end(), 0.0f);
      last_frame_skipped_ = true;
      hall_frame_available_ = true;
      clear_hall_readout_to_zeros();
      stats.timing_total_s = seconds_between(iter_t0, Clock::now());
      ++step_;
      return stats;
    }

    const auto t_map0 = Clock::now();
    ensure_device_magnetization_mapping(src_lo, src_hi, weight_hi, average_z);
    float* d_mag = gmres_solver_.device_magnetization();
    if (d_mag == nullptr) {
      throw std::runtime_error("GMRES device magnetization buffer is not prepared");
    }
    map_device_magnetization_to_device_stack(d_mx, d_my, d_mz, src_nz, src_ny, src_nx,
                                             world_.fm_layer_count(), d_map_lo_, d_map_hi_,
                                             d_map_weight_, average_z, d_mag);
    std::vector<float> magnetization_fm_stack;
    gmres_solver_.copy_magnetization_device_to_host(magnetization_fm_stack);
    stats.timing_device_magnetization_s = seconds_between(t_map0, Clock::now());

    // Host tensors required for CPU jmod/jcur extraction (AMR/AHE Ohm's law).
    const auto t_mset0 = Clock::now();
    world_.set_magnetization_fm_stack(magnetization_fm_stack);
    world_.refresh_transport_tensors();
    stats.timing_magnetization_set_s = seconds_between(t_mset0, Clock::now());

    gmres_solver_.set_max_iterations(first_solve_ ? initial_max_iterations(max_iterations_)
                                                  : max_iterations_);
    first_solve_ = false;

    const auto t0 = Clock::now();
    const auto t_update0 = Clock::now();
    gmres_solver_.update_transport_operator_and_rhs_device(world_, applied_);
    stats.timing_transport_s = seconds_between(t_update0, Clock::now());
    stats.timing_rhs_build_s = 0.0;
    stats.timing_transport_rebuild_s = 0.0;
    stats.timing_operator_upload_s = 0.0;
    const auto t_solve0 = Clock::now();
    PcgResult result = gmres_solver_.solve_device_rhs(x_);
    stats.timing_linear_solve_s = seconds_between(t_solve0, Clock::now());
    stats.stats_note =
        "gmres_cusparse_jacobi_device_update err=" + std::to_string(result.residual_relative);
    const auto t1 = Clock::now();
    stats.timing_total_s = seconds_between(iter_t0, t1);
    return finish_iterate_after_solve(stats, result, 0.0, 0, seconds_between(t0, t1));
  }

  std::vector<float> magnetization_fm_stack;
  const auto t_map0 = Clock::now();
  map_device_magnetization_to_host_stack(d_mx, d_my, d_mz, src_nz, src_ny, src_nx,
                                         world_.fm_layer_count(), src_lo, src_hi, weight_hi,
                                         average_z, magnetization_fm_stack);
  return iterate_impl(&magnetization_fm_stack, seconds_between(t_map0, Clock::now()));
}

StepStats PoissonCudaSession::iterate_impl(const std::vector<float>* magnetization_fm_stack,
                                           double timing_device_magnetization_s) {
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
  stats.timing_device_magnetization_s = timing_device_magnetization_s;
  const auto iter_t0 = Clock::now();

  if (skip_threshold_ > 0.0 && vmax < skip_threshold_) {
    stats.skipped = true;
    std::fill(jmod_out_.begin(), jmod_out_.end(), 0.0f);
    std::fill(jcur_out_.begin(), jcur_out_.end(), 0.0f);
    last_frame_skipped_ = true;
    hall_frame_available_ = true;
    clear_hall_readout_to_zeros();
    stats.timing_total_s = seconds_between(iter_t0, Clock::now());
    ++step_;
    return stats;
  }

  if (world_.transport_enabled()) {
    const auto t_transport0 = Clock::now();
    if (magnetization_fm_stack == nullptr) {
      throw std::runtime_error(
          "PoissonCudaSession: magnetization is required when AMR/AHE transport is enabled");
    }
    const auto t_mset0 = Clock::now();
    world_.set_magnetization_fm_stack(*magnetization_fm_stack);
    stats.timing_magnetization_set_s = seconds_between(t_mset0, Clock::now());

    if (solver_kind_ == PoissonLinearSolverKind::kGmresCusparse &&
        !poisson_gmres_force_host_rebuild()) {
      // Keep host tensors for CPU jmod/jcur; update matrix values on the fixed GPU pattern.
      const auto t_refresh0 = Clock::now();
      world_.refresh_transport_tensors();
      stats.timing_magnetization_set_s += seconds_between(t_refresh0, Clock::now());

      pcg_solver_.set_max_iterations(first_solve_ ? initial_max_iterations(max_iterations_)
                                                  : max_iterations_);
      gmres_solver_.set_max_iterations(first_solve_ ? initial_max_iterations(max_iterations_)
                                                    : max_iterations_);
      first_solve_ = false;

      const auto t0 = Clock::now();
      const auto t_update0 = Clock::now();
      gmres_solver_.update_transport_operator_and_rhs_device(world_, *magnetization_fm_stack,
                                                             applied_);
      stats.timing_transport_s = seconds_between(t_update0, Clock::now());
      stats.timing_transport_rebuild_s = 0.0;
      stats.timing_operator_upload_s = 0.0;
      stats.timing_rhs_build_s = 0.0;
      const auto t_solve0 = Clock::now();
      PcgResult result = gmres_solver_.solve_device_rhs(x_);
      stats.timing_linear_solve_s = seconds_between(t_solve0, Clock::now());
      stats.stats_note =
          "gmres_cusparse_jacobi_device_update err=" + std::to_string(result.residual_relative);
      const auto t1 = Clock::now();
      stats.timing_total_s = seconds_between(iter_t0, t1);
      return finish_iterate_after_solve(stats, result, 0.0, 0, seconds_between(t0, t1));
    }

    const auto t_rebuild0 = Clock::now();
    world_.rebuild_transport_operators();
    stats.timing_transport_rebuild_s = seconds_between(t_rebuild0, Clock::now());
    // Unknown count is fixed by geometry; operator values change with m.
    const auto t_upload0 = Clock::now();
    if (solver_kind_ == PoissonLinearSolverKind::kGmresCusparse) {
      gmres_solver_.upload_transport_operator(world_);
    } else {
      pcg_solver_.upload_spd_operator(world_);
      if (world_.ahe_enabled()) {
        pcg_solver_.upload_skew_operator(world_);
      } else {
        pcg_solver_.clear_skew_operator();
      }
    }
    stats.timing_operator_upload_s = seconds_between(t_upload0, Clock::now());
    stats.timing_transport_s = seconds_between(t_transport0, Clock::now());
  } else if (magnetization_fm_stack != nullptr) {
    // Magnetization is accepted but ignored on the scalar path.
  }

  pcg_solver_.set_max_iterations(first_solve_ ? initial_max_iterations(max_iterations_)
                                              : max_iterations_);
  gmres_solver_.set_max_iterations(first_solve_ ? initial_max_iterations(max_iterations_)
                                                : max_iterations_);
  first_solve_ = false;

  const auto t0 = Clock::now();
  PcgResult result;
  double picard_error = 0.0;
  int picard_sweeps_used = 0;
  if (world_.transport_enabled() && solver_kind_ == PoissonLinearSolverKind::kGmresCusparse) {
    const auto t_rhs0 = Clock::now();
    world_.build_rhs_spd(applied_, rhs_);
    if (world_.ahe_enabled()) {
      world_.build_rhs_skew(applied_, rhs_k_);
      for (std::size_t i = 0; i < rhs_.size(); ++i) {
        rhs_[i] += rhs_k_[i];
      }
    }
    stats.timing_rhs_build_s = seconds_between(t_rhs0, Clock::now());
    const auto t_solve0 = Clock::now();
    result = gmres_solver_.solve(rhs_, x_);
    stats.timing_linear_solve_s = seconds_between(t_solve0, Clock::now());
    stats.stats_note = "gmres_cusparse_jacobi err=" + std::to_string(result.residual_relative);
  } else if (world_.transport_enabled() && world_.ahe_enabled()) {
    const auto t_rhs0 = Clock::now();
    world_.build_rhs_spd(applied_, rhs_s_);
    world_.build_rhs_skew(applied_, rhs_k_);
    stats.timing_rhs_build_s = seconds_between(t_rhs0, Clock::now());
    const auto t_solve0 = Clock::now();
    PicardResult picard = pcg_solver_.solve_picard(
        rhs_s_, rhs_k_, transport_config_.picard_sweeps, transport_config_.picard_tolerance, x_);
    stats.timing_linear_solve_s = seconds_between(t_solve0, Clock::now());
    result = picard.final_pcg;
    stats.stats_note = picard.note;
    picard_error = picard.picard_error;
    picard_sweeps_used = picard.picard_sweeps_used;
    rhs_ = rhs_s_;  // for optional finite checks of primary rhs
  } else {
    const auto t_rhs0 = Clock::now();
    world_.build_rhs_spd(applied_, rhs_);
    stats.timing_rhs_build_s = seconds_between(t_rhs0, Clock::now());
    const auto t_solve0 = Clock::now();
    result = pcg_solver_.solve(rhs_, x_);
    stats.timing_linear_solve_s = seconds_between(t_solve0, Clock::now());
    if (world_.amr_enabled()) {
      stats.stats_note = "amr pcg_err=" + std::to_string(result.residual_relative);
    } else {
      stats.stats_note = "pcg_err=" + std::to_string(result.residual_relative);
    }
  }
  const auto t1 = Clock::now();
  stats.timing_total_s = seconds_between(iter_t0, t1);
  return finish_iterate_after_solve(stats, result, picard_error, picard_sweeps_used,
                                    seconds_between(t0, t1));
}

StepStats PoissonCudaSession::finish_iterate_after_solve(StepStats stats,
                                                         const PcgResult& result,
                                                         double picard_error,
                                                         int picard_sweeps_used,
                                                         double elapsed_s) {
  const auto finish_t0 = Clock::now();
  if (result.numerical_failure) {
    throw std::runtime_error("PoissonCudaSession: CUDA linear solver numerical failure");
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
  stats.elapsed_s = elapsed_s;

  const auto t_phi0 = Clock::now();
  world_.fill_full_phi(x_, applied_, phi_);
  stats.timing_fill_phi_s = seconds_between(t_phi0, Clock::now());
  assert_finite_active_poisson_data(rhs_, x_, phi_, step_);

  last_frame_skipped_ = false;
  hall_frame_available_ = true;
  const auto t_hall0 = Clock::now();
  update_hall_readout_from_phi();
  stats.timing_hall_s = seconds_between(t_hall0, Clock::now());

  const auto t_jraw0 = Clock::now();
  compute_j_raw_from_phi(world_, phi_, j_frame_);
  stats.timing_j_raw_s = seconds_between(t_jraw0, Clock::now());
  log_poisson_phi_j_region_stats(world_, phi_, j_frame_, step_, "after_compute_j");

  // FM-only torque current: zero Pt and scalar void/filler contributions.
  const auto t_jcur0 = Clock::now();
  jcur_full_ = j_frame_;
  mask_jcur_fm_layers(world_, jcur_full_);
  extract_jmod_subframe(world_, jcur_full_, output_spec_, jcur_out_);
  stats.timing_jcur_extract_s = seconds_between(t_jcur0, Clock::now());

  const auto t_jmod_post0 = Clock::now();
  apply_jmod_postprocess(world_, world_.decay_length(), j_frame_, pt_avg_xy_);
  stats.timing_jmod_postprocess_s = seconds_between(t_jmod_post0, Clock::now());
  log_poisson_phi_j_region_stats(world_, phi_, j_frame_, step_, "after_jmod_postprocess");

  const auto t_jmod_extract0 = Clock::now();
  extract_jmod_subframe(world_, j_frame_, output_spec_, jmod_out_);
  stats.timing_jmod_extract_s = seconds_between(t_jmod_extract0, Clock::now());
  stats.timing_total_s += seconds_between(finish_t0, Clock::now());

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

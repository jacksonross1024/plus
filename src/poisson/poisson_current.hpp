#pragma once

#include <cstddef>
#include <string>
#include <vector>

class PoissonWorld;

struct JmodOutputSpec {
  int ix0 = 0;
  int ix1 = 0;
  int iy0 = 0;
  int iy1 = 0;
  int iz0 = 0;
  int iz1 = 0;

  int out_nx() const { return ix1 - ix0; }
  int out_ny() const { return iy1 - iy0; }
  int out_nz() const { return iz1 - iz0; }
  std::size_t frame_elements() const {
    return static_cast<std::size_t>(out_nz()) * static_cast<std::size_t>(out_ny()) *
           static_cast<std::size_t>(out_nx()) * 3u;
  }
};

struct StepStats {
  int step = 0;
  bool skipped = false;
  int iterations = 0;
  double residual_initial = 0.0;
  double residual = 0.0;
  double rhs_inf = 0.0;
  double residual_rel = 0.0;
  /// Alias for the last inner PCG relative residual (same as residual_rel).
  double pcg_error = 0.0;
  bool pcg_converged = false;
  /// Picard iterate change ‖xⁿ − xⁿ⁻¹‖_∞ (0 when Picard is not used).
  double picard_error = 0.0;
  int picard_sweeps_used = 0;
  double elapsed_s = 0.0;
  double timing_total_s = 0.0;
  double timing_device_magnetization_s = 0.0;
  double timing_transport_s = 0.0;
  double timing_magnetization_set_s = 0.0;
  double timing_transport_rebuild_s = 0.0;
  double timing_operator_upload_s = 0.0;
  double timing_rhs_build_s = 0.0;
  double timing_linear_solve_s = 0.0;
  double timing_fill_phi_s = 0.0;
  double timing_hall_s = 0.0;
  double timing_j_raw_s = 0.0;
  double timing_jcur_extract_s = 0.0;
  double timing_jmod_postprocess_s = 0.0;
  double timing_jmod_extract_s = 0.0;
  std::string stats_note;
};

JmodOutputSpec make_jmod_output_spec(const PoissonWorld& world,
                                     const std::string& slice_x,
                                     const std::string& slice_y,
                                     const std::string& slice_z);
void extract_jmod_subframe(const PoissonWorld& world,
                           const std::vector<float>& full_frame,
                           const JmodOutputSpec& spec,
                           std::vector<float>& out_frame);
void compute_j_raw_from_phi(const PoissonWorld& world,
                            const std::vector<float>& phi,
                            std::vector<float>& j_frame);
/// Zeros non-FM cells in a full-grid **J** frame. Do not call before ``apply_jmod_postprocess``
/// (that needs Pt-region **J** intact).
void mask_jcur_fm_layers(const PoissonWorld& world, std::vector<float>& j_raw);
/// Exponential decay factor for FM layer ``k`` (0 = at Pt interface), using the
/// arithmetic mean of ``exp(-z/λ)`` at the cell bottom, midpoint, and top.
double fm_injection_decay_factor(int fm_layer_index, double cz, double decay_length);

void apply_jmod_postprocess(const PoissonWorld& world,
                            double decay_length,
                            std::vector<float>& j_frame,
                            std::vector<float>& pt_avg_xy);
/// Log φ and **J** stats for Pt (HM) vs FM when ``POISSON_DEBUG_PHI`` is set.
void log_poisson_phi_j_region_stats(const PoissonWorld& world,
                                    const std::vector<float>& phi,
                                    const std::vector<float>& j_frame,
                                    int step,
                                    const char* stage_label);

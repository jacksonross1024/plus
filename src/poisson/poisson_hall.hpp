#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

/// Per-contact virtual Hall-bar voltage probes on the high/low y edges.
struct HallProbeIndices {
  std::vector<std::vector<int>> high_y;  // one index list per contact channel
  std::vector<std::vector<int>> low_y;
};

struct HallPotentialComponents {
  std::vector<double> voltages;  // high_y_mean - low_y_mean
  std::vector<double> high_y_means;
  std::vector<double> low_y_means;
  std::vector<std::size_t> high_y_counts;
  std::vector<std::size_t> low_y_counts;
};

inline void validate_hall_probe_indices(const HallProbeIndices& probes, int cell_count) {
  if (probes.high_y.size() != probes.low_y.size()) {
    throw std::invalid_argument("Hall high_y and low_y probe lists must have equal length");
  }
  if (probes.high_y.empty()) {
    throw std::invalid_argument("Hall probe lists cannot be empty");
  }
  for (std::size_t c = 0; c < probes.high_y.size(); ++c) {
    if (probes.high_y[c].empty() || probes.low_y[c].empty()) {
      throw std::invalid_argument("Hall probe index list for contact " + std::to_string(c + 1) +
                                  " is empty");
    }
    for (int idx : probes.high_y[c]) {
      if (idx < 0 || idx >= cell_count) {
        throw std::invalid_argument("Hall high_y flat index out of range");
      }
    }
    for (int idx : probes.low_y[c]) {
      if (idx < 0 || idx >= cell_count) {
        throw std::invalid_argument("Hall low_y flat index out of range");
      }
    }
  }
}

inline double mean_phi_at_indices(const std::vector<float>& phi, const std::vector<int>& indices) {
  if (indices.empty()) {
    throw std::runtime_error("mean_phi_at_indices: empty index list");
  }
  double sum = 0.0;
  for (int idx : indices) {
    sum += static_cast<double>(phi[static_cast<std::size_t>(idx)]);
  }
  return sum / static_cast<double>(indices.size());
}

inline HallPotentialComponents compute_hall_potentials(const std::vector<float>& phi,
                                                       const HallProbeIndices& probes) {
  const std::size_t n = probes.high_y.size();
  HallPotentialComponents out;
  out.voltages.assign(n, 0.0);
  out.high_y_means.assign(n, 0.0);
  out.low_y_means.assign(n, 0.0);
  out.high_y_counts.assign(n, 0);
  out.low_y_counts.assign(n, 0);
  for (std::size_t c = 0; c < n; ++c) {
    const double high = mean_phi_at_indices(phi, probes.high_y[c]);
    const double low = mean_phi_at_indices(phi, probes.low_y[c]);
    out.high_y_means[c] = high;
    out.low_y_means[c] = low;
    out.high_y_counts[c] = probes.high_y[c].size();
    out.low_y_counts[c] = probes.low_y[c].size();
    out.voltages[c] = high - low;  // high_y_minus_low_y
  }
  return out;
}

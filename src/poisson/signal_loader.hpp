#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct ContactPotentials {
  std::vector<std::vector<float>> channels;

  std::size_t num_contacts() const { return channels.size(); }
  std::size_t size() const { return channels.empty() ? 0u : channels.front().size(); }
  /// Fill ``out`` with contact potentials at ``step`` (reuses caller buffer).
  void fill_at(std::size_t step, std::vector<double>& out) const {
    out.resize(channels.size());
    for (std::size_t c = 0; c < channels.size(); ++c) {
      out[c] = static_cast<double>(channels[c][step]);
    }
  }
  std::vector<double> at(std::size_t step) const {
    std::vector<double> out;
    fill_at(step, out);
    return out;
  }
};

ContactPotentials load_signal_file_to_contact_potentials(const std::string& path,
                                                         int nt,
                                                         double v_scale,
                                                         int skip_first,
                                                         int num_contacts);

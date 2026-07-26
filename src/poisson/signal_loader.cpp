#include "signal_loader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim_copy(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::vector<double> read_single_column_signal_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("unable to open signal file: " + path);
  }

  std::vector<double> values;
  std::string line;
  int line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    line = trim_copy(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    double v = 0.0;
    if (!(iss >> v)) {
      throw std::runtime_error("unable to parse number on line " + std::to_string(line_number));
    }
    double extra = 0.0;
    if (iss >> extra) {
      throw std::runtime_error("signal file must be a single column: " + path);
    }
    values.push_back(v);
  }
  if (values.empty()) {
    throw std::runtime_error("signal file has no numeric data lines: " + path);
  }
  return values;
}

std::vector<double> resample_segment_to_nt(const std::vector<double>& segment, int nt) {
  const int length = static_cast<int>(segment.size());
  if (length == 0) {
    throw std::runtime_error("empty signal segment after split");
  }
  if (length == nt) {
    return segment;
  }

  std::vector<double> out(static_cast<std::size_t>(nt), 0.0);
  if (nt == 1 || length == 1) {
    std::fill(out.begin(), out.end(), segment.front());
    return out;
  }

  for (int i = 0; i < nt; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(nt - 1);
    const double src = u * static_cast<double>(length - 1);
    const int left = static_cast<int>(std::floor(src));
    const int right = std::min(left + 1, length - 1);
    const double frac = src - static_cast<double>(left);
    out[static_cast<std::size_t>(i)] =
        (1.0 - frac) * segment[static_cast<std::size_t>(left)] +
        frac * segment[static_cast<std::size_t>(right)];
  }
  return out;
}

void apply_global_amplitude_scale(std::vector<double>& raw, double v_scale) {
  double abs_max = 0.0;
  for (double sample : raw) {
    abs_max = std::max(abs_max, std::abs(sample));
  }
  if (abs_max == 0.0) {
    std::fill(raw.begin(), raw.end(), 0.0);
    return;
  }
  const double scale = v_scale / abs_max;
  for (double& sample : raw) {
    sample *= scale;
  }
}

}  // namespace

ContactPotentials load_signal_file_to_contact_potentials(const std::string& path,
                                                         int nt,
                                                         double v_scale,
                                                         int skip_first,
                                                         int num_contacts) {
  if (nt <= 0) {
    throw std::runtime_error("nt must be > 0");
  }
  if (skip_first < 0) {
    throw std::runtime_error("skip_first must be >= 0");
  }
  if (num_contacts <= 0) {
    throw std::runtime_error("num_contacts must be > 0");
  }

  auto raw = read_single_column_signal_file(path);
  if (skip_first >= static_cast<int>(raw.size())) {
    throw std::runtime_error("skip_first removes the entire signal");
  }
  raw.erase(raw.begin(), raw.begin() + skip_first);
  if (raw.size() < static_cast<std::size_t>(num_contacts)) {
    throw std::runtime_error("need at least " + std::to_string(num_contacts) +
                             " samples for " + std::to_string(num_contacts) +
                             " contact segments");
  }
  for (double sample : raw) {
    if (!std::isfinite(sample)) {
      throw std::runtime_error("signal file contains NaN or Inf");
    }
  }
  apply_global_amplitude_scale(raw, v_scale);

  const int n = static_cast<int>(raw.size());
  ContactPotentials potentials;
  potentials.channels.resize(static_cast<std::size_t>(num_contacts));
  for (int c = 0; c < num_contacts; ++c) {
    const int start = (c * n) / num_contacts;
    const int end = ((c + 1) * n) / num_contacts;
    const auto segment =
        resample_segment_to_nt(std::vector<double>(raw.begin() + start, raw.begin() + end), nt);
    potentials.channels[static_cast<std::size_t>(c)].assign(segment.begin(), segment.end());
  }
  return potentials;
}

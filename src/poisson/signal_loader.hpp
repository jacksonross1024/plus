#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

struct ContactPotentials {
  std::vector<float> c0;
  std::vector<float> c1;
  std::vector<float> c2;

  std::size_t size() const { return c0.size(); }
  std::array<double, 3> at(std::size_t step) const {
    return {
        static_cast<double>(c0.at(step)),
        static_cast<double>(c1.at(step)),
        static_cast<double>(c2.at(step)),
    };
  }
};

ContactPotentials load_signal_file_to_contact_potentials(const std::string& path,
                                                         int nt,
                                                         double v_scale,
                                                         int skip_first);

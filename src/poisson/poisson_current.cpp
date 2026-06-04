#include "poisson_current.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "poisson_world.hpp"

namespace {

std::size_t component_index(int cell, int comp) {
  return static_cast<std::size_t>(cell) * 3u + static_cast<std::size_t>(comp);
}

std::string trim_copy(const std::string& s) {
  std::size_t a = 0;
  std::size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) {
    ++a;
  }
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
    --b;
  }
  return s.substr(a, b - a);
}

std::pair<int, int> parse_axis_slice(const std::string& s, int dim, const char* axis_label) {
  const std::string t = trim_copy(s);
  if (t.empty() || t == ":") {
    return {0, dim};
  }

  const auto colon = t.find(':');
  if (colon == std::string::npos) {
    throw std::runtime_error(std::string("jmod slice for ") + axis_label +
                             " must contain ':'");
  }

  const std::string left = trim_copy(t.substr(0, colon));
  const std::string right = trim_copy(t.substr(colon + 1));
  const int start = left.empty() ? 0 : std::stoi(left);
  const int stop = right.empty() ? dim : std::stoi(right);
  if (start < 0 || stop > dim || start >= stop) {
    throw std::runtime_error("invalid jmod slice for " + std::string(axis_label));
  }
  return {start, stop};
}

}  // namespace

JmodOutputSpec make_jmod_output_spec(const PoissonWorld& world,
                                     const std::string& slice_x,
                                     const std::string& slice_y,
                                     const std::string& slice_z) {
  const auto xr = parse_axis_slice(slice_x, world.nx(), "x");
  const auto yr = parse_axis_slice(slice_y, world.ny(), "y");
  const auto zr = parse_axis_slice(slice_z, world.nz(), "z");
  return {xr.first, xr.second, yr.first, yr.second, zr.first, zr.second};
}

void extract_jmod_subframe(const PoissonWorld& world,
                           const std::vector<float>& full_frame,
                           const JmodOutputSpec& spec,
                           std::vector<float>& out_frame) {
  if (full_frame.size() != world.frame_elements()) {
    throw std::runtime_error("extract_jmod_subframe: full J frame size mismatch");
  }
  out_frame.resize(spec.frame_elements());
  std::size_t d = 0;
  for (int iz = spec.iz0; iz < spec.iz1; ++iz) {
    for (int iy = spec.iy0; iy < spec.iy1; ++iy) {
      for (int ix = spec.ix0; ix < spec.ix1; ++ix) {
        const int cell = world.flat_index(iz, iy, ix);
        for (int c = 0; c < 3; ++c) {
          out_frame[d++] = full_frame[component_index(cell, c)];
        }
      }
    }
  }
}

void compute_j_raw_from_phi(const PoissonWorld& world,
                            const std::vector<float>& phi,
                            std::vector<float>& j_frame) {
  if (phi.size() != static_cast<std::size_t>(world.cell_count())) {
    throw std::runtime_error("phi buffer size mismatch");
  }

  const int nx = world.nx();
  const int ny = world.ny();
  const int nz = world.nz();
  const float dx = static_cast<float>(world.cx());
  const float dy = static_cast<float>(world.cy());
  const float dz = static_cast<float>(world.cz());
  const auto& region = world.region();
  const auto& sigma = world.sigma();
  j_frame.assign(world.frame_elements(), 0.0f);

  auto phi_at = [&](int iz, int iy, int ix) -> float {
    return phi[static_cast<std::size_t>(world.flat_index(iz, iy, ix))];
  };

  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int cell = world.flat_index(iz, iy, ix);
        if (region[static_cast<std::size_t>(cell)] == 0) {
          continue;
        }

        float dphi_dx = 0.0f;
        if (nx > 1) {
          dphi_dx = (ix == 0) ? (phi_at(iz, iy, 1) - phi_at(iz, iy, 0)) / dx
                    : (ix == nx - 1)
                        ? (phi_at(iz, iy, nx - 1) - phi_at(iz, iy, nx - 2)) / dx
                        : (phi_at(iz, iy, ix + 1) - phi_at(iz, iy, ix - 1)) / (2.0f * dx);
        }

        float dphi_dy = 0.0f;
        if (ny > 1) {
          dphi_dy = (iy == 0) ? (phi_at(iz, 1, ix) - phi_at(iz, 0, ix)) / dy
                    : (iy == ny - 1)
                        ? (phi_at(iz, ny - 1, ix) - phi_at(iz, ny - 2, ix)) / dy
                        : (phi_at(iz, iy + 1, ix) - phi_at(iz, iy - 1, ix)) / (2.0f * dy);
        }

        float dphi_dz = 0.0f;
        if (nz > 1) {
          dphi_dz = (iz == 0) ? (phi_at(1, iy, ix) - phi_at(0, iy, ix)) / dz
                    : (iz == nz - 1)
                        ? (phi_at(nz - 1, iy, ix) - phi_at(nz - 2, iy, ix)) / dz
                        : (phi_at(iz + 1, iy, ix) - phi_at(iz - 1, iy, ix)) / (2.0f * dz);
        }

        const float local_sigma = sigma[static_cast<std::size_t>(cell)];
        j_frame[component_index(cell, 0)] = -local_sigma * dphi_dx;
        j_frame[component_index(cell, 1)] = -local_sigma * dphi_dy;
        j_frame[component_index(cell, 2)] = -local_sigma * dphi_dz;
      }
    }
  }
}

void mask_jcur_fm_layers(const PoissonWorld& world, std::vector<float>& j_raw) {
  if (j_raw.size() != world.frame_elements()) {
    throw std::runtime_error("mask_jcur_fm_layers: J frame size mismatch");
  }
  const auto& region = world.region();
  for (int cell = 0; cell < world.cell_count(); ++cell) {
    if (region[static_cast<std::size_t>(cell)] == 2) {
      continue;
    }
    for (int c = 0; c < 3; ++c) {
      j_raw[component_index(cell, c)] = 0.0f;
    }
  }
}

namespace {

struct RegionAccum {
  std::size_t count = 0;
  double phi_min = 0.0;
  double phi_max = 0.0;
  double phi_sum = 0.0;
  double grad_sum = 0.0;
  double grad_max = 0.0;
  double j_sum = 0.0;
  double j_max = 0.0;

  void add_phi(double v) {
    if (count == 0) {
      phi_min = phi_max = v;
    } else {
      phi_min = std::min(phi_min, v);
      phi_max = std::max(phi_max, v);
    }
    phi_sum += v;
    ++count;
  }

  void add_grad(double g) {
    grad_sum += g;
    grad_max = std::max(grad_max, g);
  }

  void add_j(double jmag) {
    j_sum += jmag;
    j_max = std::max(j_max, jmag);
  }
};

bool poisson_debug_phi_enabled() {
  const char* env = std::getenv("POISSON_DEBUG_PHI");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

}  // namespace

void log_poisson_phi_j_region_stats(const PoissonWorld& world,
                                    const std::vector<float>& phi,
                                    const std::vector<float>& j_frame,
                                    int step,
                                    const char* stage_label) {
  if (!poisson_debug_phi_enabled()) {
    return;
  }

  const int nx = world.nx();
  const int ny = world.ny();
  const int nz = world.nz();
  const float dx = static_cast<float>(world.cx());
  const float dy = static_cast<float>(world.cy());
  const float dz = static_cast<float>(world.cz());
  const auto& region = world.region();

  auto phi_at = [&](int iz, int iy, int ix) -> float {
    return phi[static_cast<std::size_t>(world.flat_index(iz, iy, ix))];
  };

  RegionAccum pt_all;
  RegionAccum fm_all;
  RegionAccum fm_interface;  // iz == first_r2_layer

  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int cell = world.flat_index(iz, iy, ix);
        const auto reg = region[static_cast<std::size_t>(cell)];
        if (reg == 0) {
          continue;
        }

        const float p = phi_at(iz, iy, ix);
        float dphi_dx = 0.0f;
        float dphi_dy = 0.0f;
        float dphi_dz = 0.0f;
        if (nx > 1) {
          dphi_dx = (ix == 0) ? (phi_at(iz, iy, 1) - phi_at(iz, iy, 0)) / dx
                    : (ix == nx - 1)
                        ? (phi_at(iz, iy, nx - 1) - phi_at(iz, iy, nx - 2)) / dx
                        : (phi_at(iz, iy, ix + 1) - phi_at(iz, iy, ix - 1)) / (2.0f * dx);
        }
        if (ny > 1) {
          dphi_dy = (iy == 0) ? (phi_at(iz, 1, ix) - phi_at(iz, 0, ix)) / dy
                    : (iy == ny - 1)
                        ? (phi_at(iz, ny - 1, ix) - phi_at(iz, ny - 2, ix)) / dy
                        : (phi_at(iz, iy + 1, ix) - phi_at(iz, iy - 1, ix)) / (2.0f * dy);
        }
        if (nz > 1) {
          dphi_dz = (iz == 0) ? (phi_at(1, iy, ix) - phi_at(0, iy, ix)) / dz
                    : (iz == nz - 1)
                        ? (phi_at(nz - 1, iy, ix) - phi_at(nz - 2, iy, ix)) / dz
                        : (phi_at(iz + 1, iy, ix) - phi_at(iz - 1, iy, ix)) / (2.0f * dz);
        }
        const double grad =
            std::sqrt(static_cast<double>(dphi_dx) * dphi_dx +
                      static_cast<double>(dphi_dy) * dphi_dy +
                      static_cast<double>(dphi_dz) * dphi_dz);
        const double jx = j_frame[component_index(cell, 0)];
        const double jy = j_frame[component_index(cell, 1)];
        const double jz = j_frame[component_index(cell, 2)];
        const double jmag = std::sqrt(jx * jx + jy * jy + jz * jz);

        if (reg == 1) {
          pt_all.add_phi(p);
          pt_all.add_grad(grad);
          pt_all.add_j(jmag);
        } else if (reg == 2) {
          fm_all.add_phi(p);
          fm_all.add_grad(grad);
          fm_all.add_j(jmag);
          if (iz == world.first_r2_layer()) {
            fm_interface.add_phi(p);
            fm_interface.add_grad(grad);
            fm_interface.add_j(jmag);
          }
        }
      }
    }
  }

  auto print_acc = [](const char* label, const RegionAccum& a) {
    const double n = static_cast<double>(std::max<std::size_t>(a.count, 1));
    std::fprintf(stderr,
                 "    %s: n=%zu phi[min,max,mean]=[%g,%g,%g] "
                 "|<grad phi>|_mean=%g max=%g <|J|>_mean=%g max=%g\n",
                 label, a.count, a.phi_min, a.phi_max, a.phi_sum / n, a.grad_sum / n,
                 a.grad_max, a.j_sum / n, a.j_max);
  };

  std::fprintf(stderr, "POISSON_DEBUG_PHI step=%d stage=%s\n", step, stage_label);
  print_acc("Pt(HM) all", pt_all);
  print_acc("FM all", fm_all);
  print_acc("FM iz=first_r2 (interface)", fm_interface);
}

double fm_injection_decay_factor(const int fm_layer_index,
                                 const double cz,
                                 const double decay_length) {
  if (!(decay_length > 0.0) || !(cz > 0.0)) {
    return 1.0;
  }
  const double z_bottom = static_cast<double>(fm_layer_index) * cz;
  const double z_mid = z_bottom + 0.5 * cz;
  const double z_top = z_bottom + cz;
  const double inv_lambda = 1.0 / decay_length;
  return (std::exp(-z_bottom * inv_lambda) + std::exp(-z_mid * inv_lambda) +
          std::exp(-z_top * inv_lambda)) /
         3.0;
}

void apply_jmod_postprocess(const PoissonWorld& world,
                            double decay_length,
                            std::vector<float>& j_frame,
                            std::vector<float>& pt_avg_xy) {
  if (j_frame.size() != world.frame_elements()) {
    throw std::runtime_error("J frame size mismatch in J_mod postprocess");
  }

  const int nx = world.nx();
  const int ny = world.ny();
  const int nz = world.nz();
  const auto& region = world.region();
  const auto& pt_counts = world.pt_column_counts();

  pt_avg_xy.assign(static_cast<std::size_t>(nx * ny) * 3u, 0.0f);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int cell = world.flat_index(iz, iy, ix);
        if (region[static_cast<std::size_t>(cell)] != 1) {
          continue;
        }
        const int xy = world.xy_index(iy, ix);
        for (int c = 0; c < 3; ++c) {
          pt_avg_xy[component_index(xy, c)] += j_frame[component_index(cell, c)];
        }
      }
    }
  }

  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const int xy = world.xy_index(iy, ix);
      const int count = pt_counts[static_cast<std::size_t>(xy)];
      if (count <= 0) {
        continue;
      }
      const float inv_count = 1.0f / static_cast<float>(count);
      for (int c = 0; c < 3; ++c) {
        pt_avg_xy[component_index(xy, c)] *= inv_count;
      }
    }
  }

  if (!(decay_length > 0.0)) {
    return;
  }

  for (int iz = world.first_r2_layer(); iz < nz; ++iz) {
    const int fm_layer = iz - world.first_r2_layer();
    const float factor = static_cast<float>(
        fm_injection_decay_factor(fm_layer, world.cz(), decay_length));
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int cell = world.flat_index(iz, iy, ix);
        if (region[static_cast<std::size_t>(cell)] != 2) {
          continue;
        }
        const int xy = world.xy_index(iy, ix);
        for (int c = 0; c < 3; ++c) {
          j_frame[component_index(cell, c)] = pt_avg_xy[component_index(xy, c)] * factor;
        }
      }
    }
  }
}

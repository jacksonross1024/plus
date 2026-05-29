#include "poisson_world.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "npy_reader.hpp"

namespace {

std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

bool in_bounds(int value, int limit) {
  return value >= 0 && value < limit;
}

double face_area(const ManifestData& meta, int axis) {
  if (axis == 0) {
    return meta.cy * meta.cz;
  }
  if (axis == 1) {
    return meta.cx * meta.cz;
  }
  return meta.cx * meta.cy;
}

double axis_spacing(const ManifestData& meta, int axis) {
  if (axis == 0) {
    return meta.cx;
  }
  if (axis == 1) {
    return meta.cy;
  }
  return meta.cz;
}

double dirichlet_potential(std::int8_t contact_id, const std::array<double, 3>& potentials) {
  const int channel = std::abs(static_cast<int>(contact_id)) - 1;
  const double sign = contact_id > 0 ? 1.0 : -1.0;
  if (channel < 0 || channel >= static_cast<int>(potentials.size())) {
    throw std::runtime_error("invalid contact id in Poisson world");
  }
  return sign * potentials[static_cast<std::size_t>(channel)];
}

template <typename T>
T require_parsed(const std::unordered_map<std::string, std::string>& values,
                 const std::string& key) {
  const auto it = values.find(key);
  if (it == values.end()) {
    throw std::runtime_error("missing manifest key: " + key);
  }
  std::stringstream ss(it->second);
  T out{};
  ss >> out;
  if (!ss || !ss.eof()) {
    throw std::runtime_error("invalid manifest value for key: " + key);
  }
  return out;
}

template <>
std::string require_parsed<std::string>(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key) {
  const auto it = values.find(key);
  if (it == values.end()) {
    throw std::runtime_error("missing manifest key: " + key);
  }
  return it->second;
}

void validate_sigma_finite(const std::vector<float>& sigma) {
  for (float value : sigma) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite value in Poisson sigma array");
    }
  }
}

}  // namespace

ManifestData PoissonWorld::parse_manifest(const std::string& manifest_path) {
  std::ifstream in(manifest_path);
  if (!in) {
    throw std::runtime_error("unable to open manifest: " + manifest_path);
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      throw std::runtime_error("invalid manifest line: " + line);
    }
    values[trim(line.substr(0, eq_pos))] = trim(line.substr(eq_pos + 1));
  }

  ManifestData meta;
  meta.nx = require_parsed<int>(values, "nx");
  meta.ny = require_parsed<int>(values, "ny");
  meta.nz = require_parsed<int>(values, "nz");
  meta.cx = require_parsed<double>(values, "cx");
  meta.cy = require_parsed<double>(values, "cy");
  meta.cz = require_parsed<double>(values, "cz");
  meta.first_r2_layer = require_parsed<int>(values, "first_r2_layer");
  meta.theta_sh = require_parsed<double>(values, "theta_sh");
  meta.decay_length = require_parsed<double>(values, "decay_length");
  meta.region_file = require_parsed<std::string>(values, "region_file");
  meta.contact_id_file = require_parsed<std::string>(values, "contact_id_file");
  meta.sigma_file = require_parsed<std::string>(values, "sigma_file");
  return meta;
}

PoissonWorld PoissonWorld::load(const std::string& manifest_path) {
  PoissonWorld world;
  world.meta_ = parse_manifest(manifest_path);

  const std::filesystem::path manifest_fs(manifest_path);
  const auto manifest_dir = manifest_fs.parent_path();
  const auto region = poisson_npy::read_npy_file<std::int8_t>(
      (manifest_dir / world.meta_.region_file).string());
  const auto contact_id = poisson_npy::read_npy_file<std::int8_t>(
      (manifest_dir / world.meta_.contact_id_file).string());
  const auto sigma = poisson_npy::read_npy_file<float>(
      (manifest_dir / world.meta_.sigma_file).string());

  const std::vector<std::size_t> expected_shape = {
      static_cast<std::size_t>(world.meta_.nz),
      static_cast<std::size_t>(world.meta_.ny),
      static_cast<std::size_t>(world.meta_.nx),
  };
  if (region.shape != expected_shape || contact_id.shape != expected_shape ||
      sigma.shape != expected_shape) {
    throw std::runtime_error("Poisson world array shape mismatch");
  }

  world.region_ = region.data;
  world.contact_id_ = contact_id.data;
  world.sigma_ = sigma.data;
  validate_sigma_finite(world.sigma_);
  world.validate_loaded_geometry();
  world.build_operator();
  return world;
}

PoissonWorld PoissonWorld::from_arrays(ManifestData meta,
                                       std::vector<std::int8_t> region,
                                       std::vector<std::int8_t> contact_id,
                                       std::vector<float> sigma) {
  if (meta.nx <= 0 || meta.ny <= 0 || meta.nz <= 0) {
    throw std::runtime_error("PoissonWorld::from_arrays: nx, ny, and nz must be > 0");
  }
  if (!(meta.cx > 0.0) || !(meta.cy > 0.0) || !(meta.cz > 0.0)) {
    throw std::runtime_error("PoissonWorld::from_arrays: cell sizes must be > 0");
  }
  if (meta.first_r2_layer < 0 || meta.first_r2_layer > meta.nz) {
    throw std::runtime_error("PoissonWorld::from_arrays: first_r2_layer outside grid");
  }

  const std::size_t expected_size = static_cast<std::size_t>(meta.nz) *
                                    static_cast<std::size_t>(meta.ny) *
                                    static_cast<std::size_t>(meta.nx);
  if (region.size() != expected_size || contact_id.size() != expected_size ||
      sigma.size() != expected_size) {
    throw std::runtime_error("PoissonWorld::from_arrays: array sizes do not match nz*ny*nx");
  }

  PoissonWorld world;
  world.meta_ = std::move(meta);
  world.region_ = std::move(region);
  world.contact_id_ = std::move(contact_id);
  world.sigma_ = std::move(sigma);
  validate_sigma_finite(world.sigma_);
  world.validate_loaded_geometry();
  world.build_operator();
  return world;
}

void PoissonWorld::validate_loaded_geometry() const {
  const int nx = meta_.nx;
  const int ny = meta_.ny;
  const int nz = meta_.nz;
  const int plane = nx * ny;
  constexpr float k_sigma_void_max = 1e-20f;

  for (int cell = 0; cell < cell_count(); ++cell) {
    const int iz = cell / plane;
    const auto idx = static_cast<std::size_t>(cell);
    const std::int8_t cid = contact_id_[idx];
    const std::int8_t reg = region_[idx];
    const float sig = sigma_[idx];

    if (reg < 0 || reg > 2) {
      throw std::runtime_error("region must be 0, 1 (Pt), or 2 (FM)");
    }
    if (reg == 0) {
      if (sig > k_sigma_void_max || cid != 0) {
        throw std::runtime_error("void cells must have sigma==0 and contact_id==0");
      }
    } else if (!(sig > 0.0f) || !std::isfinite(sig)) {
      throw std::runtime_error("conducting cells must have finite sigma>0");
    }

    if (reg == 1 && iz >= meta_.first_r2_layer) {
      throw std::runtime_error("region==1 (Pt) found at or above first_r2_layer");
    }
    if (reg == 2 && iz < meta_.first_r2_layer) {
      throw std::runtime_error("region==2 (FM) found below first_r2_layer");
    }
    if (cid != 0) {
      const int c = static_cast<int>(cid);
      if (c < -3 || c > 3 || reg != 1 || iz >= meta_.first_r2_layer) {
        throw std::runtime_error("contact_id must be +/-1..3 on Pt layers only");
      }
    }
  }
}

void PoissonWorld::build_operator() {
  const int n_cells = cell_count();
  pt_column_counts_.assign(static_cast<std::size_t>(meta_.nx * meta_.ny), 0);
  for (int iz = 0; iz < meta_.nz; ++iz) {
    for (int iy = 0; iy < meta_.ny; ++iy) {
      for (int ix = 0; ix < meta_.nx; ++ix) {
        const int cell = flat_index(iz, iy, ix);
        if (region_[static_cast<std::size_t>(cell)] == 1) {
          pt_column_counts_[static_cast<std::size_t>(xy_index(iy, ix))] += 1;
        }
      }
    }
  }

  unknown_index_.assign(static_cast<std::size_t>(n_cells), -1);
  unknown_to_cell_.clear();
  unknown_to_cell_.reserve(static_cast<std::size_t>(n_cells));
  for (int cell = 0; cell < n_cells; ++cell) {
    if (region_[static_cast<std::size_t>(cell)] > 0 &&
        contact_id_[static_cast<std::size_t>(cell)] == 0) {
      unknown_index_[static_cast<std::size_t>(cell)] = static_cast<int>(unknown_to_cell_.size());
      unknown_to_cell_.push_back(cell);
    }
  }

  const int n_unknown = unknown_count();
  diagonal_.assign(static_cast<std::size_t>(n_unknown), 0.0f);
  rhs_weight_c0_.assign(static_cast<std::size_t>(n_unknown), 0.0f);
  rhs_weight_c1_.assign(static_cast<std::size_t>(n_unknown), 0.0f);
  rhs_weight_c2_.assign(static_cast<std::size_t>(n_unknown), 0.0f);
  std::vector<int> row_counts(static_cast<std::size_t>(n_unknown), 0);

  constexpr std::array<std::array<int, 4>, 6> kNeighbors = {{
      {{-1, 0, 0, 0}}, {{1, 0, 0, 0}}, {{0, -1, 0, 1}},
      {{0, 1, 0, 1}},  {{0, 0, -1, 2}}, {{0, 0, 1, 2}},
  }};

  const int plane = meta_.nx * meta_.ny;
  for (int row = 0; row < n_unknown; ++row) {
    const int cell = unknown_to_cell_[static_cast<std::size_t>(row)];
    const int iz = cell / plane;
    const int rem = cell % plane;
    const int iy = rem / meta_.nx;
    const int ix = rem % meta_.nx;

    for (const auto& neighbor : kNeighbors) {
      const int nix = ix + neighbor[0];
      const int niy = iy + neighbor[1];
      const int niz = iz + neighbor[2];
      const int axis = neighbor[3];
      if (!in_bounds(nix, meta_.nx) || !in_bounds(niy, meta_.ny) ||
          !in_bounds(niz, meta_.nz)) {
        continue;
      }
      const int nbr = flat_index(niz, niy, nix);
      if (region_[static_cast<std::size_t>(nbr)] == 0) {
        continue;
      }
      const float sigma_here = sigma_[static_cast<std::size_t>(cell)];
      const float sigma_nbr = sigma_[static_cast<std::size_t>(nbr)];
      if (sigma_here <= 0.0f || sigma_nbr <= 0.0f) {
        continue;
      }
      const double conductance =
          face_area(meta_, axis) *
          std::sqrt(static_cast<double>(sigma_here) * static_cast<double>(sigma_nbr)) /
          axis_spacing(meta_, axis);
      if (!(conductance > 0.0)) {
        continue;
      }
      diagonal_[static_cast<std::size_t>(row)] += static_cast<float>(conductance);
      const int nbr_row = unknown_index_[static_cast<std::size_t>(nbr)];
      if (nbr_row >= 0) {
        row_counts[static_cast<std::size_t>(row)] += 1;
      } else if (contact_id_[static_cast<std::size_t>(nbr)] != 0) {
        const int channel = std::abs(static_cast<int>(contact_id_[static_cast<std::size_t>(nbr)])) - 1;
        const float signed_g =
            static_cast<float>((contact_id_[static_cast<std::size_t>(nbr)] > 0 ? 1.0 : -1.0) *
                               conductance);
        if (channel == 0) {
          rhs_weight_c0_[static_cast<std::size_t>(row)] += signed_g;
        } else if (channel == 1) {
          rhs_weight_c1_[static_cast<std::size_t>(row)] += signed_g;
        } else if (channel == 2) {
          rhs_weight_c2_[static_cast<std::size_t>(row)] += signed_g;
        }
      }
    }
    if (!(diagonal_[static_cast<std::size_t>(row)] > 0.0f)) {
      throw std::runtime_error("encountered isolated unknown cell in operator assembly");
    }
  }

  row_offsets_.assign(static_cast<std::size_t>(n_unknown + 1), 0);
  for (int row = 0; row < n_unknown; ++row) {
    row_offsets_[static_cast<std::size_t>(row + 1)] =
        row_offsets_[static_cast<std::size_t>(row)] + row_counts[static_cast<std::size_t>(row)];
  }
  col_indices_.assign(static_cast<std::size_t>(row_offsets_.back()), -1);
  offdiag_conductance_.assign(static_cast<std::size_t>(row_offsets_.back()), 0.0f);

  for (int row = 0; row < n_unknown; ++row) {
    const int cell = unknown_to_cell_[static_cast<std::size_t>(row)];
    const int iz = cell / plane;
    const int rem = cell % plane;
    const int iy = rem / meta_.nx;
    const int ix = rem % meta_.nx;
    int write_pos = row_offsets_[static_cast<std::size_t>(row)];

    for (const auto& neighbor : kNeighbors) {
      const int nix = ix + neighbor[0];
      const int niy = iy + neighbor[1];
      const int niz = iz + neighbor[2];
      const int axis = neighbor[3];
      if (!in_bounds(nix, meta_.nx) || !in_bounds(niy, meta_.ny) ||
          !in_bounds(niz, meta_.nz)) {
        continue;
      }
      const int nbr = flat_index(niz, niy, nix);
      const int nbr_row = unknown_index_[static_cast<std::size_t>(nbr)];
      if (nbr_row < 0 || region_[static_cast<std::size_t>(nbr)] == 0) {
        continue;
      }
      const float sigma_here = sigma_[static_cast<std::size_t>(cell)];
      const float sigma_nbr = sigma_[static_cast<std::size_t>(nbr)];
      if (sigma_here <= 0.0f || sigma_nbr <= 0.0f) {
        continue;
      }
      const double conductance =
          face_area(meta_, axis) *
          std::sqrt(static_cast<double>(sigma_here) * static_cast<double>(sigma_nbr)) /
          axis_spacing(meta_, axis);
      if (!(conductance > 0.0)) {
        continue;
      }
      col_indices_[static_cast<std::size_t>(write_pos)] = nbr_row;
      offdiag_conductance_[static_cast<std::size_t>(write_pos)] = static_cast<float>(conductance);
      write_pos += 1;
    }
  }
}

void PoissonWorld::build_rhs(const std::array<double, 3>& potentials,
                             std::vector<double>& rhs) const {
  rhs.resize(static_cast<std::size_t>(unknown_count()));
  for (int row = 0; row < unknown_count(); ++row) {
    const auto idx = static_cast<std::size_t>(row);
    rhs[idx] = static_cast<double>(rhs_weight_c0_[idx]) * potentials[0] +
               static_cast<double>(rhs_weight_c1_[idx]) * potentials[1] +
               static_cast<double>(rhs_weight_c2_[idx]) * potentials[2];
  }
}

void PoissonWorld::fill_full_phi(const std::vector<double>& x,
                                 const std::array<double, 3>& potentials,
                                 std::vector<float>& phi) const {
  if (static_cast<int>(x.size()) != unknown_count()) {
    throw std::runtime_error("fill_full_phi received vector of wrong size");
  }

  phi.assign(static_cast<std::size_t>(cell_count()), 0.0f);
  for (int cell = 0; cell < cell_count(); ++cell) {
    const auto idx = static_cast<std::size_t>(cell);
    if (region_[idx] == 0) {
      continue;
    }
    if (contact_id_[idx] != 0) {
      phi[idx] = static_cast<float>(dirichlet_potential(contact_id_[idx], potentials));
      continue;
    }
    const int row = unknown_index_[idx];
    if (row < 0) {
      throw std::runtime_error("active non-contact cell missing from unknown map");
    }
    phi[idx] = static_cast<float>(x[static_cast<std::size_t>(row)]);
  }
}

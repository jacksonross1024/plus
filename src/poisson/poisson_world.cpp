#include "poisson_world.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
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

double dirichlet_potential(std::int8_t contact_id, const std::vector<double>& potentials) {
  const int channel = std::abs(static_cast<int>(contact_id)) - 1;
  const double sign = contact_id > 0 ? 1.0 : -1.0;
#ifndef NDEBUG
  if (channel < 0 || channel >= static_cast<int>(potentials.size())) {
    throw std::runtime_error("invalid contact id in Poisson world");
  }
#endif
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

float avg_diag(float a, float b) {
  if (!(a > 0.0f) || !(b > 0.0f)) {
    return 0.0f;
  }
  return static_cast<float>(std::sqrt(static_cast<double>(a) * static_cast<double>(b)));
}

float avg_signed(float a, float b) {
  return 0.5f * (a + b);
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

void PoissonWorld::validate_loaded_geometry() {
  const int nx = meta_.nx;
  const int ny = meta_.ny;
  const int plane = nx * ny;
  constexpr int kMaxSupportedContacts = 63;

  int max_channel = 0;
  std::array<bool, kMaxSupportedContacts + 1> positive_seen{};
  std::array<bool, kMaxSupportedContacts + 1> negative_seen{};
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
      if (cid != 0) {
        throw std::runtime_error("nonmagnetic void/filler cells cannot be contacts");
      }
      if (sig < 0.0f || !std::isfinite(sig)) {
        throw std::runtime_error("region==0 sigma must be finite and >= 0");
      }
    } else if (!(sig > 0.0f) || !std::isfinite(sig)) {
      throw std::runtime_error("conducting Pt/FM cells must have finite sigma>0");
    }

    if (reg == 1 && iz >= meta_.first_r2_layer) {
      throw std::runtime_error("region==1 (Pt) found at or above first_r2_layer");
    }
    if (reg == 2 && iz < meta_.first_r2_layer) {
      throw std::runtime_error("region==2 (FM) found below first_r2_layer");
    }
    if (cid != 0) {
      const int c = static_cast<int>(cid);
      const int abs_c = c < 0 ? -c : c;
      if (abs_c > kMaxSupportedContacts || reg != 1 || iz >= meta_.first_r2_layer) {
        throw std::runtime_error("contact_id must be +/-1.." +
                                 std::to_string(kMaxSupportedContacts) + " on Pt layers only");
      }
      if (c > 0) {
        positive_seen[static_cast<std::size_t>(abs_c)] = true;
      } else {
        negative_seen[static_cast<std::size_t>(abs_c)] = true;
      }
      max_channel = std::max(max_channel, abs_c);
    }
  }

  if (max_channel == 0) {
    throw std::runtime_error("Poisson world: no contact cells found");
  }
  for (int c = 1; c <= max_channel; ++c) {
    if (!positive_seen[static_cast<std::size_t>(c)] ||
        !negative_seen[static_cast<std::size_t>(c)]) {
      throw std::runtime_error("contact_id channels must be a dense +/-1..N set");
    }
  }
  num_contacts_ = max_channel;
}

void PoissonWorld::set_transport_config(TransportConfig config) {
  if (config.amr_enabled && !(config.amr_ratio >= 0.0) ) {
    throw std::runtime_error("amr_ratio must be >= 0 when AMR is enabled");
  }
  if (config.ahe_enabled && !std::isfinite(config.ahe_ratio)) {
    throw std::runtime_error("ahe_ratio must be finite when AHE is enabled");
  }
  if (config.ahe_enabled && config.picard_sweeps < 1) {
    throw std::runtime_error("picard_sweeps must be >= 1 when AHE is enabled");
  }
  config_ = config;
  if (!transport_enabled()) {
    magnetization_set_ = false;
    magnetization_.clear();
    sym_tensor_.clear();
    skew_tensor_.clear();
    skew_row_offsets_.clear();
    skew_col_indices_.clear();
    skew_values_.clear();
    skew_rhs_weight_.clear();
    build_scalar_operator();
    return;
  }
  // Operators are rebuilt after magnetization is provided.
}

void PoissonWorld::set_magnetization_fm_stack(const std::vector<float>& magnetization_mumax) {
  const int n_fm = fm_layer_count();
  if (n_fm <= 0) {
    throw std::runtime_error("Poisson world has no FM layers for magnetization");
  }
  const std::size_t expected =
      3u * static_cast<std::size_t>(n_fm) * static_cast<std::size_t>(meta_.ny) *
      static_cast<std::size_t>(meta_.nx);
  if (magnetization_mumax.size() != expected) {
    throw std::runtime_error(
        "magnetization FM-stack size mismatch; expected (3, n_fm, ny, nx) flattened");
  }
  // Finite/unit-vector checks belong at the Python boundary (once per iterate).
  // Hot path only verifies size here.
  magnetization_ = magnetization_mumax;
  magnetization_set_ = true;
}

void PoissonWorld::refresh_transport_tensors() {
  if (!transport_enabled()) {
    sym_tensor_.clear();
    skew_tensor_.clear();
    return;
  }
  if (!magnetization_set_) {
    throw std::runtime_error("refresh_transport_tensors requires magnetization");
  }
  refresh_cell_tensors();
}

SymTensor6 PoissonWorld::sym_tensor(int cell) const {
#ifndef NDEBUG
  if (cell < 0 || cell >= cell_count()) {
    throw std::runtime_error("sym_tensor: cell out of range");
  }
#endif
  if (!sym_tensor_.empty()) {
    return sym_tensor_[static_cast<std::size_t>(cell)];
  }
  return sym_tensor_for_cell(cell);
}

SkewTensor3 PoissonWorld::skew_tensor(int cell) const {
#ifndef NDEBUG
  if (cell < 0 || cell >= cell_count()) {
    throw std::runtime_error("skew_tensor: cell out of range");
  }
#endif
  if (!skew_tensor_.empty()) {
    return skew_tensor_[static_cast<std::size_t>(cell)];
  }
  return skew_tensor_for_cell(cell);
}

SymTensor6 PoissonWorld::sym_tensor_for_cell(int cell) const {
  const float s = sigma_[static_cast<std::size_t>(cell)];
  if (!is_conducting(cell)) {
    return {};
  }
  if (!config_.amr_enabled || !uses_magnetization(cell)) {
    return {s, s, s, 0.0f, 0.0f, 0.0f};
  }
  if (!magnetization_set_) {
    throw std::runtime_error("AMR enabled but magnetization has not been set");
  }

  const int n_fm = fm_layer_count();
  const int plane = meta_.nx * meta_.ny;
  const int iz = cell / plane;
  const int rem = cell % plane;
  const int iy = rem / meta_.nx;
  const int ix = rem % meta_.nx;
  const int fm_layer = iz - meta_.first_r2_layer;
  if (fm_layer < 0 || fm_layer >= n_fm) {
    return {s, s, s, 0.0f, 0.0f, 0.0f};
  }

  const std::size_t n_xy = static_cast<std::size_t>(n_fm) * static_cast<std::size_t>(plane);
  const std::size_t base =
      static_cast<std::size_t>(fm_layer) * static_cast<std::size_t>(plane) +
      static_cast<std::size_t>(iy) * static_cast<std::size_t>(meta_.nx) +
      static_cast<std::size_t>(ix);
  float mx = magnetization_[base];
  float my = magnetization_[n_xy + base];
  float mz = magnetization_[2u * n_xy + base];
  const float norm = std::sqrt(mx * mx + my * my + mz * mz);
  if (norm > 1e-12f) {
    mx /= norm;
    my /= norm;
    mz /= norm;
  } else {
    mx = my = mz = 0.0f;
  }

  const double q = 6.0 * config_.amr_ratio / (6.0 + config_.amr_ratio);
  const double base_s = static_cast<double>(s);
  return {
      static_cast<float>(base_s * (1.0 - q * (static_cast<double>(mx) * mx - 1.0 / 3.0))),
      static_cast<float>(base_s * (1.0 - q * (static_cast<double>(my) * my - 1.0 / 3.0))),
      static_cast<float>(base_s * (1.0 - q * (static_cast<double>(mz) * mz - 1.0 / 3.0))),
      static_cast<float>(-base_s * q * static_cast<double>(mx) * my),
      static_cast<float>(-base_s * q * static_cast<double>(mx) * mz),
      static_cast<float>(-base_s * q * static_cast<double>(my) * mz),
  };
}

SkewTensor3 PoissonWorld::skew_tensor_for_cell(int cell) const {
  if (!config_.ahe_enabled || !uses_magnetization(cell)) {
    return {};
  }
  if (!magnetization_set_) {
    throw std::runtime_error("AHE enabled but magnetization has not been set");
  }

  const float s = sigma_[static_cast<std::size_t>(cell)];
  const int n_fm = fm_layer_count();
  const int plane = meta_.nx * meta_.ny;
  const int iz = cell / plane;
  const int rem = cell % plane;
  const int iy = rem / meta_.nx;
  const int ix = rem % meta_.nx;
  const int fm_layer = iz - meta_.first_r2_layer;
  if (fm_layer < 0 || fm_layer >= n_fm) {
    return {};
  }

  const std::size_t n_xy = static_cast<std::size_t>(n_fm) * static_cast<std::size_t>(plane);
  const std::size_t base =
      static_cast<std::size_t>(fm_layer) * static_cast<std::size_t>(plane) +
      static_cast<std::size_t>(iy) * static_cast<std::size_t>(meta_.nx) +
      static_cast<std::size_t>(ix);
  float mx = magnetization_[base];
  float my = magnetization_[n_xy + base];
  float mz = magnetization_[2u * n_xy + base];
  const float norm = std::sqrt(mx * mx + my * my + mz * mz);
  if (norm > 1e-12f) {
    mx /= norm;
    my /= norm;
    mz /= norm;
  } else {
    return {};
  }

  const float sigma_ahe = static_cast<float>(config_.ahe_ratio * static_cast<double>(s));
  // Sigma_AHE = sigma_ahe * [[0,-mz,my],[mz,0,-mx],[-my,mx,0]]
  // Stored upper triangle (xy, xz, yz) = (-mz, my, -mx) * sigma_ahe
  return {-sigma_ahe * mz, sigma_ahe * my, -sigma_ahe * mx};
}

void PoissonWorld::refresh_cell_tensors() {
  sym_tensor_.assign(static_cast<std::size_t>(cell_count()), {});
  skew_tensor_.assign(static_cast<std::size_t>(cell_count()), {});
  for (int cell = 0; cell < cell_count(); ++cell) {
    sym_tensor_[static_cast<std::size_t>(cell)] = sym_tensor_for_cell(cell);
    skew_tensor_[static_cast<std::size_t>(cell)] = skew_tensor_for_cell(cell);
  }
}

void PoissonWorld::assemble_unknown_map() {
  pt_column_counts_.assign(static_cast<std::size_t>(meta_.nx * meta_.ny), 0);
  for (int iz = 0; iz < meta_.nz; ++iz) {
    for (int iy = 0; iy < meta_.ny; ++iy) {
      for (int ix = 0; ix < meta_.nx; ++ix) {
        const int cell = flat_index(iz, iy, ix);
        if (is_pt(cell)) {
          pt_column_counts_[static_cast<std::size_t>(xy_index(iy, ix))] += 1;
        }
      }
    }
  }

  unknown_index_.assign(static_cast<std::size_t>(cell_count()), -1);
  unknown_to_cell_.clear();
  unknown_to_cell_.reserve(static_cast<std::size_t>(cell_count()));
  for (int cell = 0; cell < cell_count(); ++cell) {
    if (is_conducting(cell) && contact_id_[static_cast<std::size_t>(cell)] == 0) {
      unknown_index_[static_cast<std::size_t>(cell)] = static_cast<int>(unknown_to_cell_.size());
      unknown_to_cell_.push_back(cell);
    }
  }
}

void PoissonWorld::add_matrix_entry(MatrixBuilder& mb,
                                    int row_cell,
                                    int col_cell,
                                    double coeff) const {
  if (!(std::isfinite(coeff)) || coeff == 0.0) {
    return;
  }
  const int row = unknown_index_[static_cast<std::size_t>(row_cell)];
  if (row < 0) {
    return;
  }
  if (row_cell == col_cell) {
    mb.diagonal[static_cast<std::size_t>(row)] += static_cast<float>(coeff);
    return;
  }
  const int col = unknown_index_[static_cast<std::size_t>(col_cell)];
  if (col >= 0) {
    // Store minus the matrix coefficient: SpMV uses y = D x - sum(off * x_col)
    mb.rows[static_cast<std::size_t>(row)].emplace_back(col, static_cast<float>(-coeff));
    return;
  }
  if (contact_id_[static_cast<std::size_t>(col_cell)] != 0) {
    const int cid = static_cast<int>(contact_id_[static_cast<std::size_t>(col_cell)]);
    const int channel = (cid < 0 ? -cid : cid) - 1;
    const double sign = cid > 0 ? 1.0 : -1.0;
    // RHS contribution for A[row, contact] * phi_contact with phi_contact = sign * V
    // SpMV form stores offdiag as -A, and build_rhs uses weight * V.
    // Want: -A_uc * (sign V) moved to RHS => RHS += A_uc * sign * V
    // With stored convention matching scalar path: weight accumulates signed conductance.
    mb.rhs_weight[static_cast<std::size_t>(channel)][static_cast<std::size_t>(row)] +=
        static_cast<float>(-coeff * sign);
  }
}

void PoissonWorld::add_face_normal_term(MatrixBuilder& mb,
                                       int cell,
                                       int nbr,
                                       int axis,
                                       float sigma_face) const {
  if (!(sigma_face > 0.0f)) {
    return;
  }
  const double g =
      face_area(meta_, axis) * static_cast<double>(sigma_face) / axis_spacing(meta_, axis);
  // Divergence of normal flux: +g*(phi_cell - phi_nbr) contribution to cell residual,
  // which matches A_ii += g, A_i,nbr -= g.
  add_matrix_entry(mb, cell, cell, g);
  add_matrix_entry(mb, cell, nbr, -g);
}

void PoissonWorld::add_cross_terms_for_face(MatrixBuilder& mb,
                                           int cell,
                                           int nbr,
                                           int axis,
                                           float s_xy,
                                           float s_xz,
                                           float s_yz,
                                           bool /*skew*/) const {
  // Finite-volume cross-derivative stencil matching mumax+ PoissonSystem::addDiff
  // (src/physics/poissonsystem.cu). For an x-face with neighbor offset sx=±1:
  //   fac_dy = -sx * Area * σ_xy / (4 cy)
  //   addDiff({0,0,0},{0,-1,0}, fac_dy); addDiff({sx,0,0},{sx,-1,0}, fac_dy);
  //   addDiff({0,1,0},{0,0,0}, fac_dy);  addDiff({sx,1,0},{sx,0,0}, fac_dy);
  // Net: face nodes (cell, nbr) cancel; only transverse neighbors receive ±fac.
  const int plane = meta_.nx * meta_.ny;
  const int iz = cell / plane;
  const int rem = cell % plane;
  const int iy = rem / meta_.nx;
  const int ix = rem % meta_.nx;
  const int niz = nbr / plane;
  const int nrem = nbr % plane;
  const int niy = nrem / meta_.nx;
  const int nix = nrem % meta_.nx;

  const int sx = (nix > ix) ? 1 : ((nix < ix) ? -1 : 0);
  const int sy = (niy > iy) ? 1 : ((niy < iy) ? -1 : 0);
  const int sz = (niz > iz) ? 1 : ((niz < iz) ? -1 : 0);

  auto cell_at = [&](int jz, int jy, int jx) -> int {
    if (!in_bounds(jx, meta_.nx) || !in_bounds(jy, meta_.ny) || !in_bounds(jz, meta_.nz)) {
      return -1;
    }
    const int c = flat_index(jz, jy, jx);
    return is_conducting(c) ? c : -1;
  };

  // Mumax Row::addDiff(r1, r2, val): if both in geometry, A[r1]+=val and A[r2]-=val.
  auto add_diff = [&](int c1, int c2, double val) {
    if (c1 < 0 || c2 < 0 || val == 0.0) {
      return;
    }
    add_matrix_entry(mb, cell, c1, val);
    add_matrix_entry(mb, cell, c2, -val);
  };

  const double area = face_area(meta_, axis);

  if (axis == 0) {
    if (s_xy != 0.0f) {
      const double fac_dy =
          -static_cast<double>(sx) * area * static_cast<double>(s_xy) / (4.0 * meta_.cy);
      const int ym1 = cell_at(iz, iy - 1, ix);
      const int ym1n = cell_at(iz, iy - 1, nix);
      const int yp1 = cell_at(iz, iy + 1, ix);
      const int yp1n = cell_at(iz, iy + 1, nix);
      add_diff(cell, ym1, fac_dy);
      add_diff(nbr, ym1n, fac_dy);
      add_diff(yp1, cell, fac_dy);
      add_diff(yp1n, nbr, fac_dy);
    }
    if (s_xz != 0.0f) {
      const double fac_dz =
          -static_cast<double>(sx) * area * static_cast<double>(s_xz) / (4.0 * meta_.cz);
      const int zm1 = cell_at(iz - 1, iy, ix);
      const int zm1n = cell_at(iz - 1, iy, nix);
      const int zp1 = cell_at(iz + 1, iy, ix);
      const int zp1n = cell_at(iz + 1, iy, nix);
      add_diff(cell, zm1, fac_dz);
      add_diff(nbr, zm1n, fac_dz);
      add_diff(zp1, cell, fac_dz);
      add_diff(zp1n, nbr, fac_dz);
    }
  } else if (axis == 1) {
    if (s_xy != 0.0f) {
      const double fac_dx =
          -static_cast<double>(sy) * area * static_cast<double>(s_xy) / (4.0 * meta_.cx);
      const int xm1 = cell_at(iz, iy, ix - 1);
      const int xm1n = cell_at(iz, niy, ix - 1);
      const int xp1 = cell_at(iz, iy, ix + 1);
      const int xp1n = cell_at(iz, niy, ix + 1);
      add_diff(cell, xm1, fac_dx);
      add_diff(nbr, xm1n, fac_dx);
      add_diff(xp1, cell, fac_dx);
      add_diff(xp1n, nbr, fac_dx);
    }
    if (s_yz != 0.0f) {
      const double fac_dz =
          -static_cast<double>(sy) * area * static_cast<double>(s_yz) / (4.0 * meta_.cz);
      const int zm1 = cell_at(iz - 1, iy, ix);
      const int zm1n = cell_at(iz - 1, niy, ix);
      const int zp1 = cell_at(iz + 1, iy, ix);
      const int zp1n = cell_at(iz + 1, niy, ix);
      add_diff(cell, zm1, fac_dz);
      add_diff(nbr, zm1n, fac_dz);
      add_diff(zp1, cell, fac_dz);
      add_diff(zp1n, nbr, fac_dz);
    }
  } else {
    if (s_xz != 0.0f) {
      const double fac_dx =
          -static_cast<double>(sz) * area * static_cast<double>(s_xz) / (4.0 * meta_.cx);
      const int xm1 = cell_at(iz, iy, ix - 1);
      const int xm1n = cell_at(niz, iy, ix - 1);
      const int xp1 = cell_at(iz, iy, ix + 1);
      const int xp1n = cell_at(niz, iy, ix + 1);
      add_diff(cell, xm1, fac_dx);
      add_diff(nbr, xm1n, fac_dx);
      add_diff(xp1, cell, fac_dx);
      add_diff(xp1n, nbr, fac_dx);
    }
    if (s_yz != 0.0f) {
      const double fac_dy =
          -static_cast<double>(sz) * area * static_cast<double>(s_yz) / (4.0 * meta_.cy);
      const int ym1 = cell_at(iz, iy - 1, ix);
      const int ym1n = cell_at(niz, iy - 1, ix);
      const int yp1 = cell_at(iz, iy + 1, ix);
      const int yp1n = cell_at(niz, iy + 1, ix);
      add_diff(cell, ym1, fac_dy);
      add_diff(nbr, ym1n, fac_dy);
      add_diff(yp1, cell, fac_dy);
      add_diff(yp1n, nbr, fac_dy);
    }
  }
}

void PoissonWorld::finalize_matrix(MatrixBuilder& mb,
                                   std::vector<int>& row_offsets,
                                   std::vector<int>& col_indices,
                                   std::vector<float>& values,
                                   std::vector<float>& diagonal,
                                   std::vector<std::vector<float>>& rhs_weight) const {
  const int n_unknown = unknown_count();
  diagonal = mb.diagonal;
  rhs_weight = mb.rhs_weight;

  // Compress duplicate column entries per row.
  std::vector<int> counts(static_cast<std::size_t>(n_unknown), 0);
  for (int row = 0; row < n_unknown; ++row) {
    auto& entries = mb.rows[static_cast<std::size_t>(row)];
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<int, float>> compressed;
    compressed.reserve(entries.size());
    for (const auto& e : entries) {
      if (!compressed.empty() && compressed.back().first == e.first) {
        compressed.back().second += e.second;
      } else {
        compressed.push_back(e);
      }
    }
    entries.swap(compressed);
    // Drop exact zeros after accumulation.
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [](const auto& e) { return e.second == 0.0f; }),
                  entries.end());
    counts[static_cast<std::size_t>(row)] = static_cast<int>(entries.size());
  }

  row_offsets.assign(static_cast<std::size_t>(n_unknown + 1), 0);
  for (int row = 0; row < n_unknown; ++row) {
    row_offsets[static_cast<std::size_t>(row + 1)] =
        row_offsets[static_cast<std::size_t>(row)] + counts[static_cast<std::size_t>(row)];
  }
  col_indices.assign(static_cast<std::size_t>(row_offsets.back()), -1);
  values.assign(static_cast<std::size_t>(row_offsets.back()), 0.0f);
  for (int row = 0; row < n_unknown; ++row) {
    int write = row_offsets[static_cast<std::size_t>(row)];
    for (const auto& e : mb.rows[static_cast<std::size_t>(row)]) {
      col_indices[static_cast<std::size_t>(write)] = e.first;
      values[static_cast<std::size_t>(write)] = e.second;
      ++write;
    }
    if (!(diagonal[static_cast<std::size_t>(row)] > 0.0f)) {
      throw std::runtime_error("encountered isolated unknown cell in operator assembly");
    }
  }
}

void PoissonWorld::build_scalar_operator() {
  assemble_unknown_map();

  const int n_unknown = unknown_count();
  diagonal_.assign(static_cast<std::size_t>(n_unknown), 0.0f);
  rhs_weight_.assign(static_cast<std::size_t>(num_contacts_),
                     std::vector<float>(static_cast<std::size_t>(n_unknown), 0.0f));
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
      if (!is_conducting(nbr)) {
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
        const int cid = static_cast<int>(contact_id_[static_cast<std::size_t>(nbr)]);
        const int channel = (cid < 0 ? -cid : cid) - 1;
        const float signed_g =
            static_cast<float>((contact_id_[static_cast<std::size_t>(nbr)] > 0 ? 1.0 : -1.0) *
                               conductance);
        rhs_weight_[static_cast<std::size_t>(channel)][static_cast<std::size_t>(row)] += signed_g;
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
      if (nbr_row < 0 || !is_conducting(nbr)) {
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

  skew_row_offsets_.clear();
  skew_col_indices_.clear();
  skew_values_.clear();
  skew_rhs_weight_.clear();
}

void PoissonWorld::build_transport_operators() {
  if (!magnetization_set_) {
    throw std::runtime_error("rebuild_transport_operators requires magnetization");
  }
  refresh_cell_tensors();
  // Unknown map depends only on geometry/sigma/contacts — assembled once in
  // build_scalar_operator() at world load / set_transport_config. Do not rebuild.
  if (unknown_to_cell_.empty()) {
    assemble_unknown_map();
  }
  const int n_unknown = unknown_count();

  MatrixBuilder spd;
  spd.diagonal.assign(static_cast<std::size_t>(n_unknown), 0.0f);
  spd.rhs_weight.assign(static_cast<std::size_t>(num_contacts_),
                        std::vector<float>(static_cast<std::size_t>(n_unknown), 0.0f));
  spd.rows.assign(static_cast<std::size_t>(n_unknown), {});

  MatrixBuilder skew;
  if (config_.ahe_enabled) {
    skew.diagonal.assign(static_cast<std::size_t>(n_unknown), 0.0f);
    skew.rhs_weight.assign(static_cast<std::size_t>(num_contacts_),
                           std::vector<float>(static_cast<std::size_t>(n_unknown), 0.0f));
    skew.rows.assign(static_cast<std::size_t>(n_unknown), {});
  }

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
    const SymTensor6 s0 = sym_tensor_[static_cast<std::size_t>(cell)];
    const SkewTensor3 k0 = skew_tensor_[static_cast<std::size_t>(cell)];

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
      if (!is_conducting(nbr)) {
        continue;
      }
      const SymTensor6 s1 = sym_tensor_[static_cast<std::size_t>(nbr)];
      const SkewTensor3 k1 = skew_tensor_[static_cast<std::size_t>(nbr)];

      float face_diag = 0.0f;
      if (axis == 0) {
        face_diag = avg_diag(s0.xx, s1.xx);
      } else if (axis == 1) {
        face_diag = avg_diag(s0.yy, s1.yy);
      } else {
        face_diag = avg_diag(s0.zz, s1.zz);
      }
      add_face_normal_term(spd, cell, nbr, axis, face_diag);

      const float face_xy = avg_signed(s0.xy, s1.xy);
      const float face_xz = avg_signed(s0.xz, s1.xz);
      const float face_yz = avg_signed(s0.yz, s1.yz);
      add_cross_terms_for_face(spd, cell, nbr, axis, face_xy, face_xz, face_yz, false);

      if (config_.ahe_enabled) {
        const float k_xy = avg_signed(k0.xy, k1.xy);
        const float k_xz = avg_signed(k0.xz, k1.xz);
        const float k_yz = avg_signed(k0.yz, k1.yz);
        // Skew tensor has zero diagonal; only cross terms contribute.
        add_cross_terms_for_face(skew, cell, nbr, axis, k_xy, k_xz, k_yz, true);
      }
    }
  }

  finalize_matrix(spd, row_offsets_, col_indices_, offdiag_conductance_, diagonal_,
                  rhs_weight_);

  if (config_.ahe_enabled) {
    // Skew diagonals should remain ~0; allow slightly non-positive diagonals.
    for (float& d : skew.diagonal) {
      if (!(std::isfinite(d))) {
        throw std::runtime_error("non-finite skew diagonal during AHE assembly");
      }
      // Keep a tiny positive diagonal so finalize_matrix isolation check still works
      // for rows that only have skew contact couplings and no unknown neighbors.
      if (!(d > 0.0f)) {
        d = 1e-30f;
      }
    }
    finalize_matrix(skew, skew_row_offsets_, skew_col_indices_, skew_values_,
                    skew.diagonal, skew_rhs_weight_);
    // True skew operator has zero diagonal; clear the artificial floor from GPU SpMV.
    // apply_skew uses values only (no diagonal term).
  } else {
    skew_row_offsets_.clear();
    skew_col_indices_.clear();
    skew_values_.clear();
    skew_rhs_weight_.clear();
  }
}

void PoissonWorld::rebuild_transport_operators() {
  if (!transport_enabled()) {
    build_scalar_operator();
    return;
  }
  build_transport_operators();
}

void PoissonWorld::build_transport_pattern_operators() {
  // Topology-only sparsity for the GMRES device-update path. Column presence must
  // not depend on coefficient cancellation: unit cross-stencil values can still
  // sum to exact zero after compression and drop CSR entries that are nonzero for
  // a real magnetization. Mark structural columns with a set instead.
  if (!transport_enabled()) {
    build_scalar_operator();
    return;
  }
  if (unknown_to_cell_.empty()) {
    assemble_unknown_map();
  }
  const int n_unknown = unknown_count();
  std::vector<std::unordered_set<int>> pattern_cols(static_cast<std::size_t>(n_unknown));
  diagonal_.assign(static_cast<std::size_t>(n_unknown), 1.0f);
  rhs_weight_.assign(static_cast<std::size_t>(num_contacts_),
                     std::vector<float>(static_cast<std::size_t>(n_unknown), 0.0f));

  auto mark_entry = [&](int row_cell, int col_cell) {
    if (col_cell < 0) {
      return;
    }
    const int row = unknown_index_[static_cast<std::size_t>(row_cell)];
    if (row < 0) {
      return;
    }
    if (row_cell == col_cell) {
      // Diagonal is injected by upload_transport_operator; keep it positive.
      diagonal_[static_cast<std::size_t>(row)] = 1.0f;
      return;
    }
    const int col = unknown_index_[static_cast<std::size_t>(col_cell)];
    if (col >= 0) {
      pattern_cols[static_cast<std::size_t>(row)].insert(col);
    }
  };

  auto cell_at = [&](int jz, int jy, int jx) -> int {
    if (!in_bounds(jx, meta_.nx) || !in_bounds(jy, meta_.ny) || !in_bounds(jz, meta_.nz)) {
      return -1;
    }
    const int c = flat_index(jz, jy, jx);
    return is_conducting(c) ? c : -1;
  };

  auto mark_diff = [&](int row_cell, int c1, int c2) {
    mark_entry(row_cell, c1);
    mark_entry(row_cell, c2);
  };

  auto mark_cross = [&](int cell, int nbr, int axis) {
    const int plane = meta_.nx * meta_.ny;
    const int iz = cell / plane;
    const int rem = cell % plane;
    const int iy = rem / meta_.nx;
    const int ix = rem % meta_.nx;
    const int niz = nbr / plane;
    const int nrem = nbr % plane;
    const int niy = nrem / meta_.nx;
    const int nix = nrem % meta_.nx;
    // Full cross stencil for all three off-diagonal tensor components.
    if (axis == 0) {
      mark_diff(cell, cell, cell_at(iz, iy - 1, ix));
      mark_diff(cell, nbr, cell_at(iz, iy - 1, nix));
      mark_diff(cell, cell_at(iz, iy + 1, ix), cell);
      mark_diff(cell, cell_at(iz, iy + 1, nix), nbr);
      mark_diff(cell, cell, cell_at(iz - 1, iy, ix));
      mark_diff(cell, nbr, cell_at(iz - 1, iy, nix));
      mark_diff(cell, cell_at(iz + 1, iy, ix), cell);
      mark_diff(cell, cell_at(iz + 1, iy, nix), nbr);
    } else if (axis == 1) {
      mark_diff(cell, cell, cell_at(iz, iy, ix - 1));
      mark_diff(cell, nbr, cell_at(iz, niy, ix - 1));
      mark_diff(cell, cell_at(iz, iy, ix + 1), cell);
      mark_diff(cell, cell_at(iz, niy, ix + 1), nbr);
      mark_diff(cell, cell, cell_at(iz - 1, iy, ix));
      mark_diff(cell, nbr, cell_at(iz - 1, niy, ix));
      mark_diff(cell, cell_at(iz + 1, iy, ix), cell);
      mark_diff(cell, cell_at(iz + 1, niy, ix), nbr);
    } else {
      mark_diff(cell, cell, cell_at(iz, iy, ix - 1));
      mark_diff(cell, nbr, cell_at(niz, iy, ix - 1));
      mark_diff(cell, cell_at(iz, iy, ix + 1), cell);
      mark_diff(cell, cell_at(niz, iy, ix + 1), nbr);
      mark_diff(cell, cell, cell_at(iz, iy - 1, ix));
      mark_diff(cell, nbr, cell_at(niz, iy - 1, ix));
      mark_diff(cell, cell_at(iz, iy + 1, ix), cell);
      mark_diff(cell, cell_at(niz, iy + 1, ix), nbr);
    }
  };

  constexpr std::array<std::array<int, 4>, 6> kNeighbors = {{
      {{-1, 0, 0, 0}}, {{1, 0, 0, 0}}, {{0, -1, 0, 1}},
      {{0, 1, 0, 1}},  {{0, 0, -1, 2}}, {{0, 0, 1, 2}},
  }};

  const bool need_cross = config_.amr_enabled || config_.ahe_enabled;
  const int plane = meta_.nx * meta_.ny;
  for (int row = 0; row < n_unknown; ++row) {
    const int cell = unknown_to_cell_[static_cast<std::size_t>(row)];
    mark_entry(cell, cell);
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
      if (!is_conducting(nbr)) {
        continue;
      }
      mark_entry(cell, cell);
      mark_entry(cell, nbr);
      if (need_cross) {
        mark_cross(cell, nbr, axis);
      }
    }
  }

  row_offsets_.assign(static_cast<std::size_t>(n_unknown + 1), 0);
  for (int row = 0; row < n_unknown; ++row) {
    row_offsets_[static_cast<std::size_t>(row + 1)] =
        row_offsets_[static_cast<std::size_t>(row)] +
        static_cast<int>(pattern_cols[static_cast<std::size_t>(row)].size());
  }
  col_indices_.assign(static_cast<std::size_t>(row_offsets_.back()), -1);
  offdiag_conductance_.assign(static_cast<std::size_t>(row_offsets_.back()), 1.0f);
  for (int row = 0; row < n_unknown; ++row) {
    std::vector<int> cols(pattern_cols[static_cast<std::size_t>(row)].begin(),
                          pattern_cols[static_cast<std::size_t>(row)].end());
    std::sort(cols.begin(), cols.end());
    int write = row_offsets_[static_cast<std::size_t>(row)];
    for (int col : cols) {
      col_indices_[static_cast<std::size_t>(write)] = col;
      // Placeholder nonzero so upload_transport_operator keeps the column slot.
      // Device update overwrites every merged CSR value each step.
      offdiag_conductance_[static_cast<std::size_t>(write)] = 1.0f;
      ++write;
    }
  }

  // Skew shares the same geometric cross stencil; merged GMRES CSR is built from
  // the SPD pattern above. Keep skew buffers empty for the pattern path.
  skew_row_offsets_.clear();
  skew_col_indices_.clear();
  skew_values_.clear();
  skew_rhs_weight_.clear();
}

void PoissonWorld::build_operator() {
  build_scalar_operator();
}

void PoissonWorld::build_rhs(const std::vector<double>& potentials,
                             std::vector<double>& rhs) const {
  build_rhs_spd(potentials, rhs);
}

void PoissonWorld::build_rhs_spd(const std::vector<double>& potentials,
                                 std::vector<double>& rhs) const {
#ifndef NDEBUG
  if (static_cast<int>(potentials.size()) != num_contacts_) {
    throw std::runtime_error("build_rhs_spd: potentials size does not match number of contacts");
  }
#endif
  rhs.assign(static_cast<std::size_t>(unknown_count()), 0.0);
  for (int c = 0; c < num_contacts_; ++c) {
    const double v = potentials[static_cast<std::size_t>(c)];
    const auto& weights = rhs_weight_[static_cast<std::size_t>(c)];
    for (int row = 0; row < unknown_count(); ++row) {
      rhs[static_cast<std::size_t>(row)] +=
          static_cast<double>(weights[static_cast<std::size_t>(row)]) * v;
    }
  }
}

void PoissonWorld::build_rhs_skew(const std::vector<double>& potentials,
                                  std::vector<double>& rhs) const {
  if (!config_.ahe_enabled) {
    rhs.assign(static_cast<std::size_t>(unknown_count()), 0.0);
    return;
  }
#ifndef NDEBUG
  if (static_cast<int>(potentials.size()) != num_contacts_) {
    throw std::runtime_error("build_rhs_skew: potentials size does not match number of contacts");
  }
#endif
  if (skew_rhs_weight_.empty()) {
    rhs.assign(static_cast<std::size_t>(unknown_count()), 0.0);
    return;
  }
  rhs.assign(static_cast<std::size_t>(unknown_count()), 0.0);
  for (int c = 0; c < num_contacts_; ++c) {
    const double v = potentials[static_cast<std::size_t>(c)];
    const auto& weights = skew_rhs_weight_[static_cast<std::size_t>(c)];
    for (int row = 0; row < unknown_count(); ++row) {
      rhs[static_cast<std::size_t>(row)] +=
          static_cast<double>(weights[static_cast<std::size_t>(row)]) * v;
    }
  }
}

void PoissonWorld::fill_full_phi(const std::vector<double>& x,
                                 const std::vector<double>& potentials,
                                 std::vector<float>& phi) const {
#ifndef NDEBUG
  if (static_cast<int>(x.size()) != unknown_count()) {
    throw std::runtime_error("fill_full_phi received vector of wrong size");
  }
  if (static_cast<int>(potentials.size()) != num_contacts_) {
    throw std::runtime_error("fill_full_phi: potentials size does not match number of contacts");
  }
#endif

  phi.assign(static_cast<std::size_t>(cell_count()), 0.0f);
  for (int cell = 0; cell < cell_count(); ++cell) {
    const auto idx = static_cast<std::size_t>(cell);
    if (!is_conducting(cell)) {
      continue;
    }
    if (contact_id_[idx] != 0) {
      phi[idx] = static_cast<float>(dirichlet_potential(contact_id_[idx], potentials));
      continue;
    }
    const int row = unknown_index_[idx];
#ifndef NDEBUG
    if (row < 0) {
      throw std::runtime_error("active non-contact cell missing from unknown map");
    }
#endif
    phi[idx] = static_cast<float>(x[static_cast<std::size_t>(row)]);
  }
}

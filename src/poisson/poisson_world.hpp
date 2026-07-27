#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ManifestData {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  int first_r2_layer = 2;
  double theta_sh = 0.0;
  double decay_length = 0.0;
  std::string region_file;
  std::string contact_id_file;
  std::string sigma_file;
};

struct TransportConfig {
  bool amr_enabled = false;
  bool ahe_enabled = false;
  double amr_ratio = 0.0;
  double ahe_ratio = 0.0;
  int picard_sweeps = 2;
  double picard_tolerance = 0.0;
};

struct SymTensor6 {
  float xx = 0.0f;
  float yy = 0.0f;
  float zz = 0.0f;
  float xy = 0.0f;
  float xz = 0.0f;
  float yz = 0.0f;
};

struct SkewTensor3 {
  // Antisymmetric AHE matrix stored by upper-triangle components:
  //   [[ 0, xy, xz],
  //    [-xy, 0, yz],
  //    [-xz,-yz, 0]]
  float xy = 0.0f;
  float xz = 0.0f;
  float yz = 0.0f;
};

class PoissonWorld {
 public:
  static constexpr float kSigmaFloor = 1e-20f;

  static PoissonWorld load(const std::string& manifest_path);
  static PoissonWorld from_arrays(ManifestData meta,
                                  std::vector<std::int8_t> region,
                                  std::vector<std::int8_t> contact_id,
                                  std::vector<float> sigma);

  int nx() const { return meta_.nx; }
  int ny() const { return meta_.ny; }
  int nz() const { return meta_.nz; }
  double cx() const { return meta_.cx; }
  double cy() const { return meta_.cy; }
  double cz() const { return meta_.cz; }
  int first_r2_layer() const { return meta_.first_r2_layer; }
  double theta_sh() const { return meta_.theta_sh; }
  double decay_length() const { return meta_.decay_length; }

  int cell_count() const { return meta_.nx * meta_.ny * meta_.nz; }
  std::size_t frame_elements() const { return static_cast<std::size_t>(cell_count()) * 3u; }
  int unknown_count() const { return static_cast<int>(unknown_to_cell_.size()); }
  int num_contacts() const { return num_contacts_; }
  int fm_layer_count() const { return meta_.nz - meta_.first_r2_layer; }

  int flat_index(int iz, int iy, int ix) const { return (iz * meta_.ny + iy) * meta_.nx + ix; }
  int xy_index(int iy, int ix) const { return iy * meta_.nx + ix; }

  const std::vector<std::int8_t>& region() const { return region_; }
  const std::vector<std::int8_t>& contact_id() const { return contact_id_; }
  const std::vector<float>& sigma() const { return sigma_; }
  const std::vector<int>& unknown_index() const { return unknown_index_; }
  const std::vector<int>& unknown_to_cell() const { return unknown_to_cell_; }
  const std::vector<int>& row_offsets() const { return row_offsets_; }
  const std::vector<int>& col_indices() const { return col_indices_; }
  const std::vector<float>& offdiag_conductance() const { return offdiag_conductance_; }
  const std::vector<float>& diagonal() const { return diagonal_; }
  const std::vector<int>& pt_column_counts() const { return pt_column_counts_; }

  const std::vector<int>& skew_row_offsets() const { return skew_row_offsets_; }
  const std::vector<int>& skew_col_indices() const { return skew_col_indices_; }
  const std::vector<float>& skew_values() const { return skew_values_; }

  const TransportConfig& transport_config() const { return config_; }
  bool transport_enabled() const { return config_.amr_enabled || config_.ahe_enabled; }
  bool amr_enabled() const { return config_.amr_enabled; }
  bool ahe_enabled() const { return config_.ahe_enabled; }

  bool is_pt(int cell) const {
    return region_[static_cast<std::size_t>(cell)] == 1;
  }
  bool is_fm(int cell) const {
    return region_[static_cast<std::size_t>(cell)] == 2;
  }
  bool is_conducting(int cell) const {
    return sigma_[static_cast<std::size_t>(cell)] > kSigmaFloor;
  }
  bool is_insulating(int cell) const { return !is_conducting(cell); }
  bool uses_magnetization(int cell) const {
    return is_fm(cell) && is_conducting(cell);
  }
  bool uses_scalar_transport(int cell) const {
    return is_conducting(cell) && !uses_magnetization(cell);
  }

  SymTensor6 sym_tensor(int cell) const;
  SkewTensor3 skew_tensor(int cell) const;

  void set_transport_config(TransportConfig config);
  void set_magnetization_fm_stack(const std::vector<float>& magnetization_mumax);
  void refresh_transport_tensors();
  void rebuild_transport_operators();
  /// Build maximal AMR/AHE CSR sparsity from geometry only (no m dependence).
  /// Values are placeholders; GMRES device update overwrites them each step.
  void build_transport_pattern_operators();

  void build_rhs(const std::vector<double>& potentials, std::vector<double>& rhs) const;
  void build_rhs_spd(const std::vector<double>& potentials, std::vector<double>& rhs) const;
  void build_rhs_skew(const std::vector<double>& potentials, std::vector<double>& rhs) const;
  void fill_full_phi(const std::vector<double>& x,
                     const std::vector<double>& potentials,
                     std::vector<float>& phi) const;

 private:
  struct MatrixBuilder {
    std::vector<float> diagonal;
    std::vector<std::vector<float>> rhs_weight;
    std::vector<std::vector<std::pair<int, float>>> rows;
  };

  void build_operator();
  void build_scalar_operator();
  void build_transport_operators();
  void assemble_unknown_map();
  void finalize_matrix(MatrixBuilder& mb,
                       std::vector<int>& row_offsets,
                       std::vector<int>& col_indices,
                       std::vector<float>& values,
                       std::vector<float>& diagonal,
                       std::vector<std::vector<float>>& rhs_weight) const;
  void add_matrix_entry(MatrixBuilder& mb, int row_cell, int col_cell, double coeff) const;
  void add_face_normal_term(MatrixBuilder& mb,
                            int cell,
                            int nbr,
                            int axis,
                            float sigma_face) const;
  void add_cross_terms_for_face(MatrixBuilder& mb,
                                int cell,
                                int nbr,
                                int axis,
                                float s_xy,
                                float s_xz,
                                float s_yz,
                                bool skew) const;

  SymTensor6 sym_tensor_for_cell(int cell) const;
  SkewTensor3 skew_tensor_for_cell(int cell) const;
  void refresh_cell_tensors();

  static ManifestData parse_manifest(const std::string& manifest_path);
  void validate_loaded_geometry();

  ManifestData meta_;
  TransportConfig config_;
  std::vector<std::int8_t> region_;
  std::vector<std::int8_t> contact_id_;
  std::vector<float> sigma_;
  int num_contacts_ = 0;

  std::vector<int> unknown_index_;
  std::vector<int> unknown_to_cell_;
  std::vector<int> row_offsets_;
  std::vector<int> col_indices_;
  std::vector<float> offdiag_conductance_;
  std::vector<float> diagonal_;
  std::vector<std::vector<float>> rhs_weight_;
  std::vector<int> pt_column_counts_;

  std::vector<int> skew_row_offsets_;
  std::vector<int> skew_col_indices_;
  std::vector<float> skew_values_;
  std::vector<std::vector<float>> skew_rhs_weight_;

  std::vector<SymTensor6> sym_tensor_;
  std::vector<SkewTensor3> skew_tensor_;
  // Magnetization on the Poisson FM stack, mumax layout (3, n_fm, ny, nx).
  std::vector<float> magnetization_;
  bool magnetization_set_ = false;
};

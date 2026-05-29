#pragma once

#include <array>
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

class PoissonWorld {
 public:
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

  int flat_index(int iz, int iy, int ix) const { return (iz * meta_.ny + iy) * meta_.nx + ix; }
  int xy_index(int iy, int ix) const { return iy * meta_.nx + ix; }

  const std::vector<std::int8_t>& region() const { return region_; }
  const std::vector<float>& sigma() const { return sigma_; }
  const std::vector<int>& row_offsets() const { return row_offsets_; }
  const std::vector<int>& col_indices() const { return col_indices_; }
  const std::vector<float>& offdiag_conductance() const { return offdiag_conductance_; }
  const std::vector<float>& diagonal() const { return diagonal_; }
  const std::vector<int>& pt_column_counts() const { return pt_column_counts_; }

  void build_rhs(const std::array<double, 3>& potentials, std::vector<double>& rhs) const;
  void fill_full_phi(const std::vector<double>& x,
                     const std::array<double, 3>& potentials,
                     std::vector<float>& phi) const;

 private:
  void build_operator();
  static ManifestData parse_manifest(const std::string& manifest_path);
  void validate_loaded_geometry() const;

  ManifestData meta_;
  std::vector<std::int8_t> region_;
  std::vector<std::int8_t> contact_id_;
  std::vector<float> sigma_;

  std::vector<int> unknown_index_;
  std::vector<int> unknown_to_cell_;
  std::vector<int> row_offsets_;
  std::vector<int> col_indices_;
  std::vector<float> offdiag_conductance_;
  std::vector<float> diagonal_;
  std::vector<float> rhs_weight_c0_;
  std::vector<float> rhs_weight_c1_;
  std::vector<float> rhs_weight_c2_;
  std::vector<int> pt_column_counts_;
};

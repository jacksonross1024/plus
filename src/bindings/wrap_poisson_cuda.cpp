#include "wrappers.hpp"

#include <chrono>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "field.hpp"
#include "poisson_cuda_session.hpp"
#include "variable.hpp"

namespace {

ContactPotentials contact_potentials_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> potentials_array) {
  const py::buffer_info info = potentials_array.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("contact_potentials must have shape (nt, num_contacts)");
  }
  if (info.shape[0] <= 0 || info.shape[1] <= 0) {
    throw std::invalid_argument(
        "contact_potentials must contain at least one row and one contact column");
  }

  const auto* data = static_cast<const double*>(info.ptr);
  const std::size_t nt = static_cast<std::size_t>(info.shape[0]);
  const std::size_t num_contacts = static_cast<std::size_t>(info.shape[1]);
  ContactPotentials potentials;
  potentials.channels.assign(num_contacts, std::vector<float>(nt));
  for (std::size_t i = 0; i < nt; ++i) {
    for (std::size_t c = 0; c < num_contacts; ++c) {
      potentials.channels[c][i] = static_cast<float>(data[i * num_contacts + c]);
    }
  }
  return potentials;
}

template <typename T>
std::vector<T> vector_from_3d_array(py::array_t<T, py::array::c_style | py::array::forcecast> array,
                                    int nz,
                                    int ny,
                                    int nx,
                                    const char* name) {
  const py::buffer_info info = array.request();
  if (info.ndim != 3 || info.shape[0] != nz || info.shape[1] != ny || info.shape[2] != nx) {
    throw std::invalid_argument(std::string(name) + " must have shape (nz, ny, nx)");
  }
  const auto* data = static_cast<const T*>(info.ptr);
  const std::size_t n = static_cast<std::size_t>(nz) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nx);
  return std::vector<T>(data, data + n);
}

std::vector<float> vector_from_magnetization_array(
    py::array_t<float, py::array::c_style | py::array::forcecast> array) {
  const py::buffer_info info = array.request();
  if (info.ndim != 4 || info.shape[0] != 3) {
    throw std::invalid_argument("magnetization must have shape (3, nz, ny, nx)");
  }
  const auto* data = static_cast<const float*>(info.ptr);
  const std::size_t n = static_cast<std::size_t>(info.shape[0]) *
                        static_cast<std::size_t>(info.shape[1]) *
                        static_cast<std::size_t>(info.shape[2]) *
                        static_cast<std::size_t>(info.shape[3]);
  return std::vector<float>(data, data + n);
}

TransportConfig transport_from_args(bool amr_enabled,
                                    double amr_ratio,
                                    bool ahe_enabled,
                                    double ahe_ratio,
                                    int picard_sweeps,
                                    double picard_tolerance) {
  TransportConfig cfg;
  cfg.amr_enabled = amr_enabled;
  cfg.amr_ratio = amr_ratio;
  cfg.ahe_enabled = ahe_enabled;
  cfg.ahe_ratio = ahe_ratio;
  cfg.picard_sweeps = picard_sweeps;
  cfg.picard_tolerance = picard_tolerance;
  return cfg;
}

PoissonLinearSolverKind solver_kind_from_string(const std::string& solver) {
  if (solver == "pcg") {
    return PoissonLinearSolverKind::kPcg;
  }
  if (solver == "gmres" || solver == "gmres_cusparse") {
    return PoissonLinearSolverKind::kGmresCusparse;
  }
  throw std::invalid_argument("poisson solver must be 'pcg' or 'gmres_cusparse'");
}

std::string solver_kind_to_string(PoissonLinearSolverKind solver) {
  switch (solver) {
    case PoissonLinearSolverKind::kPcg:
      return "pcg";
    case PoissonLinearSolverKind::kGmresCusparse:
      return "gmres_cusparse";
  }
  return "unknown";
}

template <typename T>
std::vector<T> vector_from_1d_array(py::array_t<T, py::array::c_style | py::array::forcecast> array,
                                    int n,
                                    const char* name) {
  const py::buffer_info info = array.request();
  if (info.ndim != 1 || info.shape[0] != n) {
    throw std::invalid_argument(std::string(name) + " must have shape (n_fm,)");
  }
  const auto* data = static_cast<const T*>(info.ptr);
  return std::vector<T>(data, data + n);
}

/// Internal Poisson buffers use ``(nz, ny, nx, 3)``; mumax+ uses ``(3, nz, ny, nx)``.
py::array_t<float> frame_to_numpy_mumax_copy(const std::vector<float>& frame, int nz, int ny,
                                             int nx) {
  py::array_t<float> out({3, nz, ny, nx});
  if (frame.empty()) {
    return out;
  }

  auto buf = out.mutable_unchecked<4>();
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t cell =
            (static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
             static_cast<std::size_t>(iy)) *
                static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        for (int c = 0; c < 3; ++c) {
          buf(c, iz, iy, ix) = frame[cell * 3u + static_cast<std::size_t>(c)];
        }
      }
    }
  }
  return out;
}

py::dict stats_to_dict(const StepStats& stats) {
  py::dict out;
  out["step"] = stats.step;
  out["skipped"] = stats.skipped;
  out["iterations"] = stats.iterations;
  out["residual_initial"] = stats.residual_initial;
  out["residual"] = stats.residual;
  out["rhs_inf"] = stats.rhs_inf;
  out["residual_rel"] = stats.residual_rel;
  out["pcg_error"] = stats.pcg_error;
  out["pcg_converged"] = stats.pcg_converged;
  out["picard_error"] = stats.picard_error;
  out["picard_sweeps_used"] = stats.picard_sweeps_used;
  out["elapsed_s"] = stats.elapsed_s;
  out["timing_total_s"] = stats.timing_total_s;
  out["timing_device_magnetization_s"] = stats.timing_device_magnetization_s;
  out["timing_transport_s"] = stats.timing_transport_s;
  out["timing_magnetization_set_s"] = stats.timing_magnetization_set_s;
  out["timing_transport_rebuild_s"] = stats.timing_transport_rebuild_s;
  out["timing_operator_upload_s"] = stats.timing_operator_upload_s;
  out["timing_rhs_build_s"] = stats.timing_rhs_build_s;
  out["timing_linear_solve_s"] = stats.timing_linear_solve_s;
  out["timing_fill_phi_s"] = stats.timing_fill_phi_s;
  out["timing_hall_s"] = stats.timing_hall_s;
  out["timing_j_raw_s"] = stats.timing_j_raw_s;
  out["timing_jcur_extract_s"] = stats.timing_jcur_extract_s;
  out["timing_jmod_postprocess_s"] = stats.timing_jmod_postprocess_s;
  out["timing_jmod_extract_s"] = stats.timing_jmod_extract_s;
  out["note"] = stats.stats_note;
  return out;
}

HallProbeIndices hall_probes_from_nested_arrays(
    py::list high_y_list,
    py::list low_y_list) {
  if (high_y_list.size() != low_y_list.size()) {
    throw std::invalid_argument("Hall high_y and low_y lists must have equal length");
  }
  if (high_y_list.empty()) {
    throw std::invalid_argument("Hall probe lists cannot be empty");
  }
  HallProbeIndices probes;
  probes.high_y.resize(static_cast<std::size_t>(high_y_list.size()));
  probes.low_y.resize(static_cast<std::size_t>(low_y_list.size()));
  for (py::ssize_t c = 0; c < high_y_list.size(); ++c) {
    auto high_arr = py::cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>(
        high_y_list[c]);
    auto low_arr = py::cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>(
        low_y_list[c]);
    const py::buffer_info high_info = high_arr.request();
    const py::buffer_info low_info = low_arr.request();
    if (high_info.ndim != 1 || low_info.ndim != 1) {
      throw std::invalid_argument("Hall probe index arrays must be 1D");
    }
    const auto* high_data = static_cast<const std::int64_t*>(high_info.ptr);
    const auto* low_data = static_cast<const std::int64_t*>(low_info.ptr);
    probes.high_y[static_cast<std::size_t>(c)].assign(high_data, high_data + high_info.shape[0]);
    probes.low_y[static_cast<std::size_t>(c)].assign(low_data, low_data + low_info.shape[0]);
  }
  return probes;
}

py::array_t<double> vector_to_numpy_1d(const std::vector<double>& values) {
  py::array_t<double> out({static_cast<py::ssize_t>(values.size())});
  if (!values.empty()) {
    std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(double));
  }
  return out;
}

py::dict hall_components_to_dict(const HallPotentialComponents& comps) {
  py::dict out;
  out["voltages"] = vector_to_numpy_1d(comps.voltages);
  out["high_y_means"] = vector_to_numpy_1d(comps.high_y_means);
  out["low_y_means"] = vector_to_numpy_1d(comps.low_y_means);
  py::list high_counts;
  py::list low_counts;
  for (std::size_t c = 0; c < comps.high_y_counts.size(); ++c) {
    high_counts.append(comps.high_y_counts[c]);
    low_counts.append(comps.low_y_counts[c]);
  }
  out["high_y_counts"] = high_counts;
  out["low_y_counts"] = low_counts;
  return out;
}

py::dict iterate_to_dict(PoissonCudaSession& session) {
  StepStats stats;
  {
    py::gil_scoped_release release;
    stats = session.iterate();
  }

  py::dict out;
  const auto t_jmod0 = std::chrono::steady_clock::now();
  out["jmod"] = frame_to_numpy_mumax_copy(session.jmod_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  const auto t_jmod1 = std::chrono::steady_clock::now();
  out["jcur"] = frame_to_numpy_mumax_copy(session.jcur_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  const auto t_jcur1 = std::chrono::steady_clock::now();
  py::dict stats_dict = stats_to_dict(stats);
  stats_dict["timing_numpy_jmod_s"] = std::chrono::duration<double>(t_jmod1 - t_jmod0).count();
  stats_dict["timing_numpy_jcur_s"] = std::chrono::duration<double>(t_jcur1 - t_jmod1).count();
  out["stats"] = stats_dict;
  return out;
}

py::dict iterate_with_magnetization_to_dict(
    PoissonCudaSession& session,
    py::array_t<float, py::array::c_style | py::array::forcecast> magnetization) {
  std::vector<float> mag = vector_from_magnetization_array(magnetization);
  StepStats stats;
  {
    py::gil_scoped_release release;
    stats = session.iterate_with_magnetization(mag);
  }

  py::dict out;
  const auto t_jmod0 = std::chrono::steady_clock::now();
  out["jmod"] = frame_to_numpy_mumax_copy(session.jmod_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  const auto t_jmod1 = std::chrono::steady_clock::now();
  out["jcur"] = frame_to_numpy_mumax_copy(session.jcur_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  const auto t_jcur1 = std::chrono::steady_clock::now();
  py::dict stats_dict = stats_to_dict(stats);
  stats_dict["timing_numpy_jmod_s"] = std::chrono::duration<double>(t_jmod1 - t_jmod0).count();
  stats_dict["timing_numpy_jcur_s"] = std::chrono::duration<double>(t_jcur1 - t_jmod1).count();
  out["stats"] = stats_dict;
  return out;
}

py::dict iterate_with_magnetization_variable_to_dict(
    PoissonCudaSession& session,
    const Variable& magnetization,
    py::array_t<int, py::array::c_style | py::array::forcecast> src_lo,
    py::array_t<int, py::array::c_style | py::array::forcecast> src_hi,
    py::array_t<float, py::array::c_style | py::array::forcecast> weight_hi,
    bool average_z) {
  const Field& field = magnetization.field();
  if (field.ncomp() != 3) {
    throw std::invalid_argument("magnetization variable must have 3 components");
  }
  const int3 size = field.grid().size();
  if (size.y != session.ny() || size.x != session.nx()) {
    throw std::invalid_argument("magnetization variable xy shape does not match Poisson world");
  }
  const int n_fm = session.fm_layer_count();
  std::vector<int> lo = vector_from_1d_array<int>(src_lo, n_fm, "src_lo");
  std::vector<int> hi = vector_from_1d_array<int>(src_hi, n_fm, "src_hi");
  std::vector<float> wh = vector_from_1d_array<float>(weight_hi, n_fm, "weight_hi");

  StepStats stats;
  {
    py::gil_scoped_release release;
#if FP_PRECISION == SINGLE
    stats = session.iterate_with_magnetization_device(field.device_ptr(0), field.device_ptr(1),
                                                      field.device_ptr(2), size.z, size.y, size.x,
                                                      lo, hi, wh, average_z);
#else
    throw std::runtime_error(
        "device magnetization Poisson path requires single-precision mumax+ fields");
#endif
  }

  py::dict out;
  const auto t_jmod0 = std::chrono::steady_clock::now();
  out["jmod"] = frame_to_numpy_mumax_copy(session.jmod_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  const auto t_jmod1 = std::chrono::steady_clock::now();
  out["jcur"] = frame_to_numpy_mumax_copy(session.jcur_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  const auto t_jcur1 = std::chrono::steady_clock::now();
  py::dict stats_dict = stats_to_dict(stats);
  stats_dict["timing_numpy_jmod_s"] = std::chrono::duration<double>(t_jmod1 - t_jmod0).count();
  stats_dict["timing_numpy_jcur_s"] = std::chrono::duration<double>(t_jcur1 - t_jmod1).count();
  out["stats"] = stats_dict;
  return out;
}

}  // namespace

void wrap_poisson_cuda(py::module& m) {
  py::class_<PoissonCudaSession>(m, "PoissonCudaSolver")
      .def_static(
          "from_manifest",
          [](const std::string& manifest_path,
             py::array_t<double, py::array::c_style | py::array::forcecast> potentials_array,
             double tolerance,
             int max_iterations,
             double skip_threshold,
             const std::string& slice_x,
             const std::string& slice_y,
             const std::string& slice_z,
             int cuda_tol_batch_first,
             int cuda_tol_batch_next,
             bool amr_enabled,
             double amr_ratio,
             bool ahe_enabled,
             double ahe_ratio,
             int picard_sweeps,
             double picard_tolerance,
             const std::string& solver,
             int gmres_restart) {
            return std::make_unique<PoissonCudaSession>(
                PoissonWorld::load(manifest_path), contact_potentials_from_array(potentials_array),
                tolerance, max_iterations, skip_threshold, slice_x, slice_y, slice_z,
                cuda_tol_batch_first, cuda_tol_batch_next,
                transport_from_args(amr_enabled, amr_ratio, ahe_enabled, ahe_ratio, picard_sweeps,
                                    picard_tolerance),
                solver_kind_from_string(solver), gmres_restart);
          },
          py::arg("manifest_path"),
          py::arg("contact_potentials"),
          py::arg("tolerance") = 1e-5,
          py::arg("max_iterations") = 2000,
          py::arg("skip_threshold") = 1e-5,
          py::arg("slice_x") = "",
          py::arg("slice_y") = "",
          py::arg("slice_z") = "",
          py::arg("cuda_tol_batch_first") = 1000,
          py::arg("cuda_tol_batch_next") = 500,
          py::arg("amr_enabled") = false,
          py::arg("amr_ratio") = 0.0,
          py::arg("ahe_enabled") = false,
          py::arg("ahe_ratio") = 0.0,
          py::arg("picard_sweeps") = 2,
          py::arg("picard_tolerance") = 0.0,
          py::arg("solver") = "gmres_cusparse",
          py::arg("gmres_restart") = 50)
      .def_static(
          "from_arrays",
          [](int nx,
             int ny,
             int nz,
             double cx,
             double cy,
             double cz,
             int first_r2_layer,
             double theta_sh,
             double decay_length,
             py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> region,
             py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> contact_id,
             py::array_t<float, py::array::c_style | py::array::forcecast> sigma,
             py::array_t<double, py::array::c_style | py::array::forcecast> potentials_array,
             double tolerance,
             int max_iterations,
             double skip_threshold,
             const std::string& slice_x,
             const std::string& slice_y,
             const std::string& slice_z,
             int cuda_tol_batch_first,
             int cuda_tol_batch_next,
             bool amr_enabled,
             double amr_ratio,
             bool ahe_enabled,
             double ahe_ratio,
             int picard_sweeps,
             double picard_tolerance,
             const std::string& solver,
             int gmres_restart) {
            ManifestData meta;
            meta.nx = nx;
            meta.ny = ny;
            meta.nz = nz;
            meta.cx = cx;
            meta.cy = cy;
            meta.cz = cz;
            meta.first_r2_layer = first_r2_layer;
            meta.theta_sh = theta_sh;
            meta.decay_length = decay_length;

            PoissonWorld world = PoissonWorld::from_arrays(
                meta,
                vector_from_3d_array<std::int8_t>(region, nz, ny, nx, "region"),
                vector_from_3d_array<std::int8_t>(contact_id, nz, ny, nx, "contact_id"),
                vector_from_3d_array<float>(sigma, nz, ny, nx, "sigma"));
            return std::make_unique<PoissonCudaSession>(
                std::move(world), contact_potentials_from_array(potentials_array), tolerance,
                max_iterations, skip_threshold, slice_x, slice_y, slice_z, cuda_tol_batch_first,
                cuda_tol_batch_next,
                transport_from_args(amr_enabled, amr_ratio, ahe_enabled, ahe_ratio, picard_sweeps,
                                    picard_tolerance),
                solver_kind_from_string(solver), gmres_restart);
          },
          py::arg("nx"),
          py::arg("ny"),
          py::arg("nz"),
          py::arg("cx"),
          py::arg("cy"),
          py::arg("cz"),
          py::arg("first_r2_layer"),
          py::arg("theta_sh"),
          py::arg("decay_length"),
          py::arg("region"),
          py::arg("contact_id"),
          py::arg("sigma"),
          py::arg("contact_potentials"),
          py::arg("tolerance") = 1e-5,
          py::arg("max_iterations") = 2000,
          py::arg("skip_threshold") = 1e-5,
          py::arg("slice_x") = "",
          py::arg("slice_y") = "",
          py::arg("slice_z") = "",
          py::arg("cuda_tol_batch_first") = 1000,
          py::arg("cuda_tol_batch_next") = 500,
          py::arg("amr_enabled") = false,
          py::arg("amr_ratio") = 0.0,
          py::arg("ahe_enabled") = false,
          py::arg("ahe_ratio") = 0.0,
          py::arg("picard_sweeps") = 2,
          py::arg("picard_tolerance") = 0.0,
          py::arg("solver") = "gmres_cusparse",
          py::arg("gmres_restart") = 50)
      .def_static(
          "from_signal_file",
          [](const std::string& manifest_path,
             const std::string& signal_path,
             int nt,
             double v_scale,
             int skip_first,
             int num_contacts,
             double tolerance,
             int max_iterations,
             double skip_threshold,
             const std::string& slice_x,
             const std::string& slice_y,
             const std::string& slice_z,
             int cuda_tol_batch_first,
             int cuda_tol_batch_next,
             bool amr_enabled,
             double amr_ratio,
             bool ahe_enabled,
             double ahe_ratio,
             int picard_sweeps,
             double picard_tolerance,
             const std::string& solver,
             int gmres_restart) {
            return std::make_unique<PoissonCudaSession>(
                PoissonWorld::load(manifest_path),
                load_signal_file_to_contact_potentials(signal_path, nt, v_scale, skip_first,
                                                       num_contacts),
                tolerance, max_iterations, skip_threshold, slice_x, slice_y, slice_z,
                cuda_tol_batch_first, cuda_tol_batch_next,
                transport_from_args(amr_enabled, amr_ratio, ahe_enabled, ahe_ratio, picard_sweeps,
                                    picard_tolerance),
                solver_kind_from_string(solver), gmres_restart);
          },
          py::arg("manifest_path"),
          py::arg("signal_path"),
          py::arg("nt"),
          py::arg("v_scale") = 0.005,
          py::arg("skip_first") = 1000,
          py::arg("num_contacts") = 3,
          py::arg("tolerance") = 1e-5,
          py::arg("max_iterations") = 2000,
          py::arg("skip_threshold") = 1e-5,
          py::arg("slice_x") = "",
          py::arg("slice_y") = "",
          py::arg("slice_z") = "",
          py::arg("cuda_tol_batch_first") = 1000,
          py::arg("cuda_tol_batch_next") = 500,
          py::arg("amr_enabled") = false,
          py::arg("amr_ratio") = 0.0,
          py::arg("ahe_enabled") = false,
          py::arg("ahe_ratio") = 0.0,
          py::arg("picard_sweeps") = 2,
          py::arg("picard_tolerance") = 0.0,
          py::arg("solver") = "gmres_cusparse",
          py::arg("gmres_restart") = 50)
      .def("iterate", &iterate_to_dict)
      .def("iterate_with_magnetization", &iterate_with_magnetization_to_dict,
           py::arg("magnetization"))
      .def("iterate_with_magnetization_variable", &iterate_with_magnetization_variable_to_dict,
           py::arg("magnetization"),
           py::arg("src_lo"),
           py::arg("src_hi"),
           py::arg("weight_hi"),
           py::arg("average_z") = false)
      .def("reset", &PoissonCudaSession::reset)
      .def(
          "set_hall_probe_indices",
          [](PoissonCudaSession& session, py::list high_y, py::list low_y) {
            session.set_hall_probe_indices(hall_probes_from_nested_arrays(high_y, low_y));
          },
          py::arg("high_y"),
          py::arg("low_y"))
      .def(
          "hall_potentials",
          [](const PoissonCudaSession& session) {
            return vector_to_numpy_1d(session.hall_potentials());
          })
      .def(
          "hall_potential_components",
          [](const PoissonCudaSession& session) {
            return hall_components_to_dict(session.hall_potential_components());
          })
      .def_property_readonly("hall_probes_configured", &PoissonCudaSession::hall_probes_configured)
      .def_property_readonly("hall_frame_available", &PoissonCudaSession::hall_frame_available)
      .def_property_readonly("last_frame_skipped", &PoissonCudaSession::last_frame_skipped)
      .def_property_readonly("current_step", &PoissonCudaSession::current_step)
      .def_property_readonly("n_steps", &PoissonCudaSession::n_steps)
      .def_property_readonly("exhausted", &PoissonCudaSession::exhausted)
      .def_property_readonly("unknown_count", &PoissonCudaSession::unknown_count)
      .def_property_readonly("num_contacts", &PoissonCudaSession::num_contacts)
      .def_property_readonly("transport_enabled", &PoissonCudaSession::transport_enabled)
      .def_property_readonly("amr_enabled", &PoissonCudaSession::amr_enabled)
      .def_property_readonly("ahe_enabled", &PoissonCudaSession::ahe_enabled)
      .def_property_readonly("amr_ratio", &PoissonCudaSession::amr_ratio)
      .def_property_readonly("ahe_ratio", &PoissonCudaSession::ahe_ratio)
      .def_property_readonly("picard_sweeps", &PoissonCudaSession::picard_sweeps)
      .def_property_readonly("solver",
                             [](const PoissonCudaSession& s) {
                               return solver_kind_to_string(s.solver_kind());
                             })
      .def_property_readonly("fm_layer_count", &PoissonCudaSession::fm_layer_count)
      .def_property_readonly("world_shape",
                             [](const PoissonCudaSession& s) {
                               return py::make_tuple(s.nz(), s.ny(), s.nx());
                             })
      .def_property_readonly(
          "buffer_shape",
          [](const PoissonCudaSession& s) {
            return py::make_tuple(s.out_nz(), s.out_ny(), s.out_nx(), 3);
          })
      .def_property_readonly("output_shape",
                             [](const PoissonCudaSession& s) {
                               return py::make_tuple(3, s.out_nz(), s.out_ny(), s.out_nx());
                             })
      .def_property_readonly("cellsize",
                             [](const PoissonCudaSession& s) {
                               return py::make_tuple(s.cx(), s.cy(), s.cz());
                             })
      .def_property_readonly("first_r2_layer", &PoissonCudaSession::first_r2_layer)
      .def_property_readonly("theta_sh", &PoissonCudaSession::theta_sh)
      .def_property_readonly("decay_length", &PoissonCudaSession::decay_length);
}

#include "wrappers.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "poisson_cuda_session.hpp"

namespace {

ContactPotentials contact_potentials_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> potentials_array) {
  const py::buffer_info info = potentials_array.request();
  if (info.ndim != 2 || info.shape[1] != 3) {
    throw std::invalid_argument("contact_potentials must have shape (nt, 3)");
  }
  if (info.shape[0] <= 0) {
    throw std::invalid_argument("contact_potentials must contain at least one row");
  }

  const auto* data = static_cast<const double*>(info.ptr);
  ContactPotentials potentials;
  const std::size_t nt = static_cast<std::size_t>(info.shape[0]);
  potentials.c0.resize(nt);
  potentials.c1.resize(nt);
  potentials.c2.resize(nt);
  for (std::size_t i = 0; i < nt; ++i) {
    potentials.c0[i] = static_cast<float>(data[i * 3u + 0u]);
    potentials.c1[i] = static_cast<float>(data[i * 3u + 1u]);
    potentials.c2[i] = static_cast<float>(data[i * 3u + 2u]);
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
  out["elapsed_s"] = stats.elapsed_s;
  out["note"] = stats.stats_note;
  return out;
}

py::dict iterate_to_dict(PoissonCudaSession& session) {
  StepStats stats;
  {
    py::gil_scoped_release release;
    stats = session.iterate();
  }

  py::dict out;
  out["jmod"] = frame_to_numpy_mumax_copy(session.jmod_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  out["jcur"] = frame_to_numpy_mumax_copy(session.jcur_frame(), session.out_nz(), session.out_ny(),
                                          session.out_nx());
  out["stats"] = stats_to_dict(stats);
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
             int cuda_tol_batch_next) {
            return std::make_unique<PoissonCudaSession>(
                PoissonWorld::load(manifest_path), contact_potentials_from_array(potentials_array),
                tolerance, max_iterations, skip_threshold, slice_x, slice_y, slice_z,
                cuda_tol_batch_first, cuda_tol_batch_next);
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
          py::arg("cuda_tol_batch_next") = 500)
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
             int cuda_tol_batch_next) {
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
                cuda_tol_batch_next);
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
          py::arg("cuda_tol_batch_next") = 500)
      .def_static(
          "from_signal_file",
          [](const std::string& manifest_path,
             const std::string& signal_path,
             int nt,
             double v_scale,
             int skip_first,
             double tolerance,
             int max_iterations,
             double skip_threshold,
             const std::string& slice_x,
             const std::string& slice_y,
             const std::string& slice_z,
             int cuda_tol_batch_first,
             int cuda_tol_batch_next) {
            return std::make_unique<PoissonCudaSession>(
                PoissonWorld::load(manifest_path),
                load_signal_file_to_contact_potentials(signal_path, nt, v_scale, skip_first),
                tolerance, max_iterations, skip_threshold, slice_x, slice_y, slice_z,
                cuda_tol_batch_first, cuda_tol_batch_next);
          },
          py::arg("manifest_path"),
          py::arg("signal_path"),
          py::arg("nt"),
          py::arg("v_scale") = 0.005,
          py::arg("skip_first") = 1000,
          py::arg("tolerance") = 1e-5,
          py::arg("max_iterations") = 2000,
          py::arg("skip_threshold") = 1e-5,
          py::arg("slice_x") = "",
          py::arg("slice_y") = "",
          py::arg("slice_z") = "",
          py::arg("cuda_tol_batch_first") = 1000,
          py::arg("cuda_tol_batch_next") = 500)
      .def("iterate", &iterate_to_dict)
      .def("reset", &PoissonCudaSession::reset)
      .def_property_readonly("current_step", &PoissonCudaSession::current_step)
      .def_property_readonly("n_steps", &PoissonCudaSession::n_steps)
      .def_property_readonly("exhausted", &PoissonCudaSession::exhausted)
      .def_property_readonly("unknown_count", &PoissonCudaSession::unknown_count)
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

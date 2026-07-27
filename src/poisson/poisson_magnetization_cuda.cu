#include "poisson_magnetization_cuda.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

__device__ void normalize(float& mx, float& my, float& mz) {
  const float n = sqrtf(mx * mx + my * my + mz * mz);
  if (n > 1e-12f) {
    mx /= n;
    my /= n;
    mz /= n;
  } else {
    mx = 0.0f;
    my = 0.0f;
    mz = 0.0f;
  }
}

__global__ void k_map_magnetization(const float* mx_in,
                                    const float* my_in,
                                    const float* mz_in,
                                    int src_nz,
                                    int ny,
                                    int nx,
                                    int dst_nz,
                                    const int* src_lo,
                                    const int* src_hi,
                                    const float* weight_hi,
                                    bool average_z,
                                    float* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = ny * nx;
  const int total = dst_nz * plane;
  if (idx >= total) {
    return;
  }

  const int iz = idx / plane;
  const int xy = idx - iz * plane;
  float mx = 0.0f;
  float my = 0.0f;
  float mz = 0.0f;

  if (average_z) {
    for (int z = 0; z < src_nz; ++z) {
      const int src = z * plane + xy;
      mx += mx_in[src];
      my += my_in[src];
      mz += mz_in[src];
    }
    const float inv = 1.0f / static_cast<float>(src_nz);
    mx *= inv;
    my *= inv;
    mz *= inv;
  } else {
    const int lo = src_lo[iz];
    const int hi = src_hi[iz];
    const float wh = weight_hi[iz];
    const int src0 = lo * plane + xy;
    const int src1 = hi * plane + xy;
    const float wl = 1.0f - wh;
    mx = wl * mx_in[src0] + wh * mx_in[src1];
    my = wl * my_in[src0] + wh * my_in[src1];
    mz = wl * mz_in[src0] + wh * mz_in[src1];
  }

  normalize(mx, my, mz);
  const int n_xy = dst_nz * plane;
  out[idx] = mx;
  out[n_xy + idx] = my;
  out[2 * n_xy + idx] = mz;
}

}  // namespace

void map_device_magnetization_to_device_stack(const float* d_mx,
                                              const float* d_my,
                                              const float* d_mz,
                                              int src_nz,
                                              int ny,
                                              int nx,
                                              int dst_nz,
                                              const int* d_src_lo,
                                              const int* d_src_hi,
                                              const float* d_weight_hi,
                                              bool average_z,
                                              float* d_out) {
  const int plane = ny * nx;
  const int total = dst_nz * plane;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  k_map_magnetization<<<blocks, threads>>>(d_mx, d_my, d_mz, src_nz, ny, nx, dst_nz, d_src_lo,
                                           d_src_hi, d_weight_hi, average_z, d_out);
  check_cuda(cudaGetLastError(), "k_map_magnetization launch");
}

void map_device_magnetization_to_host_stack(const float* d_mx,
                                            const float* d_my,
                                            const float* d_mz,
                                            int src_nz,
                                            int ny,
                                            int nx,
                                            int dst_nz,
                                            const std::vector<int>& src_lo,
                                            const std::vector<int>& src_hi,
                                            const std::vector<float>& weight_hi,
                                            bool average_z,
                                            std::vector<float>& out) {
  if (src_nz <= 0 || ny <= 0 || nx <= 0 || dst_nz <= 0) {
    throw std::invalid_argument("invalid magnetization device mapping shape");
  }
  if (!average_z &&
      (static_cast<int>(src_lo.size()) != dst_nz || static_cast<int>(src_hi.size()) != dst_nz ||
       static_cast<int>(weight_hi.size()) != dst_nz)) {
    throw std::invalid_argument("magnetization device mapping arrays must match FM layer count");
  }

  int* d_lo = nullptr;
  int* d_hi = nullptr;
  float* d_weight = nullptr;
  float* d_out = nullptr;
  const int plane = ny * nx;
  const std::size_t dst_values = 3u * static_cast<std::size_t>(dst_nz) *
                                 static_cast<std::size_t>(plane);
  out.assign(dst_values, 0.0f);

  if (!average_z) {
    check_cuda(cudaMalloc(&d_lo, static_cast<std::size_t>(dst_nz) * sizeof(int)),
               "cudaMalloc magnetization lo");
    check_cuda(cudaMalloc(&d_hi, static_cast<std::size_t>(dst_nz) * sizeof(int)),
               "cudaMalloc magnetization hi");
    check_cuda(cudaMalloc(&d_weight, static_cast<std::size_t>(dst_nz) * sizeof(float)),
               "cudaMalloc magnetization weight");
    check_cuda(cudaMemcpy(d_lo, src_lo.data(), static_cast<std::size_t>(dst_nz) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy magnetization lo");
    check_cuda(cudaMemcpy(d_hi, src_hi.data(), static_cast<std::size_t>(dst_nz) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy magnetization hi");
    check_cuda(cudaMemcpy(d_weight, weight_hi.data(),
                          static_cast<std::size_t>(dst_nz) * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy magnetization weight");
  }

  check_cuda(cudaMalloc(&d_out, dst_values * sizeof(float)), "cudaMalloc magnetization stack");
  map_device_magnetization_to_device_stack(d_mx, d_my, d_mz, src_nz, ny, nx, dst_nz, d_lo, d_hi,
                                           d_weight, average_z, d_out);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize magnetization map");
  check_cuda(cudaMemcpy(out.data(), d_out, dst_values * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy magnetization stack out");

  cudaFree(d_lo);
  cudaFree(d_hi);
  cudaFree(d_weight);
  cudaFree(d_out);
}

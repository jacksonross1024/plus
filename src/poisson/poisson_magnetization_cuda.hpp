#pragma once

#include <vector>

// Map mumax device magnetization components onto a Poisson FM stack layout
// (3, dst_nz, ny, nx) flattened as [mx | my | mz], writing directly to d_out.
// d_src_lo / d_src_hi / d_weight_hi may be null when average_z is true.
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
                                              float* d_out);

// Convenience: device map then D2H into out. Mapping tables are uploaded each call;
// prefer map_device_magnetization_to_device_stack with cached device tables on hot paths.
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
                                            std::vector<float>& out);

// Map mumax device applied-field components onto the full Poisson grid
// (3, dst_nz, ny, nx) including Pt, flattened as [bx | by | bz]. Same z-extend
// style as magnetization, without unit-vector normalisation. Device-to-device.
void map_device_applied_field_to_poisson_grid(const float* d_bx,
                                              const float* d_by,
                                              const float* d_bz,
                                              int src_nz,
                                              int ny,
                                              int nx,
                                              int dst_nz,
                                              const int* d_src_lo,
                                              const int* d_src_hi,
                                              const float* d_weight_hi,
                                              bool average_z,
                                              float* d_out);

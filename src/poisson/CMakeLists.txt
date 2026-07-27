add_library(poisson_native STATIC
    poisson_world.cpp
    poisson_current.cpp
    signal_loader.cpp
    poisson_pcg_cuda.cu
    poisson_gmres_cuda.cu
    poisson_magnetization_cuda.cu
    poisson_cuda_session.cpp
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(poisson_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(poisson_native PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(poisson_native PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(poisson_native PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)

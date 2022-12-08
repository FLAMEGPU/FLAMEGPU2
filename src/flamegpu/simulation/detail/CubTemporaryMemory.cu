#include "flamegpu/simulation/detail/CubTemporaryMemory.cuh"
#include <cuda_runtime.h>

#include <cassert>

#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {
namespace detail {

CubTemporaryMemory::CubTemporaryMemory()
  : d_cub_temp(nullptr)
  , d_cub_temp_size(0) { }
CubTemporaryMemory::~CubTemporaryMemory() {
    // @todo - cuda is not allowed in destructor
    if (d_cub_temp) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_cub_temp));
        d_cub_temp_size = 0;
    }
}
void CubTemporaryMemory::resize(const size_t newSize) {
    if (newSize > d_cub_temp_size) {
        flamegpu::util::nvtx::Range range{"CubTemporaryMemory::resizeTempStorage"};
        if (d_cub_temp) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_cub_temp));
        }
        gpuErrchk(cudaMalloc(&d_cub_temp, newSize));
        d_cub_temp_size = newSize;
    }
}

}  // namespace detail
}  // namespace flamegpu

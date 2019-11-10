#include "DeviceRandomArray.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include<ctime>

#include <cassert>
#include <cstdio>
#include <algorithm>

#include "flamegpu/gpu/CUDAErrorChecking.h"

/**
 * Internal namespace to hide __device__ declarations from modeller
 */
namespace flamegpu_internal {
    /**
     * Device array holding curand states
     * They should always be initialised
     */
    __device__ curandState *d_random_state;
    /**
     * Device copy of the length of d_random_state
     */
    __device__ DeviceRandomArray::size_type d_random_size;
    /**
     * Host mirror of d_random_state
     */
    curandState *hd_random_state;
    /**
     * Host mirror of d_random_size
     */
    DeviceRandomArray::size_type hd_random_size;
}  // namespace flamegpu_internal

/**
 * Static member vars
 */
uint64_t DeviceRandomArray::mSeed = 0;
DeviceRandomArray::size_type DeviceRandomArray::length = 0;
const DeviceRandomArray::size_type DeviceRandomArray::min_length = 1024;
float DeviceRandomArray::growthModifier = 1.5;
float DeviceRandomArray::shrinkModifier = 1.0;
curandState *DeviceRandomArray::h_max_random_state = nullptr;
DeviceRandomArray::size_type DeviceRandomArray::h_max_random_size = 0;
/**
 * Member fns
 */
uint64_t DeviceRandomArray::seedFromTime() {
    return static_cast<uint64_t>(time(nullptr));
}
void DeviceRandomArray::init(const uint64_t &seed) {
    DeviceRandomArray::mSeed = seed;
    free();
}
void DeviceRandomArray::free() {
    // Clear size
    length = 0;
    flamegpu_internal::hd_random_size = 0;
    gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::d_random_size, &flamegpu_internal::hd_random_size, sizeof(DeviceRandomArray::size_type)))
        printf("(%s:%d) CUDA Error initialising curand.", __FILE__, __LINE__);
    // Release old
    if (flamegpu_internal::hd_random_state != nullptr) {
        gpuErrchk(cudaFree(flamegpu_internal::hd_random_state));
    }
    // Update pointers
    flamegpu_internal::hd_random_state = nullptr;
    gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::d_random_state, &flamegpu_internal::hd_random_state, sizeof(curandState*)))
    // Release host_max
    if (h_max_random_state)
        ::free(h_max_random_state);
    h_max_random_size = 0;
}

bool DeviceRandomArray::resize(const size_type &_length) {
    assert(growthModifier > 1.0);
    assert(shrinkModifier > 0.0);
    assert(shrinkModifier <= 1.0);
    auto t_length = length;
    if (length) {
        while (t_length < _length) {
            t_length = static_cast<DeviceRandomArray::size_type>(t_length * growthModifier);
            if (shrinkModifier < 1.0f) {
                while (t_length * shrinkModifier > _length) {
                    t_length = static_cast<DeviceRandomArray::size_type>(t_length * shrinkModifier);
                }
            }
        }
    } else {  // Special case for first run
        t_length = _length;
    }
    // Don't allow array to go below DeviceRandomArray::min_length elements
    t_length = std::max<size_type>(t_length, DeviceRandomArray::min_length);
    if (t_length != length)
        resizeDeviceArray(t_length);
    return t_length != length;
}
__global__ void init_curand(unsigned int threadCount, uint64_t seed, DeviceRandomArray::size_type offset) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < threadCount)
        curand_init(seed, offset + id, 0, &flamegpu_internal::d_random_state[offset + id]);
}
void DeviceRandomArray::resizeDeviceArray(const size_type &_length) {
    if (_length > length) {
        // Growing array
        curandState *t_hd_random_state = nullptr;
        // Allocate new mem to t_hd
        gpuErrchk(cudaMalloc(&t_hd_random_state, _length * sizeof(curandState)));
            printf("(%s:%d) CUDA Error DeviceRandomArray::resizeDeviceArray().", __FILE__, __LINE__);
        // Copy hd->t_hd[****    ]
        if (flamegpu_internal::hd_random_state) {
            gpuErrchk(cudaMemcpy(t_hd_random_state, flamegpu_internal::hd_random_state, length * sizeof(curandState), cudaMemcpyDeviceToDevice));
        }
        // Update pointers hd=t_hd
        if (flamegpu_internal::hd_random_state) {
            gpuErrchk(cudaFree(flamegpu_internal::hd_random_state));
        }
        flamegpu_internal::hd_random_state = t_hd_random_state;
        gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::d_random_state, &flamegpu_internal::hd_random_state, sizeof(curandState*)));
        // Init new[    ****]
        if (h_max_random_size > length) {
            // We have part/all host backup, copy to device array
            // Reinit backup[    **  ]
            size_type copy_len = std::min(h_max_random_size, _length);
            gpuErrchk(cudaMemcpy(flamegpu_internal::hd_random_state + length, h_max_random_state + length, copy_len * sizeof(curandState), cudaMemcpyHostToDevice));
            length += copy_len;
        }
        if (_length > length) {
            // Init remainder[     **]
            unsigned int initThreads = 512;
            unsigned int initBlocks = (_length - length / initThreads) + 1;
            init_curand<<<initBlocks, initThreads>>>(_length - length, mSeed, length);  // This could be async with above memcpy?
            gpuErrchkLaunch();
        }
    } else {
        // Shrinking array
        curandState *t_hd_random_state = nullptr;
        curandState *t_h_max_random_state = nullptr;
        // Allocate new
        gpuErrchk(cudaMalloc(&t_hd_random_state, _length * sizeof(curandState)));
        // Allocate host backup
        if (length > h_max_random_size)
            t_h_max_random_state = reinterpret_cast<curandState *>(malloc(length * sizeof(curandState)));
        else
            t_h_max_random_state = h_max_random_state;
        // Copy old->new
        assert(flamegpu_internal::hd_random_state);
        gpuErrchk(cudaMemcpy(t_hd_random_state, flamegpu_internal::hd_random_state, _length * sizeof(curandState), cudaMemcpyDeviceToDevice));
        // Copy part being shrunk away to host storage (This could be async with above memcpy?)
        gpuErrchk(cudaMemcpy(t_h_max_random_state + _length, flamegpu_internal::hd_random_state + _length, (length - _length) * sizeof(curandState), cudaMemcpyDeviceToHost));
        // Release and replace old host ptr
        if (length > h_max_random_size) {
            if (h_max_random_state)
                ::free(h_max_random_state);
            h_max_random_state = t_h_max_random_state;
            h_max_random_size = length;
        }
        // Update pointers
        flamegpu_internal::hd_random_state = t_hd_random_state;
        gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::d_random_state, &flamegpu_internal::hd_random_state, sizeof(curandState*)));
        // Release old
        if (flamegpu_internal::hd_random_state != nullptr) {
            gpuErrchk(cudaFree(flamegpu_internal::hd_random_state));
        }
    }
    // Update length
    length = _length;
    flamegpu_internal::hd_random_size = _length;
    gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::d_random_size, &flamegpu_internal::hd_random_size, sizeof(DeviceRandomArray::size_type)));
}
void DeviceRandomArray::setGrowthModifier(float _growthModifier) {
    assert(growthModifier > 1.0);
    DeviceRandomArray::growthModifier = _growthModifier;
}
float DeviceRandomArray::getGrowthModifier() {
    return DeviceRandomArray::growthModifier;
}
void DeviceRandomArray::setShrinkModifier(float _shrinkModifier) {
    assert(shrinkModifier > 0.0);
    assert(shrinkModifier <= 1.0);
    DeviceRandomArray::shrinkModifier = _shrinkModifier;
}
float DeviceRandomArray::getShrinkModifier() {
    return DeviceRandomArray::shrinkModifier;
}
DeviceRandomArray::size_type DeviceRandomArray::size() {
    return length;
}
uint64_t DeviceRandomArray::seed() {
    return mSeed;
}

#include "flamegpu/simulation/detail/RandomManager.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include<ctime>

#include <cassert>
#include <cstdio>
#include <algorithm>

#include "flamegpu/detail/curand.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {
namespace detail {

RandomManager::RandomManager() :
    deviceInitialised(false) {
    reseed(static_cast<uint64_t>(seedFromTime() % UINT_MAX));
}
RandomManager::~RandomManager() {
    free();  // @todo call free/freeDevice not in the constructor! instead just log that?
}
/**
 * Member fns
 */
uint64_t RandomManager::seedFromTime() {
    return static_cast<uint64_t>(time(nullptr));
}

void RandomManager::reseedHost() {
    freeHost();
    host_rng = std::mt19937_64();
    // Reset host random generator/s
    host_rng.seed(mSeed);
}

void RandomManager::reseedDevice() {
    freeDevice();
    // curand is initialised on access if length does not match. This would need a second device length?
}

void RandomManager::reseed(const uint64_t seed) {
    // Set the instance's seed to the new value
    mSeed = seed;

    // Apply the new seed to the host
    reseedHost();
    // Apply the new seed to the device.
    reseedDevice();
}

void RandomManager::freeHost() {
    // Release host_max
    if (h_max_random_state) {
        std::free(h_max_random_state);
        h_max_random_state = nullptr;
    }
    h_max_random_size = 0;
}

void RandomManager::freeDevice() {
    // Clear size - length is just for the device portion?
    length = 0;

    if (deviceInitialised) {
        // Set the device's internal size to 0.
        length = 0;
        // Release old random states on the deivce and update pointers.
        if (d_random_state) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_random_state));
        }
        d_random_state = nullptr;
    }
}

void RandomManager::free() {
    // Free the host and device.
    freeHost();
    freeDevice();
}

detail::curandState *RandomManager::resize(size_type _length, cudaStream_t stream) {
    assert(growthModifier > 1.0);
    assert(shrinkModifier > 0.0);
    assert(shrinkModifier <= 1.0);
    auto t_length = length;
    if (length) {
        while (t_length < _length) {
            t_length = static_cast<flamegpu::size_type>(t_length * growthModifier);
            if (shrinkModifier < 1.0f) {
                while (t_length * shrinkModifier > _length) {
                    t_length = static_cast<flamegpu::size_type>(t_length * shrinkModifier);
                }
            }
        }
    } else {  // Special case for first run
        t_length = _length;
    }
    // Don't allow array to go below RandomManager::min_length elements
    t_length = std::max<size_type>(t_length, RandomManager::min_length);
    if (t_length != length)
        resizeDeviceArray(t_length, stream);
    return d_random_state;
}
__global__ void init_curand(detail::curandState *d_random_state, unsigned int threadCount, uint64_t seed, flamegpu::size_type offset) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < threadCount)
        curand_init(seed, offset + id, 0, &d_random_state[offset + id]);
}
void RandomManager::resizeDeviceArray(const size_type _length, cudaStream_t stream) {
    // Mark that the device hsa now been initialised.
    deviceInitialised = true;
    if (_length > h_max_random_size) {
        // Growing array
        detail::curandState *t_hd_random_state = nullptr;
        // Allocate new mem to t_hd
        gpuErrchk(cudaMalloc(&t_hd_random_state, _length * sizeof(detail::curandState)));
        // Copy hd->t_hd[****    ]
        if (d_random_state) {
            gpuErrchk(cudaMemcpyAsync(t_hd_random_state, d_random_state, length * sizeof(detail::curandState), cudaMemcpyDeviceToDevice, stream));
        }
        // Update pointers hd=t_hd
        if (d_random_state) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_random_state));
        }
        d_random_state = t_hd_random_state;
        // Init new[    ****]
        if (h_max_random_size > length) {
            // We have part/all host backup, copy to device array
            // Reinit backup[    **  ]
            const size_type copy_len = std::min(h_max_random_size, _length);
            gpuErrchk(cudaMemcpyAsync(d_random_state + length, h_max_random_state + length, copy_len * sizeof(detail::curandState), cudaMemcpyHostToDevice, stream));  // Host not pinned
            length += copy_len;
        }
        if (_length > length) {
            // Init remainder[     **]
            unsigned int initThreads = 512;
            unsigned int initBlocks = ((_length - length) / initThreads) + 1;
            init_curand<<<initBlocks, initThreads, 0,  stream>>>(d_random_state, _length - length, mSeed, length);  // This could be async with above memcpy in diff stream
            gpuErrchkLaunch();
        }
    } else {
        // Shrinking array
        detail::curandState *t_hd_random_state = nullptr;
        detail::curandState *t_h_max_random_state = nullptr;
        // Allocate new
        gpuErrchk(cudaMalloc(&t_hd_random_state, _length * sizeof(detail::curandState)));
        // Allocate host backup
        if (length > h_max_random_size)
            t_h_max_random_state = reinterpret_cast<detail::curandState*>(malloc(length * sizeof(detail::curandState)));
        else
            t_h_max_random_state = h_max_random_state;
        // Copy old->new
        assert(d_random_state);
        gpuErrchk(cudaMemcpyAsync(t_hd_random_state, d_random_state, _length * sizeof(detail::curandState), cudaMemcpyDeviceToDevice, stream));
        // Copy part being shrunk away to host storage (This could be async with above memcpy?)
        gpuErrchk(cudaMemcpyAsync(t_h_max_random_state + _length, d_random_state + _length, (length - _length) * sizeof(detail::curandState), cudaMemcpyDeviceToHost, stream));
        // Release and replace old host ptr
        if (length > h_max_random_size) {
            if (h_max_random_state)
                ::free(h_max_random_state);
            h_max_random_state = t_h_max_random_state;
            h_max_random_size = length;
        }
        // Release old
        if (d_random_state != nullptr) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_random_state));
        }
        // Update pointer
        d_random_state = t_hd_random_state;
    }
    // Update length
    length = _length;
    gpuErrchk(cudaStreamSynchronize(stream));
}
void RandomManager::setGrowthModifier(float _growthModifier) {
    assert(growthModifier > 1.0);
    RandomManager::growthModifier = _growthModifier;
}
float RandomManager::getGrowthModifier() {
    return RandomManager::growthModifier;
}
void RandomManager::setShrinkModifier(float _shrinkModifier) {
    assert(shrinkModifier > 0.0);
    assert(shrinkModifier <= 1.0);
    RandomManager::shrinkModifier = _shrinkModifier;
}
float RandomManager::getShrinkModifier() {
    return RandomManager::shrinkModifier;
}
flamegpu::size_type RandomManager::size() {
    return length;
}
uint64_t RandomManager::seed() {
    return mSeed;
}
detail::curandState *RandomManager::cudaRandomState() {
    return d_random_state;
}

}  // namespace detail
}  // namespace flamegpu

#include "flamegpu/runtime/utility/RandomManager.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include<ctime>

#include <cassert>
#include <cstdio>
#include <algorithm>

#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/gpu/CUDASimulation.h"

namespace flamegpu {

RandomManager::RandomManager() :
    deviceInitialised(false) {
    reseed(static_cast<uint64_t>(seedFromTime() % UINT_MAX));
}
RandomManager::~RandomManager() {
    free();  // @todo call free/freeDevice not in the constructor! instead just log that?
}
void RandomManager::purge() {
    length = 0;
    d_random_state = nullptr;
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

void RandomManager::reseed(const uint64_t &seed) {
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
            gpuErrchk(cudaFree(d_random_state));
        }
        d_random_state = nullptr;
    }
}

void RandomManager::free() {
    // Free the host and device.
    freeHost();
    freeDevice();
}

curandState *RandomManager::resize(const size_type &_length) {
    assert(growthModifier > 1.0);
    assert(shrinkModifier > 0.0);
    assert(shrinkModifier <= 1.0);
    auto t_length = length;
    if (length) {
        while (t_length < _length) {
            t_length = static_cast<RandomManager::size_type>(t_length * growthModifier);
            if (shrinkModifier < 1.0f) {
                while (t_length * shrinkModifier > _length) {
                    t_length = static_cast<RandomManager::size_type>(t_length * shrinkModifier);
                }
            }
        }
    } else {  // Special case for first run
        t_length = _length;
    }
    // Don't allow array to go below RandomManager::min_length elements
    t_length = std::max<size_type>(t_length, RandomManager::min_length);
    if (t_length != length)
        resizeDeviceArray(t_length);
    return d_random_state;
}
__global__ void init_curand(curandState *d_random_state, unsigned int threadCount, uint64_t seed, RandomManager::size_type offset) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < threadCount)
        curand_init(seed, offset + id, 0, &d_random_state[offset + id]);
}
void RandomManager::resizeDeviceArray(const size_type &_length) {
    // Mark that the device hsa now been initialised.
    deviceInitialised = true;
    if (_length > h_max_random_size) {
        // Growing array
        curandState *t_hd_random_state = nullptr;
        // Allocate new mem to t_hd
        gpuErrchk(cudaMalloc(&t_hd_random_state, _length * sizeof(curandState)));
        // Copy hd->t_hd[****    ]
        if (d_random_state) {
            gpuErrchk(cudaMemcpy(t_hd_random_state, d_random_state, length * sizeof(curandState), cudaMemcpyDeviceToDevice));
        }
        // Update pointers hd=t_hd
        if (d_random_state) {
            gpuErrchk(cudaFree(d_random_state));
        }
        d_random_state = t_hd_random_state;
        // Init new[    ****]
        if (h_max_random_size > length) {
            // We have part/all host backup, copy to device array
            // Reinit backup[    **  ]
            const size_type copy_len = std::min(h_max_random_size, _length);
            gpuErrchk(cudaMemcpy(d_random_state + length, h_max_random_state + length, copy_len * sizeof(curandState), cudaMemcpyHostToDevice));
            length += copy_len;
        }
        if (_length > length) {
            // Init remainder[     **]
            unsigned int initThreads = 512;
            unsigned int initBlocks = ((_length - length) / initThreads) + 1;
            init_curand<<<initBlocks, initThreads>>>(d_random_state, _length - length, mSeed, length);  // This could be async with above memcpy?
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
        assert(d_random_state);
        gpuErrchk(cudaMemcpy(t_hd_random_state, d_random_state, _length * sizeof(curandState), cudaMemcpyDeviceToDevice));
        // Copy part being shrunk away to host storage (This could be async with above memcpy?)
        gpuErrchk(cudaMemcpy(t_h_max_random_state + _length, d_random_state + _length, (length - _length) * sizeof(curandState), cudaMemcpyDeviceToHost));
        // Release and replace old host ptr
        if (length > h_max_random_size) {
            if (h_max_random_state)
                ::free(h_max_random_state);
            h_max_random_state = t_h_max_random_state;
            h_max_random_size = length;
        }
        // Release old
        if (d_random_state != nullptr) {
            gpuErrchk(cudaFree(d_random_state));
        }
        // Update pointer
        d_random_state = t_hd_random_state;
    }
    // Update length
    length = _length;
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
RandomManager::size_type RandomManager::size() {
    return length;
}
uint64_t RandomManager::seed() {
    return mSeed;
}
curandState *RandomManager::cudaRandomState() {
    return d_random_state;
}

}  // namespace flamegpu

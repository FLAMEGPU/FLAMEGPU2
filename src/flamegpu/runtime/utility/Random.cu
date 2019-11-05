#include "Random.cuh"

#include<ctime>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <cstdio>
#include <algorithm>

/**
 * Internal namespace to hide __device__ declarations from modeller
 */
namespace flamegpu_internal {
	__device__ curandState *d_random_state;
	__device__ Random::size_type d_random_size;
    curandState *hd_random_state;
    Random::size_type hd_random_size;
}
/**
 * Static member vars
 */
unsigned long long Random::mSeed = 0;
Random::size_type Random::length = 0;
Random::size_type Random::min_length = 1024;
float Random::growthModifier = 1.5;
float Random::shrinkModifier = 1.0;
curandState *Random::h_max_random_state = nullptr;
Random::size_type Random::h_max_random_size = 0;
/**
 * Member fns
 */
unsigned long long Random::seedFromTime() {
    return static_cast<unsigned long long>(time(nullptr));
}
void Random::init(const unsigned long long &seed) {
    Random::mSeed = seed;
    free();
}
void Random::free() {   
    // Clear size
    length = 0;
    flamegpu_internal::hd_random_size = 0;
    if (cudaMemcpyToSymbol(flamegpu_internal::d_random_size, &flamegpu_internal::hd_random_size, sizeof(Random::size_type)) != cudaSuccess)
        printf("(%s:%d) CUDA Error initialising curand.", __FILE__, __LINE__);
    // Release old
    if (flamegpu_internal::hd_random_state != nullptr && cudaFree(flamegpu_internal::hd_random_state) != cudaSuccess)
        printf("(%s:%d) CUDA Error Random::~Random().", __FILE__, __LINE__);
    // Update pointers
    flamegpu_internal::hd_random_state = nullptr;
    if (cudaMemcpyToSymbol(flamegpu_internal::d_random_state, &flamegpu_internal::hd_random_state, sizeof(curandState*)) != cudaSuccess)
        printf("(%s:%d) CUDA Error Random::~Random().", __FILE__, __LINE__);
    // Release host_max
    if (h_max_random_state)
        ::free(h_max_random_state);
    h_max_random_size = 0;
}

bool Random::resize(const size_type &_length) {
    auto t_length = length;
    if (length)
    {
        while (t_length < _length) {
            t_length = static_cast<Random::size_type>(t_length * growthModifier);
            if (shrinkModifier < 1.0f) {
                while (t_length * shrinkModifier > _length) {
                    t_length = static_cast<Random::size_type>(t_length * shrinkModifier);
                }
            }
        }
    }
    else //Special case for first run
        t_length = _length;
    //Don't allow array to go below Random::min_length elements
    t_length = std::max<size_type>(t_length, Random::min_length);
    if(t_length != length)
        resizeDeviceArray(t_length);
    return t_length != length;
}
__global__ void init_curand(unsigned long threadCount, unsigned long long seed, Random::size_type offset) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < threadCount)
        curand_init(seed, offset + id, 0, &flamegpu_internal::d_random_state[offset + id]);
}
void Random::resizeDeviceArray(const size_type &_length) {
    if(_length > length) {
        // Growing array
        curandState *t_hd_random_state = nullptr;
        // Allocate new
        if (cudaMalloc(&t_hd_random_state, _length * sizeof(curandState)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        // Copy old->new[****    ]
        if (flamegpu_internal::hd_random_state)
            if (cudaMemcpy(t_hd_random_state, flamegpu_internal::hd_random_state, length * sizeof(curandState), cudaMemcpyDeviceToDevice))
                printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        // Update pointers
        if(flamegpu_internal::hd_random_state)
            if (cudaFree(flamegpu_internal::hd_random_state) != cudaSuccess)
                printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        flamegpu_internal::hd_random_state = t_hd_random_state;
        if (cudaMemcpyToSymbol(flamegpu_internal::d_random_state, &flamegpu_internal::hd_random_state, sizeof(curandState*)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        // Init new[    ****]
        if (h_max_random_size > length) {
            //We have part/all host backup, copy to device array
            size_type copy_len = std::min(h_max_random_size, _length);
            if (cudaMemcpy(t_hd_random_state + length, flamegpu_internal::hd_random_state + length, copy_len * sizeof(curandState), cudaMemcpyDeviceToDevice))
                printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
            length += copy_len;
        }
        if (_length > length)
        {
            //Init remainder for first time
            unsigned int initThreads = 512;
            unsigned int initBlocks = (_length - length / initThreads) + 1;
            init_curand<<<initBlocks, initThreads>>>(_length - length, mSeed, length);// This could be async with above memcpy?
        }
    } else {
        // Shrinking array
        curandState *t_hd_random_state = nullptr;
        curandState *t_h_max_random_state = nullptr;
        // Allocate new
        if (cudaMalloc(&t_hd_random_state, _length * sizeof(curandState)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        // Allocate host backup
        if (length > h_max_random_size)
            t_h_max_random_state = reinterpret_cast<curandState *>(malloc(length * sizeof(curandState)));
        else
            t_h_max_random_state = h_max_random_state;
        // Copy old->new
        assert(flamegpu_internal::hd_random_state);
        if (cudaMemcpy(t_hd_random_state, flamegpu_internal::hd_random_state, _length * sizeof(curandState), cudaMemcpyDeviceToDevice))
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        // Copy part being shrunk away to host storage (This could be async with above memcpy?)
        if (cudaMemcpy(t_h_max_random_state + _length, flamegpu_internal::hd_random_state + _length, (length - _length) * sizeof(curandState), cudaMemcpyDeviceToHost))
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        // Release and replace old host ptr
        if (length > h_max_random_size) {
            if(h_max_random_state)
                ::free(h_max_random_state);
            h_max_random_state = t_h_max_random_state;
            h_max_random_size = length;
        }
        // Update pointers
        flamegpu_internal::hd_random_state = t_hd_random_state;
        if (cudaMemcpyToSymbol(flamegpu_internal::d_random_state, &flamegpu_internal::hd_random_state, sizeof(curandState*)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        //Release old
        if (flamegpu_internal::hd_random_state!=nullptr && cudaFree(flamegpu_internal::hd_random_state) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
    }
    // Update length
    length = _length;
    flamegpu_internal::hd_random_size = _length;
    if (cudaMemcpyToSymbol(flamegpu_internal::d_random_size, &flamegpu_internal::hd_random_size, sizeof(Random::size_type)) != cudaSuccess)
        printf("(%s:%d) CUDA Error initialising curand.", __FILE__, __LINE__);
}
void Random::setGrowthModifier(float _growthModifier) {
    assert(growthModifier > 1.0);
    Random::growthModifier = _growthModifier;
}
float Random::getGrowthModifier() {
    return Random::growthModifier;
}
void Random::setShrinkModifier(float _shrinkModifier) {
    assert(shrinkModifier > 0.0);
    assert(shrinkModifier <= 1.0);
    Random::shrinkModifier = _shrinkModifier;
}
float Random::getShrinkModifier() {
    return Random::shrinkModifier;
}
Random::size_type Random::size() {
    return length;
}
unsigned long long Random::seed() {
    return mSeed;
}
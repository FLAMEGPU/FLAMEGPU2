#ifndef __AgentRandom_cuh__
#define __AgentRandom_cuh__

#include <cassert>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

/**
 * Utility for accessing random generation within agent functions
 * Wraps curand
 */
class AgentRandom {
 public:
    typedef unsigned int size_type;

    __forceinline__ __device__ AgentRandom();
    /**
     * Returns a float uniformly distributed between 0.0 and 1.0. 
     * @note It may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
     * @note Available as float or double
     */
    template<typename T>
    __forceinline__ __device__ T uniform() const;
    /**
     * Returns a normally distributed float with mean 0.0 and standard deviation 1.0.
     * @note This result can be scaled and shifted to produce normally distributed values with any mean/stddev.
     * @note Available as float or double
     */
    template<typename T>
    __forceinline__ __device__ T normal() const;
    /**
     * Returns a log-normally distributed float based on a normal distribution with the given mean and standard deviation.
     * @note Available as float or double
     */
    template<typename T>
    __forceinline__ __device__ T logNormal(const T& mean, const T& stddev) const;
    /**
     * Returns an integer uniformly distributed in the inclusive range [min, max]
     * @note Available as signed and unsigned: char, short, int, long, long long
     */
    template<typename T>
    __forceinline__ __device__ T uniform(const T& min, const T& max) const;

private:
    __forceinline__ __device__ static unsigned int tid() {
        // 3D incase
        // Regardless, this should be optimised away
        auto blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        auto threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;
        return threadId;
    }
};

/**
 * Anonymous namespace to hide __device__ declarations
 */
namespace {
    extern __device__ curandState *d_random_state;
    extern __device__ AgentRandom::size_type d_random_size;
}

/**
 * Implmenetation
 */
__forceinline__ __device__ AgentRandom::AgentRandom() {
    // Check once per agent per kernel
    // as opposed to every time rng is called
    assert(tid() < d_random_size);
}
/**
 * All templates are specialised
 */

/**
 * Uniform floating point
 */
template<>
__forceinline__ __device__ float AgentRandom::uniform() const {
    return curand_uniform(&d_random_state[tid()]);
}
template<>
__forceinline__ __device__ double AgentRandom::uniform() const {
    return curand_uniform_double(&d_random_state[tid()]);
}

/**
 * Normal floating point
 */
template<>
__forceinline__ __device__ float AgentRandom::normal() const {
    return curand_normal(&d_random_state[tid()]);
}
template<>
__forceinline__ __device__ double AgentRandom::normal() const {
    return curand_normal_double(&d_random_state[tid()]);
}
/**
 * Log Normal floating point
 */
template<>
__forceinline__ __device__ float AgentRandom::logNormal(const float& mean, const float& stddev) const {
    return curand_log_normal(&d_random_state[tid()], mean, stddev);
}
template<>
__forceinline__ __device__ double AgentRandom::logNormal(const double& mean, const double& stddev) const {
    return curand_log_normal_double(&d_random_state[tid()], mean, stddev);
}
/**
* Uniform Int
*/
template<>
__forceinline__ __device__ char AgentRandom::uniform(const char& min, const char& max) const {
    return static_cast<char>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ unsigned char AgentRandom::uniform(const unsigned char& min, const unsigned char& max) const {
    return static_cast<unsigned char>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ short AgentRandom::uniform(const short& min, const short& max) const {
    return static_cast<short>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ unsigned short AgentRandom::uniform(const unsigned short& min, const unsigned short& max) const {
    return static_cast<unsigned short>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ int AgentRandom::uniform(const int& min, const int& max) const {
    return static_cast<int>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ unsigned int AgentRandom::uniform(const unsigned int& min, const unsigned int& max) const {
    return static_cast<unsigned int>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ long AgentRandom::uniform(const long& min, const long& max) const {
    //Platform specific, will be optimised away
    if (sizeof(long) == sizeof(float))
        return static_cast<long>(min + (max - min) * uniform<float>());
    return static_cast<long>(min + (max - min) * uniform<double>());
}
template<>
__forceinline__ __device__ unsigned long AgentRandom::uniform(const unsigned long& min, const unsigned long& max) const {
    //Platform specific, will be optimised away
    if (sizeof(unsigned long) == sizeof(float))
        return static_cast<unsigned long>(min + (max - min) * uniform<float>());
    return static_cast<unsigned long>(min + (max - min) * uniform<double>());
}
template<>
__forceinline__ __device__ long long AgentRandom::uniform(const long long& min, const long long& max) const {
    return static_cast<long long>(min + (max - min) * uniform<double>());
}
template<>
__forceinline__ __device__ unsigned long long AgentRandom::uniform(const unsigned long long& min, const unsigned long long& max) const {
    return static_cast<unsigned long long>(min + (max - min) * uniform<double>());
}
#endif //__AgentRandom_cuh__
#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_AGENTRANDOM_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_AGENTRANDOM_CUH_

#include <curand_kernel.h>
#include <cassert>

#include "flamegpu/exception/FGPUStaticAssert.h"

/**
 * Utility for accessing random generation within agent functions
 * This should only be instantiated by FLAMEGPU_API
 * Wraps curand functions to access an internal curand state * 
 */
class AgentRandom {
 public:
    typedef unsigned int size_type;
    /**
     * Constructs an AgentRandom instance
     * @param d_rng ThreadSafe device curand state instance
     *   this is a unique instance for the thread among all concurrently executing kernels
     */
    __forceinline__ __device__ AgentRandom(curandState *d_rng);
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
     * @note Available as signed and unsigned: char, short, int, long long
     */
    template<typename T>
    __forceinline__ __device__ T uniform(const T& min, const T& max) const;

 private:
    /**
     * Thread-safe index for accessing curand
     */
    curandState *d_random_state;
};

__forceinline__ __device__ AgentRandom::AgentRandom(curandState *d_rng) : d_random_state(d_rng) { }
/**
 * All templates are specialised
 */

/**
 * Uniform floating point
 */
template<>
__forceinline__ __device__ float AgentRandom::uniform() const {
    return curand_uniform(d_random_state);
}
template<>
__forceinline__ __device__ double AgentRandom::uniform() const {
    return curand_uniform_double(d_random_state);
}

/**
 * Normal floating point
 */
template<>
__forceinline__ __device__ float AgentRandom::normal() const {
    return curand_normal(d_random_state);
}
template<>
__forceinline__ __device__ double AgentRandom::normal() const {
    return curand_normal_double(d_random_state);
}
/**
 * Log Normal floating point
 */
template<>
__forceinline__ __device__ float AgentRandom::logNormal(const float& mean, const float& stddev) const {
    return curand_log_normal(d_random_state, mean, stddev);
}
template<>
__forceinline__ __device__ double AgentRandom::logNormal(const double& mean, const double& stddev) const {
    return curand_log_normal_double(d_random_state, mean, stddev);
}
/**
* Uniform Int
*/
template<typename T>
__forceinline__ __device__ T AgentRandom::uniform(const T& min, const T& max) const {
    static_assert(FGPU_SA::_Is_IntType<T>::value, "Invalid template argument for AgentRandom::uniform(const T& min, const T& max)");
    return static_cast<T>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ int64_t AgentRandom::uniform(const int64_t& min, const int64_t& max) const {
    return static_cast<int64_t>(min + (max - min) * uniform<double>());
}
template<>
__forceinline__ __device__ uint64_t AgentRandom::uniform(const uint64_t& min, const uint64_t& max) const {
    return static_cast<uint64_t>(min + (max - min) * uniform<double>());
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_AGENTRANDOM_CUH_

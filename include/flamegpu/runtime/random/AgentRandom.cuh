#ifndef INCLUDE_FLAMEGPU_RUNTIME_RANDOM_AGENTRANDOM_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_RANDOM_AGENTRANDOM_CUH_

#include <cassert>

#include "flamegpu/detail/curand.cuh"
#include "flamegpu/detail/StaticAssert.h"
#include "flamegpu/exception/FLAMEGPUDeviceException.cuh"

namespace flamegpu {

/**
 * Utility for accessing random generation within agent functions
 * This should only be instantiated by FLAMEGPU_API
 * Wraps curand functions to access an internal curand state * 
 */
class AgentRandom {
 public:
    /**
     * Constructs an AgentRandom instance
     * @param d_rng ThreadSafe device curand state instance
     *   this is a unique instance for the thread among all concurrently executing kernels
     */
    __forceinline__ __device__ AgentRandom(detail::curandState *d_rng);
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
    __forceinline__ __device__ T logNormal(T mean, T stddev) const;
    /**
     * Returns an integer uniformly distributed in the inclusive range [min, max]
     * or
     * Returns a floating point value uniformly distributed in the exclusive-inclusive range (min, max]
     * @tparam T return type
     * @note Available as signed and unsigned: char, short, int, long long, float, double
     */
    template<typename T>
    __forceinline__ __device__ T uniform(T min, T max) const;

 private:
    /**
     * Thread-safe index for accessing curand
     */
    detail::curandState *d_random_state;
};

__forceinline__ __device__ AgentRandom::AgentRandom(detail::curandState *d_rng) : d_random_state(d_rng) { }
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
__forceinline__ __device__ float AgentRandom::logNormal(const float mean, const float stddev) const {
    return curand_log_normal(d_random_state, mean, stddev);
}
template<>
__forceinline__ __device__ double AgentRandom::logNormal(const double mean, const double stddev) const {
    return curand_log_normal_double(d_random_state, mean, stddev);
}
/**
* Uniform Range
*/
template<typename T>
__forceinline__ __device__ T AgentRandom::uniform(T min, T max) const {
    static_assert(detail::StaticAssert::_Is_IntType<T>::value, "Invalid template argument for AgentRandom::uniform(T lowerBound, T max)");
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (min > max) {
        DTHROW("Invalid arguments passed to AgentRandom::uniform(), %lld > %lld\n", static_cast<int64_t>(min), static_cast<int64_t>(max));
    }
#endif
    return static_cast<T>(min + (max - min) * uniform<float>());
}
template<>
__forceinline__ __device__ int64_t AgentRandom::uniform(const int64_t min, const int64_t max) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (min > max) {
        DTHROW("Invalid arguments passed to AgentRandom::uniform(), %lld > %lld\n", static_cast<int64_t>(min), static_cast<int64_t>(max));
    }
#endif
    return static_cast<int64_t>(min + (max - min) * uniform<double>());
}
template<>
__forceinline__ __device__ uint64_t AgentRandom::uniform(const uint64_t min, const uint64_t max) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (min > max) {
        DTHROW("Invalid arguments passed to AgentRandom::uniform(), %lld > %lld\n", static_cast<int64_t>(min), static_cast<int64_t>(max));
    }
#endif
    return static_cast<uint64_t>(min + (max - min) * uniform<double>());
}
template<>
__forceinline__ __device__ float AgentRandom::uniform(const float min, const float max) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (min > max) {
        DTHROW("Invalid arguments passed to AgentRandom::uniform(), %f > %f\n", min, max);
    }
#endif
    return min + (max - min) * uniform<float>();
}
template<>
__forceinline__ __device__ double AgentRandom::uniform(const double min, const double max) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (min > max) {
        DTHROW("Invalid arguments passed to AgentRandom::uniform(), %f > %f\n", min, max);
    }
#endif
    return min + (max - min) * uniform<double>();
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_RANDOM_AGENTRANDOM_CUH_

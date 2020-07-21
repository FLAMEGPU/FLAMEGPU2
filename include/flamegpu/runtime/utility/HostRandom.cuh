#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_

#include <random>

#include "flamegpu/exception/FGPUStaticAssert.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"

/**
* Utility for accessing random generation within host functions
* This is prefered over using std random, as it uses a common seed with the device random
* This should only be instantiated by FLAMEGPU_HOST_API
*/
class HostRandom {
    friend class FLAMEGPU_HOST_API;
 public:
    /**
    * Returns a float uniformly distributed between 0.0 and 1.0.
    * @tparam T return type (must be floating point)
    * @note It may return from 0.0 to 1.0, where 0.0 is included and 1.0 is excluded.
    * @note Available as float or double
    */
    template<typename T>
    inline T uniform() const;
    /**
    * Returns a normally distributed float with mean 0.0 and standard deviation 1.0.
    * @tparam T return type (must be floating point)
    * @note This result can be scaled and shifted to produce normally distributed values with any mean/stddev.
    * @note Available as float or double
    */
    template<typename T>
    inline T normal() const;
    /**
    * Returns a log-normally distributed float based on a normal distribution with the given mean and standard deviation.
    * @tparam T return type (must be floating point)
    * @note Available as float or double
    */
    template<typename T>
    inline T logNormal(const T& mean, const T& stddev) const;
    /**
    * Returns an integer uniformly distributed in the inclusive range [lowerBound, max]
    * @tparam T return type (must be integer)
    * @note Available as signed and unsigned: char, short, int, long long
    */
    template<typename T>
    inline T uniform(const T& min, const T& max) const;

 private:
    explicit HostRandom(RandomManager &_rng) : rng(_rng) { }
    RandomManager &rng;
};



template<typename T>
inline T HostRandom::uniform() const {
    static_assert(FGPU_SA::_Is_RealType<T>::value, "Invalid template argument for HostRandom::uniform()");
    std::uniform_real_distribution<T> dist(0, 1);
    return rng.getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::normal() const {
    static_assert(FGPU_SA::_Is_RealType<T>::value, "Invalid template argument for HostRandom::normal()");
    std::normal_distribution<T> dist(0, 1);
    return rng.getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::logNormal(const T& mean, const T& stddev) const {
    static_assert(FGPU_SA::_Is_RealType<T>::value, "Invalid template argument for HostRandom::logNormal(const T& mean, const T& stddev)");
    std::lognormal_distribution<T> dist(mean, stddev);
    return rng.getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::uniform(const T& min, const T& max) const {
    static_assert(FGPU_SA::_Is_IntType<T>::value, "Invalid template argument for HostRandom::uniform(const T& lowerBound, const T& max)");
    std::uniform_int_distribution<T> dist(min, max);
    return rng.getDistribution<T>(dist);
}

/**
 * Special cases, std::random doesn't support char, emulate behaviour
 */
template<>
inline char HostRandom::uniform(const char& min, const char& max) const {
    std::uniform_int_distribution<int16_t> dist(min, max);
    return static_cast<char>(rng.getDistribution<int16_t>(dist));
}

template<>
inline unsigned char HostRandom::uniform(const unsigned char& min, const unsigned char& max) const {
    std::uniform_int_distribution<uint16_t> dist(min, max);
    return static_cast<unsigned char>(rng.getDistribution<uint16_t>(dist));
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_

#include <Random>

#include "flamegpu/runtime/utility/RandomManager.cuh"

/**
* Utility for accessing random generation within host functions
* This is prefered over using std random, as it uses a common seed with the device random
* This should only be instantiated by FLAMEGPU_HOST_API
*/
class HostRandom {
 public:
    /**
    * Returns a float uniformly distributed between 0.0 and 1.0.
    * @note It may return from 0.0 to 1.0, where 0.0 is included and 1.0 is excluded.
    * @note Available as float or double
    */
    template<typename T>
    inline T uniform() const;
    /**
    * Returns a normally distributed float with mean 0.0 and standard deviation 1.0.
    * @note This result can be scaled and shifted to produce normally distributed values with any mean/stddev.
    * @note Available as float or double
    */
    template<typename T>
    inline T normal() const;
    /**
    * Returns a log-normally distributed float based on a normal distribution with the given mean and standard deviation.
    * @note Available as float or double
    */
    template<typename T>
    inline T logNormal(const T& mean, const T& stddev) const;
    /**
    * Returns an integer uniformly distributed in the inclusive range [min, max]
    * @note Available as signed and unsigned: char, short, int, long long
    */
    template<typename T>
    inline T uniform(const T& min, const T& max) const;
};

template<typename T>
inline T HostRandom::uniform() const {
    std::uniform_real_distribution<T> dist(0, 1);
    return RandomManager::getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::normal() const {
    std::normal_distribution<T> dist(0, 1);
    return RandomManager::getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::logNormal(const T& mean, const T& stddev) const {
    std::lognormal_distribution<T> dist(mean, stddev);
    return RandomManager::getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::uniform(const T& min, const T& max) const {
    std::uniform_int_distribution<T> dist(min, max);
    return RandomManager::getDistribution<T>(dist);
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_

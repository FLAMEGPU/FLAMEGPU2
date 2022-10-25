#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_

#include <random>

#include "flamegpu/util/detail/StaticAssert.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"

namespace flamegpu {

/**
* Utility for accessing random generation within host functions
* This is prefered over using std random, as it uses a common seed with the device random
* This should only be instantiated by HostAPI
*/
class HostRandom {
    friend class HostAPI;
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
     * Returns an integer uniformly distributed in the inclusive range [min, max]
     * or
     * Returns a floating point value uniformly distributed in the inclusive-exclusive range [min, max)
     * @tparam T return type
     * @note Available as signed and unsigned: char, short, int, long long
     */
    template<typename T>
    inline T uniform(const T& min, const T& max) const;
    /**
     * Change the seed used for random generation
     * @param seed New random seed
     */
    void setSeed(const uint64_t &seed);
    /**
     * Returns the last value used to seed random generation
     */
    uint64_t getSeed() const;

 private:
    explicit HostRandom(RandomManager &_rng) : rng(_rng) { }
    RandomManager &rng;
};



template<typename T>
inline T HostRandom::uniform() const {
    static_assert(util::detail::StaticAssert::_Is_RealType<T>::value, "Invalid template argument for HostRandom::uniform()");
    std::uniform_real_distribution<T> dist(0, 1);
    return rng.getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::normal() const {
    static_assert(util::detail::StaticAssert::_Is_RealType<T>::value, "Invalid template argument for HostRandom::normal()");
    std::normal_distribution<T> dist(0, 1);
    return rng.getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::logNormal(const T& mean, const T& stddev) const {
    static_assert(util::detail::StaticAssert::_Is_RealType<T>::value, "Invalid template argument for HostRandom::logNormal(const T& mean, const T& stddev)");
    std::lognormal_distribution<T> dist(mean, stddev);
    return rng.getDistribution<T>(dist);
}

template<typename T>
inline T HostRandom::uniform(const T& min, const T& max) const {
    static_assert(util::detail::StaticAssert::_Is_IntType<T>::value, "Invalid template argument for HostRandom::uniform(const T& lowerBound, const T& max)");
    std::uniform_int_distribution<T> dist(min, max);
    return rng.getDistribution<T>(dist);
}

/**
 * Special cases, std::random doesn't support char, emulate behaviour
 */
template<>
inline char HostRandom::uniform(const char min, const char max) const {
    std::uniform_int_distribution<int16_t> dist(min, max);
    return static_cast<char>(rng.getDistribution<int16_t>(dist));
}

template<>
inline unsigned char HostRandom::uniform(const unsigned char min, const unsigned char max) const {
    std::uniform_int_distribution<uint16_t> dist(min, max);
    return static_cast<unsigned char>(rng.getDistribution<uint16_t>(dist));
}

template<>
inline signed char HostRandom::uniform(const signed char& min, const signed char& max) const {
    std::uniform_int_distribution<int16_t> dist(min, max);
    return static_cast<signed char>(rng.getDistribution<int16_t>(dist));
}
template<>
inline float HostRandom::uniform(const float& min, const float& max) const {
    std::uniform_real_distribution<float> dist(min, max);
    return rng.getDistribution<float>(dist);
}
template<>
inline double HostRandom::uniform(const double& min, const double& max) const {
    std::uniform_real_distribution<double> dist(min, max);
    return rng.getDistribution<double>(dist);
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTRANDOM_CUH_

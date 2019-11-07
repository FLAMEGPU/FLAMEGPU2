#ifndef SRC_FLAMEGPU_RUNTIME_UTILITY_RANDOM_CUH_
#define SRC_FLAMEGPU_RUNTIME_UTILITY_RANDOM_CUH_

#include <cstdint>

#include "./curand_kernel.h"
#include "flamegpu/runtime/utility/AgentRandom.cuh"

/**
 * Static manager for the shared array of curand state used by a simulation
 * Pairs with device size AgentRandom
 */
class DeviceRandomArray {
 public:
    /**
     * Inherit size_type from include-public partner class
     */
    typedef AgentRandom::size_type size_type;
    /**
     * Utility for passing to init()
     */
    static uint64_t seedFromTime();
    /**
     * Acts as constructor
     * @note Can be called multiple times to reseed, doing so releases existing memory allocations
     */
    static void init(const uint64_t &seed);
    /**
     * Acts as destructor
     * @note Safe to call multiple times
     */
    static void free();
    /**
     * Resizes random array according to the rules:
     *   while(length<_length)
     *     length*=growthModifier
     *   if(shrinkModifier<1.0)
     *     while(length*shrinkModifier>_length)
     *       length*=shrinkModifier
     */
    static bool resize(const size_type &_length);
    /**
     * Accessors
     */
    static void setGrowthModifier(float);
    static float getGrowthModifier();
    static void setShrinkModifier(float);
    static float getShrinkModifier();
    /**
     * Returns length of curand state array currently allocated
     */
    static size_type size();
    static uint64_t seed();

 private:
    static uint64_t mSeed;
    static size_type length;
    static size_type min_length;
    static float shrinkModifier;
    static float growthModifier;
    static void resizeDeviceArray(const size_type &_length);
    /**
     * @note h_max_random_state will be allocated to length h_max_random_size
     * However, it will only be initialised from hd_random_size(aka length) onwards
     */
    static curandState *h_max_random_state;
    static DeviceRandomArray::size_type h_max_random_size;
};

#endif  // SRC_FLAMEGPU_RUNTIME_UTILITY_RANDOM_CUH_

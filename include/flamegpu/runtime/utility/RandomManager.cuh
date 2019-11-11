#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_RANDOMMANAGER_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_RANDOMMANAGER_CUH_

#include <curand_kernel.h>
#include <cstdint>
#include <random>

#include "flamegpu/runtime/utility/AgentRandom.cuh"

/**
 * Static manager for initialising simulation wide random with a common seed
 * Manages the shared array of curand state use by agent functions
 * Manages the random engine/s used by host functions
 * Pairs with HostRandom and device side AgentRandom
 */
class RandomManager {
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
    static void init(const unsigned int &seed);
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
    template<typename T, typename dist>
    /**
     * Generates a random number with the provided distribution
     * Note: Not believed to be thread-safe!
     */
    static T getDistribution(dist &distribution);
    /**
     * Returns length of curand state array currently allocated
     */
    static size_type size();
    static uint64_t seed();

 protected:
    /**
     * Protected constructor prevents the class being instantiated
     * Device array of currand must be declared as __device__
     * Therefore it cannot be instanced, so neither should it's manager.
     */
    RandomManager() {}

 private:
    /**
     * Random seed used to initialise all currently allocated curand states
     */
    static unsigned int mSeed;
    /**
     * Local copy of the length of d_random_state
     */
    static size_type length;
    /**
     * Minimum allocation length of d_random_state
     */
    static const size_type min_length;
    /**
     * Rate at which d_random_state's length shrinks
     * Must be in the range 0 < x <= 1.0
     * A value of 1.0 means the curand array will never decrease in size 
     */
    static float shrinkModifier;
    /**
     * Rate at which d_random_state's length grows
     * Must be in greater than 1.0
     */
    static float growthModifier;
    /**
     * Actually performs the resizing of the curand state array
     * If shrinking, 'deallocated' curand states are backed up to host until next required,
     *  this prevents them being reinitialised with the same seed.
     */
    static void resizeDeviceArray(const size_type &_length);
    /**
     * Host copy of 'deallocated' curand states
     * When the device array shrinks in size, shrunk away curand states are stored here
     * @note h_max_random_state will be allocated to length h_max_random_size
     * However, it will only be initialised from hd_random_size(aka length) onwards
     */
    static curandState *h_max_random_state;
    /**
     * Allocated length of h_max_random_state
     */
    static RandomManager::size_type h_max_random_size;
    /**
     * Seeded host random generator
     * Don't believe this to be thread-safe!
     */
    static std::default_random_engine host_rng;
};


template<typename T, typename dist>
T RandomManager::getDistribution(dist &distribution) {
    return distribution(host_rng);
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_RANDOMMANAGER_CUH_

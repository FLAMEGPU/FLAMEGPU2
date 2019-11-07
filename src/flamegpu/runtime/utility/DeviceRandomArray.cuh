#ifndef SRC_FLAMEGPU_RUNTIME_UTILITY_RANDOM_CUH_
#define SRC_FLAMEGPU_RUNTIME_UTILITY_RANDOM_CUH_

#include <curand_kernel.h>
#include <cstdint>

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
protected:
    /**
     * Protected constructor prevents the class being instantiated
     * Device array of currand must be declared as __device__
     * Therefore it cannot be instanced, so neither should it's manager.
     */
    DeviceRandomArray() {}
 private:
    /**
     * Random seed used to initialise all currently allocated curand states
     */
    static uint64_t mSeed;
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
    static DeviceRandomArray::size_type h_max_random_size;
};

#endif  // SRC_FLAMEGPU_RUNTIME_UTILITY_RANDOM_CUH_

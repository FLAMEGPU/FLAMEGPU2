#ifndef __Random_cuh__
#define __Random_cuh__
#include <vector>
#include <curand_kernel.h>
#include "flamegpu/runtime/utility/AgentRandom.cuh"

/**
 * Static manager for the shared array of curand state used by a simulation
 * Pairs with device size AgentRandom
 */
class Random {
 public:
    typedef AgentRandom::size_type size_type;
    /**
     * Utility for passing to init()
     */
    static unsigned long long seedFromTime();
    static void init(const unsigned long long &seed);
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
    static void setGrowthModifier(float);
    static float getGrowthModifier();
    static void setShrinkModifier(float);
    static float getShrinkModifier();
    /**
     * Returns length of curand state array currently allocated
     */
    static size_type size();
 private:
    static unsigned long long seed;
    static size_type length;
    static float shrinkModifier;
    static float growthModifier;
    static void resizeDeviceArray(const size_type &_length);
};

#endif //__Random_cuh__
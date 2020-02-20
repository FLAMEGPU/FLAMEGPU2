#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_RANDOMMANAGER_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_RANDOMMANAGER_CUH_

#include <curand_kernel.h>
#include <cstdint>
#include <random>
#include <string>

#include "flamegpu/runtime/utility/AgentRandom.cuh"

#include "flamegpu/sim/Simulation.h"

/**
 * Singleton manager for initialising simulation wide random with a common seed
 * This is an internal class, that should not be accessed directly by modellers
 * Manages the shared array of curand state use by agent functions
 * Manages the random engine/s used by host functions
 * @see AgentRandom For random number generation during agent functions on the device
 * @see HostRandom For random number generation during host functions
 */
class RandomManager {
    /**
     * Requires instance for host function random generation
     */
    friend class HostRandom;
    /**
     * Handles seeding of random generation
     */
    friend void Simulation::applyConfig();
    /**
     * Calls resize() during simulation execution to resize device random array
     */
    friend class CUDAAgentModel;  // bool CUDAAgentModel::step(const Simulation&)
 public:
    /**
     * Inherit size_type from include-public partner class
     */
    typedef AgentRandom::size_type size_type;
    /**
     * Utility for generating a psuesdo-random seed to pass to init
     */
    uint64_t seedFromTime();
    /**
     * Reseeds all owned random generators
     * @note Can be called multiple times to reseed, doing so releases existing memory allocations
     */
    void reseed(const unsigned int &seed);
    /**
     * Resizes random array according to the rules:
     *   while(length<_length)
     *     length*=growthModifier
     *   if(shrinkModifier<1.0)
     *     while(length*shrinkModifier>_length)
     *       length*=shrinkModifier
     */
    bool resize(const size_type &_length);
    /**
     * Accessors
     */
    void setGrowthModifier(float);
    float getGrowthModifier();
    void setShrinkModifier(float);
    float getShrinkModifier();
    /**
     * Generates a random number with the provided distribution
     * @param distribution A distribution object defined by <random>
     * @tparam dist random distribution type to be used for generation (this should be implicitly detected)
     * @tparam T return type
     * @note Not believed to be thread-safe!
     */
    template<typename T, typename dist>
    T getDistribution(dist &distribution);
    /**
     * Returns length of curand state array currently allocated
     */
    size_type size();
    uint64_t seed();

 private:
    /**
     * Random seed used to initialise all currently allocated curand states
     */
    unsigned int mSeed = 0;
    /**
     * Local copy of the length of d_random_state
     */
    size_type length = 0;
    /**
     * Minimum allocation length of d_random_state
     */
    const size_type min_length = 1024;
    /**
     * Rate at which d_random_state's length shrinks
     * Must be in the range 0 < x <= 1.0
     * A value of 1.0 means the curand array will never decrease in size 
     */
    float shrinkModifier = 1.0f;
    /**
     * Rate at which d_random_state's length grows
     * Must be in greater than 1.0
     */
    float growthModifier = 1.5f;
    /**
     * Actually performs the resizing of the curand state array
     * If shrinking, 'deallocated' curand states are backed up to host until next required,
     *  this prevents them being reinitialised with the same seed.
     */
    void resizeDeviceArray(const size_type &_length);
    /**
     * Host copy of 'deallocated' curand states
     * When the device array shrinks in size, shrunk away curand states are stored here
     * @note h_max_random_state will be allocated to length h_max_random_size
     * However, it will only be initialised from hd_random_size(aka length) onwards
     */
    curandState *h_max_random_state = nullptr;
    /**
     * Allocated length of h_max_random_state
     */
    size_type h_max_random_size = 0;
    /**
     * Seeded host random generator
     * Don't believe this to be thread-safe!
     * @note - std::default_random_engine is platform (compiler) specific. GCC (7.4) defaults to a linear_congruential_engine, which returns the same sequence for seeds 0 and 1. mt19937 is the default in MSVC and generally seems more sane.
     */
    std::mt19937 host_rng;

    /**
     * Flag indicating that the device memory has been initialised, and therefore might need resetting
     */
    bool deviceInitialised;
    /**
     * Acts as destructor
     * @note Safe to call multiple times
     */
    void free();

    /**
     * Destroys host stuff 
     */
    void freeHost();
    /**
     * Destroys device memory / resets the curand states
     * @note includes cuda commands so not safe from c++ destructor
     */
    void freeDevice();

    /**
     * Reinitialises host RNG from the current seed.
     */
    void reseedHost();

    /**
     * Reinitialises device RNG from the current seed.
     * @note includes cuda commands.
     */
    void reseedDevice();
    /**
     * Remainder of class is singleton pattern
     */
    /**
     * Creates the singleton and calls reseed() with the return value from seedFromTime()
     */
    RandomManager();
    /**
     * Logs how many CUDAAgentModel objects exist, if this reaches 0, free is called
     */
    static unsigned int simulationInstances;
    /**
     * Increases internal counter of CUDAAgentModel instances
     */
    void increaseSimCounter();
    /**
     * Decreases internal counter of CUDAAgentModel instances
     * If this reaches 0, free() is called
     */
    void decreaseSimCounter();

 protected:
     /**
      * Returns the RandomManager singleton instance
      */
     static RandomManager& getInstance() {
         static RandomManager instance;  // Guaranteed to be destroyed.
         return instance;                // Instantiated on first use.
     }
     ~RandomManager();

 public:
    // Public deleted creates better compiler errors
    RandomManager(RandomManager const&) = delete;
    void operator=(RandomManager const&) = delete;
};


template<typename T, typename dist>
T RandomManager::getDistribution(dist &distribution) {
    return distribution(host_rng);
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_RANDOMMANAGER_CUH_

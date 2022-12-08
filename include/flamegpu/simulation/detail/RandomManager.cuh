#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_RANDOMMANAGER_CUH_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_RANDOMMANAGER_CUH_

#include <cstdint>
#include <random>
#include <string>

#include "flamegpu/defines.h"
#include "flamegpu/detail/curand.cuh"
#include "flamegpu/simulation/Simulation.h"

namespace flamegpu {
// forward declare classes
class CUDASimulation;
namespace detail {


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
    friend class CUDASimulation;  // bool CUDASimulation::step(const Simulation&)
 public:
    /**
     * Creates the random manager and calls reseed() with the return value from seedFromTime()
     */
    RandomManager();

     ~RandomManager();
    /**
     * Utility for generating a psuesdo-random seed to pass to init
     */
    uint64_t seedFromTime();
    /**
     * Reseeds all owned random generators
     * @note Can be called multiple times to reseed, doing so releases existing memory allocations
     */
    void reseed(uint64_t seed);
    /**
     * Resizes random array according to the rules:
     *   while(length<_length)
     *     length*=growthModifier
     *   if(shrinkModifier<1.0)
     *     while(length*shrinkModifier>_length)
     *       length*=shrinkModifier
     */
    detail::curandState*resize(size_type _length, cudaStream_t stream);
    /**
     * Accessors
     */
    void setGrowthModifier(float);
    float getGrowthModifier();
    void setShrinkModifier(float);
    float getShrinkModifier();
    /**
     * Generates a random number with the provided distribution
     * @param distribution A distribution object defined by \<random\>
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
    detail::curandState*cudaRandomState();

 private:
    /**
     * Device array holding curand states
     * They should always be initialised
     */
     detail::curandState*d_random_state = nullptr;
    /**
     * Random seed used to initialise all currently allocated curand states
     */
    uint64_t mSeed = 0;
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
    void resizeDeviceArray(size_type _length, cudaStream_t stream);
    /**
     * Host copy of 'deallocated' curand states
     * When the device array shrinks in size, shrunk away curand states are stored here
     * @note h_max_random_state will be allocated to length h_max_random_size
     * However, it will only be initialised from hd_random_size(aka length) onwards
     */
    detail::curandState *h_max_random_state = nullptr;
    /**
     * Allocated length of h_max_random_state
     */
    size_type h_max_random_size = 0;
    /**
     * Seeded host random generator
     * Don't believe this to be thread-safe!
     * @note - std::default_random_engine is platform (compiler) specific. GCC (7.4) defaults to a linear_congruential_engine, which returns the same sequence for seeds 0 and 1. mt19937 is the default in MSVC and generally seems more sane (but using the 64 bit variant).
     */
    std::mt19937_64 host_rng;

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

 public:
    // Public deleted creates better compiler errors
    RandomManager(RandomManager const&) = delete;
    void operator=(RandomManager const&) = delete;
};


template<typename T, typename dist>
T RandomManager::getDistribution(dist &distribution) {
    return distribution(host_rng);
}

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_RANDOMMANAGER_CUH_

#ifndef INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_

#include <map>
#include <string>

#include "flamegpu/model/ModelData.h"
/**
 * Singleton class for performing generic scatters
 * This is used for optional messages, agent death, agent birth
 */
class CUDAScatter {
    /**
     * Needs access for the template instantiation
     */
    friend class std::array<CUDAScatter, flamegpu_internal::CUDAScanCompaction::MAX_STREAMS>;
    /**
     * Has access for calling increaseSimCounter() decreaseSimCounter()
     */
    friend class CUDAAgentModel;

 public:
    enum Type {Agent, Message};
    struct ScatterData {
        size_t typeLen;
        char *const in;
        char *out;
    };
    unsigned int scatter(
        Type messageOrAgent,
        const ModelData::VariableMap &vars,
        const std::map<std::string, void*> &in,
        const std::map<std::string, void*> &out,
        const unsigned int &itemCount);

 private:
    unsigned int streamId;
    ScatterData *d_data;
    unsigned int data_len;
    void resize(const unsigned int &newLen);
    /**
     * Remainder of class is singleton pattern
     */
    /**
     * Creates the singleton and calls reseed() with the return value from seedFromTime()
     */
     CUDAScatter();
    /**
     * Logs how many CUDAAgentModel objects exist, if this reaches 0, free is called
     */
    static unsigned int simulationInstances;
    /**
     * Releases all CUDA allocations, called by decreaseSimCounter()
     */
    void free();
    /**
     * Increases internal counter of CUDAAgentModel instances
     */
    static void increaseSimCounter();
    /**
     * Decreases internal counter of CUDAAgentModel instances
     * If this reaches 0, free() is called on all instances
     */
    void decreaseSimCounter();

 protected:
    /**
     * Frees cuda allocations
     */
    ~CUDAScatter();

 public:
    /**
    * Returns the RandomManager singleton instance
    */
    static CUDAScatter& getInstance(unsigned int streamId) {
        // Guaranteed to be destroyed.
        static std::array<CUDAScatter, flamegpu_internal::CUDAScanCompaction::MAX_STREAMS> instance;
        // Basic err check
        assert(streamId < flamegpu_internal::CUDAScanCompaction::MAX_STREAMS);
        instance[streamId].streamId = streamId;
        // Instantiated on first use.
        return instance[streamId];
    }
    // Public deleted creates better compiler errors
    CUDAScatter(CUDAScatter const&) = delete;
    void operator=(CUDAScatter const&) = delete;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_

#ifndef INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_

#include <map>
#include <string>
#include <array>

#include "flamegpu/model/Variable.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
struct VarOffsetStruct;

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
    /**
     * This utility class provides a wrapper for `unsigned int *`
     * When the iterator is dereferenced the pointed to unsigned int is evaluated
     * using invert()
     * This is useful when trying to partition and sort a dataset using only scatter and scan
     */
     struct InversionIterator {
         using difference_type = unsigned int;
         using value_type = unsigned int;
         using pointer = unsigned int*;
         using reference = unsigned int&;
         using iterator_category = std::random_access_iterator_tag;
         __host__ __device__ explicit InversionIterator(unsigned int *_p) : p(_p) { }

         __device__ InversionIterator &operator=(const InversionIterator&other) { p = other.p; return *this; }
         __device__ InversionIterator operator++ (int a) { p += a;  return *this; }
         __device__ InversionIterator operator++ () { p++;  return *this; }
         __device__ unsigned int operator *() { return invert(*p); }
         __device__ InversionIterator operator+(const int &b) const { return InversionIterator(p + b); }
         __device__ unsigned int operator[](int b) const { return  invert(p[b]); }
      private:
         __device__ unsigned int invert(unsigned int c) const { return c == 0 ? 1 : 0; }
         unsigned int *p;
     };
    /**
     * Flag used to decide which scan_flag array should be used
     * @see flamegpu_internal::CUDAScanCompaction::type
     */
    enum Type { Message, AgentDeath, AgentBirth};
    /**
     * As we scatter per variable, this structure holds all the data required for a single variable
     */
    struct ScatterData {
        size_t typeLen;
        char *const in;
        char *out;
    };
    /**
     * Scatters agents from SoA to SoA according to d_position flag
     * Used for device agent creation and agent death
     * flamegpu_internal::CUDAScanCompaction::scan_flag is used to decide who should be scattered
     * flamegpu_internal::CUDAScanCompaction::position is used to decide where to scatter to
     * @param messageOrAgent Flag of whether message or agent flamegpu_internal::CUDAScanCompaction arrays should be used
     * @param vars Variable description map from ModelData hierarchy
     * @param in Input variable name:ptr map
     * @paramn out Output variable name:ptr map
     * @param itemCount Total number of items in input array to consider
     * @param out_index_offset The offset to be applied to the ouput index (e.g. if out already contains data)
     * @parma invert_scan_flag If true, agents with scan_flag set to 0 will be moved instead
     */
    unsigned int scatter(
        Type messageOrAgent,
        const VariableMap &vars,
        const std::map<std::string, void*> &in,
        const std::map<std::string, void*> &out,
        const unsigned int &itemCount,
        const unsigned int &out_index_offset = 0,
        const bool &invert_scan_flag = false);
    /**
     * Scatters a contigous block from SoA to SoA
     * flamegpu_internal::CUDAScanCompaction::scan_flag/position are not used
     * @param vars Variable description map from ModelData hierarchy
     * @param in Input variable name:ptr map
     * @paramn out Output variable name:ptr map
     * @param itemCount Total number of items in input array to consider
     * @param out_index_offset The offset to be applied to the ouput index (e.g. if out already contains data)
     */
    unsigned int scatterAll(
        const VariableMap &vars,
        const std::map<std::string, void*> &in,
        const std::map<std::string, void*> &out,
        const unsigned int &itemCount,
        const unsigned int &out_index_offset = 0);
    /**
     * Used for reordering messages from SoA to SoA
     * Position information is taken using PBM data, rather than d_position
     * Used by spatial messaging.
     * @param vars Variable description map from ModelData hierarchy
     * @param in Input variable name:ptr map
     * @paramn out Output variable name:ptr map
     * @param itemCount Total number of items in input array to consider
     * @param d_bin_index This idenitifies which bin each index should be sorted to
     * @param d_bin_sub_index This indentifies where within it's bin, an index should be sorted to
     * @param d_pbm This is the PBM, it identifies at which index a bin's storage begins
     */
    void pbm_reorder(
        const VariableMap &vars,
        const std::map<std::string, void*> &in,
        const std::map<std::string, void*> &out,
        const unsigned int &itemCount,
        const unsigned int *d_bin_index,
        const unsigned int *d_bin_sub_index,
        const unsigned int *d_pbm);
    /**
     * Scatters agents from AoS to SoA
     * Used by host agent creation
     */
    void scatterNewAgents(
        const VariableMap &vars,
        const std::map<std::string, void*> &out,
        void *in_buff,
        const VarOffsetStruct &inOffsetData,
        const unsigned int &inCount,
        const unsigned int outIndexOffset);
    /**
     * Broadcasts a single value  for each variable to a contiguous block in SoA
     * Used priot to device agent creation
     */
    void broadcastInit(
        const VariableMap &vars,
        const std::map<std::string, void*> &out,
        const unsigned int &inCount,
        const unsigned int outIndexOffset);

 private:
    // set by getInstance()
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

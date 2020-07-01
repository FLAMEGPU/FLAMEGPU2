#ifndef INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_

#include <map>
#include <string>
#include <array>
#include <memory>
#include <list>
#include <vector>

#include "flamegpu/model/Variable.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
struct VarOffsetStruct;
struct VariableBuffer;

/**
 * Singleton class for performing generic scatters
 * This is used for optional messages, agent death, agent birth
 */
class CUDAScatter {
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
     * @see CUDAScanCompaction::type
     */
    typedef CUDAScanCompaction::Type Type;
    /**
     * As we scatter per variable, this structure holds all the data required for a single variable
     */
    struct ScatterData {
        size_t typeLen;
        char *const in;
        char *out;
    };

 private:
    /**
     * Needs access for the template instantiation
     */
    struct StreamData {
        friend class std::array<StreamData, CUDAScanCompaction::MAX_STREAMS>;
        ScatterData *d_data;
        unsigned int data_len;
        StreamData();
        ~StreamData();
        void purge();
        void resize(const unsigned int &newLen);
    };
    std::array<StreamData, CUDAScanCompaction::MAX_STREAMS> streams;

 public:
    CUDAScanCompaction &Scan() { return scan; }
    /**
     * Convenience wrapper for scatter()
     * Scatters agents from SoA to SoA according to d_position flag
     * Used for device agent creation and agent death
     * CUDAScanCompaction::scan_flag is used to decide who should be scattered
     * CUDAScanCompaction::position is used to decide where to scatter to
     * @param streamId Index of internal resources to use
     * @param messageOrAgent Flag of whether message or agent CUDAScanCompaction arrays should be used
     * @param vars Variable description map from ModelData hierarchy
     * @param in Input variable name:ptr map
     * @paramn out Output variable name:ptr map
     * @param itemCount Total number of items in input array to consider
     * @param out_index_offset The offset to be applied to the ouput index (e.g. if out already contains data)
     * @parma invert_scan_flag If true, agents with scan_flag set to 0 will be moved instead
     * @parma scatter_all_count The number of agents at the start of in to be copied, ones after this use scanflag
     * @note This is deprecated, unclear if still used
     */
     unsigned int scatter(
        const unsigned int &streamId,
        const Type &messageOrAgent,
        const VariableMap &vars,
        const std::map<std::string, void*> &in,
        const std::map<std::string, void*> &out,
        const unsigned int &itemCount,
        const unsigned int &out_index_offset = 0,
        const bool &invert_scan_flag = false,
        const unsigned int &scatter_all_count = 0);
    /**
     * Scatters agents from SoA to SoA according to d_position flag
     * Used for device agent creation and agent death
     * CUDAScanCompaction::scan_flag is used to decide who should be scattered
     * CUDAScanCompaction::position is used to decide where to scatter to
     * @param streamId Index of internal resources to use
     * @param messageOrAgent Flag of whether message or agent CUDAScanCompaction arrays should be used
     * @param scatterData Vector of scatter configuration for each variable to be scattered
     * @param itemCount Total number of items in input array to consider
     * @param out_index_offset The offset to be applied to the ouput index (e.g. if out already contains data)
     * @parma invert_scan_flag If true, agents with scan_flag set to 0 will be moved instead
     * @parma scatter_all_count The number of agents at the start of in to be copied, ones after this use scanflag
     */
    unsigned int scatter(
        const unsigned int &streamId,
        const Type &messageOrAgent,
        const std::vector<ScatterData> &scatterData,
        const unsigned int &itemCount,
        const unsigned int &out_index_offset = 0,
        const bool &invert_scan_flag = false,
        const unsigned int &scatter_all_count = 0);
    /**
     * Scatters agents from SoA to SoA according to d_position flag as input_source, all variables are scattered
     * Used for Host function sort agent
     * CUDAScanCompaction::position is used to decide where to scatter to
     * @param streamId Index of internal resources to use
     * @param messageOrAgent Flag of whether message or agent CUDAScanCompaction arrays should be used
     * @param scatterData Vector of scatter configuration for each variable to be scattered
     * @param itemCount Total number of items in input array to consider
     */
    void scatterPosition(
        const unsigned int &streamId,
        const Type &messageOrAgent,
        const std::vector<ScatterData> &scatterData,
        const unsigned int &itemCount);
    /**
     * Returns the final CUDAScanCompaction::position item 
     * Same value as scatter, - scatter_a__count
     * @param streamId Index of internal resources to use
     * @param messageOrAgent Flag of whether message or agent CUDAScanCompaction arrays should be used
     * @param itemCount Total number of items in input array to consider
     * @parma scatter_all_count The number offset into the array where the scan began
     */
    unsigned int scatterCount(
        const unsigned int &streamId,
        const Type &messageOrAgent,
        const unsigned int &itemCount,
        const unsigned int &scatter_all_count = 0);
    /**
     * Scatters a contigous block from SoA to SoA
     * CUDAScanCompaction::scan_flag/position are not used
     * @param streamId Index of internal resources to use
     * @param scatterData Vector of scatter configuration for each variable to be scattered
     * @param itemCount Total number of items in input array to consider
     * @param out_index_offset The offset to be applied to the ouput index (e.g. if out already contains data)
     * @note If calling scatter() with itemCount == scatter_all_count works the same
     */
    unsigned int scatterAll(
        const unsigned int &streamId,
        const std::vector<ScatterData> &scatterData,
        const unsigned int &itemCount,
        const unsigned int &out_index_offset = 0);
    /**
     * Convenience wrapper to scatterAll()
     * @param streamId Index of internal resources to use
     * @param vars Variable description map from ModelData hierarchy
     * @param in Input variable name:ptr map
     * @paramn out Output variable name:ptr map
     */
    unsigned int scatterAll(
        const unsigned int &streamId,
        const VariableMap &vars,
        const std::map<std::string, void*> &in,
        const std::map<std::string, void*> &out,
        const unsigned int &itemCount,
        const unsigned int &out_index_offset);
    /**
     * Used for reordering messages from SoA to SoA
     * Position information is taken using PBM data, rather than d_position
     * Used by spatial messaging.
     * @param streamId Index of internal resources to use
     * @param vars Variable description map from ModelData hierarchy
     * @param in Input variable name:ptr map
     * @paramn out Output variable name:ptr map
     * @param itemCount Total number of items in input array to consider
     * @param d_bin_index This idenitifies which bin each index should be sorted to
     * @param d_bin_sub_index This indentifies where within it's bin, an index should be sorted to
     * @param d_pbm This is the PBM, it identifies at which index a bin's storage begins
     */
    void pbm_reorder(
        const unsigned int &streamId,
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
     * @param streamId Index of internal resources to use
     * @param scatterData Vector of scatter configuration for each variable to be scattered
     * @param totalAgentSize Total size of all of the variables in an agent
     * @param inCount Total number of items in input array to consider
     * @param out_index_offset The offset to be applied to the ouput index (e.g. if out already contains data)
     */
    void scatterNewAgents(
        const unsigned int &streamId,
        const std::vector<ScatterData> &scatterData,
        const size_t &totalAgentSize,
        const unsigned int &inCount,
        const unsigned int &out_index_offset);
    /**
     * Broadcasts a single value for each variable to a contiguous block in SoA
     * Used prior to device agent creation
     * @param streamId Index of internal resources to use
     * @param vars Variable description map from ModelData hierarchy
     * @param itemCount Total number of items in input array to consider
     * @param out_index_offset The offset to be applied to the ouput index (e.g. if out already contains data)
     */
    void broadcastInit(
        const unsigned int &streamId,
        const std::list<std::shared_ptr<VariableBuffer>> &vars,
        const unsigned int &itemCount,
        const unsigned int out_index_offset);
    void broadcastInit(
        const unsigned int &streamId,
        const VariableMap &vars,
        void * const d_newBuff,
        const unsigned int &itemCount,
        const unsigned int out_index_offset);
    /**
     * Used to reorder array messages based on __INDEX variable, that variable is not sorted
     * Also throws exception if any indexes are repeated
     * @param streamId Index of internal resources to use
     * @param array_length Length of the array messages are to be stored in (max index + 1)
     * @param d_write_flag Device pointer to array for tracking how many messages output to each bin, caller responsibiltiy to ensure it is array_length or longer
     */
    void arrayMessageReorder(
        const unsigned int &streamId,
        const VariableMap &vars,
        const std::map<std::string, void*> &in,
        const std::map<std::string, void*> &out,
        const unsigned int &itemCount,
        const unsigned int &array_length,
        unsigned int *d_write_flag = nullptr);

 private:
    CUDAScanCompaction scan;

 public:
    CUDAScatter() { }
    /**
     * Wipes out host mirrors of device memory
     * Only really to be used after calls to cudaDeviceReset()
     * @note Only currently used after some tests
     */
    void purge();
    // Public deleted creates better compiler errors
    CUDAScatter(CUDAScatter const&) = delete;
    void operator=(CUDAScatter const&) = delete;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASCATTER_H_

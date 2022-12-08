#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDASCANCOMPACTION_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDASCANCOMPACTION_H_
#include <driver_types.h>

namespace flamegpu {
// forward declare classes from other modules
class CUDASimulation;
namespace detail {

/**
 * A pair of device buffers for performing scan/compaction operations
 */
struct CUDAScanCompactionPtrs {
    /**
     * Array to mark whether an item is to be retained
     */
    unsigned int *scan_flag = nullptr;
    /**
     * scan_flag is exclusive summed into this array if messages are optional
     */
    unsigned int *position = nullptr;
};
/**
 * Scan and Compaction buffer data for a specific stream and scan type
 */
struct CUDAScanCompactionConfig {
   /**
    * Default constructor
    * Initially no memory is allocated, all buffers are empty.
    */
    CUDAScanCompactionConfig()
        : scan_flag_len(0)
    { }
    /**
     * Releases all allocated memory
     */
    ~CUDAScanCompactionConfig();
    /**
     * Copy construction is disabled
     */
    CUDAScanCompactionConfig(CUDAScanCompactionConfig const&) = delete;
    /**
     * Assignment is disabled
     */
    void operator=(CUDAScanCompactionConfig const&) = delete;
    unsigned int scan_flag_len = 0;
    /**
     * Structure of scan flag buffers
     */
    CUDAScanCompactionPtrs d_ptrs;
    /**
     * Release the two scan buffers inside d_ptrs
     */
    void free_scan_flag();
    /**
     * Resize the two scan buffers inside d_ptrs
     * @param count The number of items required to fit in the resized buffers
     */
    void resize_scan_flag(unsigned int count);
    /**
     * Reset all data inside the two scan buffers to 0
     * @param stream The CUDA stream used to execute the memset
     * @note This method is async, the cuda stream is not synchronised
     */
    void zero_scan_flag_async(cudaStream_t stream);
};

/**
 * Utility for managing storage of scan/compaction buffers shared between all functions of a particular stream
 */
class CUDAScanCompaction {
 public:
    /**
    * Number of valid values in Type
    */
    static const unsigned int MAX_TYPES = 3;
    /** 
     * As of Compute Capability 7.5; 128 is the max concurrent streams
     */
    static const unsigned int MAX_STREAMS = 128;
    /**
     * Different scan reasons have different buffers, as it's possible an agent function uses all at once
     */
    enum Type : unsigned int {
        MESSAGE_OUTPUT = 0,
        AGENT_DEATH = 1,
        AGENT_OUTPUT = 2
    };
    /**
     * Default constructor
     */
    CUDAScanCompaction() { }
    /**
     * Copy construction is disabled
     */
    CUDAScanCompaction(CUDAScanCompaction const&) = delete;
    /**
     * Assignment is disabled
     */
    void operator=(CUDAScanCompaction const&) = delete;
    /**
     * Resize the scan flag buffer for the specified stream and type for the provided number of items
     * @param newCount The number of scan flags that the resized buffer must be able to hold
     * @param type The type of the scan flag buffer to be resized
     * @param streamId The stream index of the scan flag buffer to be resized
     */
    void resize(unsigned int newCount, const Type& type, unsigned int streamId);
    /**
     * Reset all scan flags in the buffer for the specified stream and type to zero
     * @param type The type of the scan flag buffer to be zerod
     * @param stream The CUDA stream used to execute the memset
     * @param streamId The stream index of the scan flag buffer to be zerod
     * @note This method is async, the cuda stream is not synchronised
     */
    void zero_async(const Type& type, cudaStream_t stream, unsigned int streamId);
    /**
     * Returns a const reference to the scan flag config structure for the specified stream and type
     * @param type The type of the scan flag buffer to return
     * @param streamId The stream index of the scan flag buffer to return
     * @see Config() for the non-const variant.
     */
    const CUDAScanCompactionConfig &getConfig(const Type& type, unsigned int streamId);
    /**
     * Returns a reference to the scan flag config structure for the specified stream and type
     * @param type The type of the scan flag buffer to return
     * @param streamId The stream index of the scan flag buffer to return
     * @see getConfig() for the const variant.
     */
    CUDAScanCompactionConfig &Config(const Type& type, unsigned int streamId);

 private:
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    CUDAScanCompactionConfig configs[MAX_TYPES][MAX_STREAMS];
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDASCANCOMPACTION_H_

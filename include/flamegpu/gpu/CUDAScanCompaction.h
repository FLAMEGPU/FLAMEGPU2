#ifndef INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_

/**
 * PLEASE NOTE: This implementation currently assumes there is only one instance of CUDAAgentModel executing at once
 * PLEASE NOTE: There is not currently a mechanism to release these (could trigger something via CUDAAgentModel destructor)
 */

// forward declare classes from other modules
class CUDAAgentModel;

/**
 * Could make this cleaner with an array of a nested struct and enums for access, rather than copy paste/rename
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
struct CUDAScanCompactionConfig {
    CUDAScanCompactionConfig()
        : scan_flag_len(0)
        , hd_cub_temp(nullptr)
        , cub_temp_size(0)
        , cub_temp_size_max_list_size(0)
    { }
    ~CUDAScanCompactionConfig();
    CUDAScanCompactionConfig(CUDAScanCompactionConfig const&) = delete;
    void operator=(CUDAScanCompactionConfig const&) = delete;
    unsigned int scan_flag_len = 0;

    CUDAScanCompactionPtrs d_ptrs;

    void* hd_cub_temp = nullptr;
    size_t cub_temp_size = 0;
    unsigned int cub_temp_size_max_list_size = 0;

    void free_scan_flag();
    void resize_scan_flag(const unsigned int& count);
    void zero_scan_flag();
};

class CUDAScanCompaction {
 public:
    /**
    * Number of valid values in Type
    */
    static const unsigned int MAX_TYPES = 3;
    /** 
     * As of Compute Cability 7.5; 128 is the max concurrent streams
     */
    static const unsigned int MAX_STREAMS = 128;
    /**
     * Try and be a bit more dynamic by using 2d array here
     */
    enum Type : unsigned int {
        MESSAGE_OUTPUT = 0,
        AGENT_DEATH = 1,
        AGENT_OUTPUT = 2
    };

    CUDAScanCompaction() { }
    CUDAScanCompaction(CUDAScanCompaction const&) = delete;
    void operator=(CUDAScanCompaction const&) = delete;
    /**
     * Wipes out host mirrors of device memory
     * Only really to be used after calls to cudaDeviceReset()
     * @note Only currently used after some tests
     */
    void purge();

    void resize(const unsigned int& newCount, const Type& type, const unsigned int& streamId);
    void zero(const Type& type, const unsigned int& streamId);

    const CUDAScanCompactionConfig &getConfig(const Type& type, const unsigned int& streamId);
    CUDAScanCompactionConfig &Config(const Type& type, const unsigned int& streamId);

 private:
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    CUDAScanCompactionConfig configs[MAX_TYPES][MAX_STREAMS];
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_

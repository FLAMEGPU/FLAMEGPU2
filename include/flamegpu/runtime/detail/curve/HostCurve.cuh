#ifndef INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_HOSTCURVE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_HOSTCURVE_CUH_

#include <cstring>
#include <cstdio>
#include <mutex>
#include <shared_mutex>
#include <typeindex>
#include <string>

#include "flamegpu/runtime/detail/curve/Curve.cuh"
#include "flamegpu/exception/FLAMEGPUDeviceException.cuh"

namespace flamegpu {
// forward declare classes from other modules
class CUDAAgent;
namespace detail {
namespace curve {

/**
 * Host instance of Curve
 *
 * This class can be used to register variables within it's curve hash table
 * These variables can then have their values (pointers) updated at a later stage
 *
 * The hash table is represented by a Curve::CurveTable, which can be copied to device memory for use with DeviceCurve
 *
 * This interface does not support reading or unregistering variables,
 * it is expected that variables registered to a HostCurve will persist for the lifetime of the HostCurve instance
 */
class HostCurve {
 public:
    typedef Curve::Variable          Variable;                    // !< Typedef for cuRVE variable handle
    typedef Curve::VariableHash      VariableHash;                // !< Typedef for cuRVE variable name string hash
    typedef Curve::NamespaceHash     NamespaceHash;               // !< Typedef for cuRVE variable namespace string hash
    static const int MAX_VARIABLES = Curve::MAX_VARIABLES;        // !< Default maximum number of cuRVE variables (must be a power of 2)
    static const VariableHash EMPTY_FLAG = Curve::EMPTY_FLAG;

    /**
     * Registers the specified agent variable within the curve hash table, storing the specified metadata
     * Initially the variables devive ptr will be set to nullptr
     *
     * @param variable_name The name used to access the agent variable
     * @param type The type index of the variable, this is currently unused as device code does not support type checking
     * @param type_size Size of the data type (this should be the size of a single element if an array variable)
     * @param elements Number of elements (1 unless the variable is an array)
     * @return Variable Handle of registered variable or UNKNOWN_VARIABLE if an error is encountered.
     * @note It is recommend that you instead use the appropriate registerVariable() template function.
     */
    void registerAgentVariable(const std::string& variable_name, std::type_index type, size_t type_size, unsigned int elements);
    void registerMessageInputVariable(const std::string& variable_name, std::type_index type, size_t type_size, unsigned int elements);
    void registerMessageOutputVariable(const std::string &variable_name, std::type_index type, size_t type_size, unsigned int elements);
    void registerAgentOutputVariable(const std::string& variable_name, std::type_index type, size_t type_size, unsigned int elements);
    void registerSetEnvironmentProperty(const std::string &variable_name, std::type_index type, size_t type_size, unsigned int elements, ptrdiff_t offset);
    void registerSetMacroEnvironmentProperty(const std::string &variable_name, std::type_index type, size_t type_size, unsigned int elements, void* d_ptr);
    /**
     * Updates the device pointer stored for the specified agent variable
     * @param variable_name The name of the variable to update
     * @param d_ptr The pointer to the variables buffer in device memory
     * @param agent_count The number of variables in the buffer (currently unused?)
     */
    void setAgentVariable(const std::string& variable_name, void* d_ptr, unsigned int agent_count);
    void setMessageInputVariable(const std::string& variable_name, void *d_ptr, unsigned int message_in_count);
    void setMessageOutputVariable(const std::string& variable_name, void *d_ptr, unsigned int message_out_count);
    void setAgentOutputVariable(const std::string& variable_name, void *d_ptr, unsigned int agent_count);
    /**
     * Check how many items are in the hash table
     *
     * @return The number of items currently registered in the hash table
     */
    int size() const;
    /**
     * Copy host structures to device
     *
     * This function copies the host hash table to the device, it must be used prior to launching agent functions (and agent function conditions) if Curve has been updated.
     * 1 memcpy to device is always performed, CURVE does not track whether it has been changed internally.
     * @param stream cuda stream for the copy
     */
    void updateDevice_async(cudaStream_t stream);

 private:
     void registerVariable(VariableHash variable_hash, std::type_index type, size_t type_size, unsigned int elements);
     void setVariable(VariableHash variable_hash, void* d_ptr, unsigned int count = 0);

    /**
     * Initialises cuRVE on the currently active device.
     * This performs a single cudaMalloc() if the pointer is null, the memory is not updated until use
     */
    void initialiseDevice();
    /**
     * Wipes out host mirrors of device memory and reallocates the device buffer
     * Only really to be used after calls to cudaDeviceReset(), as this does not free device allocations
     */
    void purge();
    /**
     * Has access to call purge
     */
    friend class flamegpu::CUDAAgent;
    /**
     * Host and device storage of curve table
     */
    CurveTable h_curve_table, *d_curve_table;
    /**
     * Internal tracking of type_index of stored variables
     * Not currently support in device code, but maintained incase future-support is added
     */
    // std::type_index h_curve_table_ext_type[MAX_VARIABLES]; //@todo How to init this safely?
    /**
     * Device curve does not currently differentiate between array-variable length, and agent count
     */
    unsigned int h_curve_table_ext_count[MAX_VARIABLES];
    /**
     * Namespace hashes used to calculate variable hashes
     * @todo These could be static inside Curve/namespace
     */
    const NamespaceHash message_in_hash, message_out_hash, agent_out_hash, environment_hash, macro_environment_hash;

 public:
     /**
      * Default constructor
      */
     HostCurve();
     /**
      * Default destructor
      * @note This attempts to free the Curve device buffer, so may throw an exception if the device has been reset, and purge() was not called
      */
     ~HostCurve();
     /**
      * Returns the pointer to curve in device memory
      */
     const CurveTable* getDevicePtr() const;
};

}  // namespace curve
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_HOSTCURVE_CUH_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_RTC_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_RTC_CUH_

#include <driver_types.h>
#include <array>
#include <cstring>
#include <string>
#include <cstdio>
#include <typeindex>
#include <map>
#include <vector>

#include "flamegpu/util/StringPair.h"

namespace jitify {
namespace experimental {
class KernelInstantiation;
}  // namespace experimental
}  // namespace jitify
namespace flamegpu {
namespace detail {
namespace curve {


/**
 * The Curve RTC host is a class for dynamically building a header file for use in RTC functions.
 * Rather than providing a hashmap of string variable names it will dynamically create a header with agent variables directly accessible via compile time string comparisons.
 * This must be kept around after the header has been compiled, as it also provides access for updating the data available via Curve
 */
class CurveRTCHost {
 public:
    /**
     * Default constructor
     */
    CurveRTCHost();
    /**
     * Destructor
     * Frees allocated memory
     */
    ~CurveRTCHost();
    /**
     * Specify an agent variable to be included in the dynamic header
     * @param variableName The variable's name
     * @param type The name of the variable's type (%std::type_index::name())
     * @param type_size The type size of the variable's base type (sizeof()), this is the size of a single element if the variable is an array variable. 
     * @param elements The number of elements in the variable (1 unless the variable is an array variable)
     * @param read True if the variable should be readable
     * @param write True if the variable should be writable
     * @throws exception::UnknownInternalError If an agent variable with the same name is already registered
     */
    void registerAgentVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements = 1, bool read = true, bool write = true);
    /**
     * Specify an output message variable to be included in the dynamic header
     * @param variableName The variable's name
     * @param type The name of the variable's type (%std::type_index::name())
     * @param type_size The type size of the variable's base type (sizeof()), this is the size of a single element if the variable is an array variable.
     * @param elements The number of elements in the variable (1 unless the variable is an array variable)
     * @param read True if the variable should be readable
     * @param write True if the variable should be writable
     * @throws exception::UnknownInternalError If an output message variable with the same name is already registered
     */
    void registerMessageOutVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements = 1, bool read = true, bool write = true);
    /**
     * Specify an input message variable to be included in the dynamic header
     * @param variableName The variable's name
     * @param type The name of the variable's type (%std::type_index::name())
     * @param type_size The type size of the variable's base type (sizeof()), this is the size of a single element if the variable is an array variable.
     * @param elements The number of elements in the variable (1 unless the variable is an array variable)
     * @param read True if the variable should be readable
     * @param write True if the variable should be writable
     * @throws exception::UnknownInternalError If an input message variable with the same name is already registered
     */
    void registerMessageInVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements = 1, bool read = true, bool write = true);
    /**
     * Specify an output agent variable (device agent birth) to be included in the dynamic header
     * @param variableName The variable's name
     * @param type The name of the variable's type (%std::type_index::name())
     * @param type_size The type size of the variable's base type (sizeof()), this is the size of a single element if the variable is an array variable.
     * @param elements The number of elements in the variable (1 unless the variable is an array variable)
     * @param read True if the variable should be readable
     * @param write True if the variable should be writable
     * @throws exception::UnknownInternalError If an output agent variable with the same name is already registered
     */
    void registerNewAgentVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements = 1, bool read = true, bool write = true);
    /**
     * Specify an environment directed graph vertex property to be included in the dynamic header
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @param type The name of the property's type (%std::type_index::name())
     * @param type_size The type size of the property's base type (sizeof()), this is the size of a single element if the variable is an array property.
     * @param elements The number of elements in the property (1 unless the property is an array property)
     * @param read True if the property should be readable
     * @param write True if the property should be writable
     * @throws exception::UnknownInternalError If an environment directed graph vertex property with the same name is already registered
     */
    void registerEnvironmentDirectedGraphVertexProperty(const std::string& graphName, const std::string& propertyName, const char* type, size_t type_size, unsigned int elements = 1, bool read = true, bool write = false);
    /**
     * Specify an environment directed graph edge property to be included in the dynamic header
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @param type The name of the property's type (%std::type_index::name())
     * @param type_size The type size of the property's base type (sizeof()), this is the size of a single element if the variable is an array property.
     * @param elements The number of elements in the property (1 unless the property is an array property)
     * @param read True if the property should be readable
     * @param write True if the property should be writable
     * @throws exception::UnknownInternalError If an environment directed graph edge property with the same name is already registered
     */
    void registerEnvironmentDirectedGraphEdgeProperty(const std::string &graphName, const std::string &propertyName, const char* type, size_t type_size, unsigned int elements = 1, bool read = true, bool write = false);

    /**
     * Unregister an agent variable, so that it is nolonger included in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void unregisterAgentVariable(const char* variableName);
    /**
     * Unregister an output message variable, so that it is nolonger included in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void unregisterMessageOutVariable(const char* variableName);
    /**
     * Unregister an input message variable, so that it is nolonger included in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void unregisterMessageInVariable(const char* variableName);
    /**
     * Unregister an output agent variable (device agent birth), so that it is nolonger included in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void unregisterNewAgentVariable(const char* variableName);
    /**
     * Unregister an environment directed graph vertex property, so that it is nolonger included in the dynamic header
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void unregisterEnvironmentDirectedGraphVertexProperty(const std::string& graphName, const std::string& propertyName);
    /**
     * Unregister an environment directed graph edge property, so that it is nolonger included in the dynamic header
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void unregisterEnvironmentDirectedGraphEdgeProperty(const std::string& graphName, const std::string& propertyName);
    /**
     * Returns a host pointer to the memory which stores the device pointer to be included for the specified variable in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void* getAgentVariableCachePtr(const char* variableName);
    /**
     * Returns a host pointer to the memory which stores the device pointer to be included for the specified variable in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void* getMessageOutVariableCachePtr(const char* variableName);
    /**
     * Returns a host pointer to the memory which stores the device pointer to be included for the specified variable in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void* getMessageInVariableCachePtr(const char* variableName);
    /**
     * Returns a host pointer to the memory which stores the device pointer to be included for the specified variable in the dynamic header
     * @param variableName The variable's name
     * @throws exception::UnknownInternalError If the specified variable is not registered
     */
    void* getNewAgentVariableCachePtr(const char* variableName);
    /**
     * Returns a host pointer to the memory which stores the device pointer to be included for the specified property in the dynamic header
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @throws exception::UnknownInternalError If the specified property is not registered
     */
    void* getEnvironmentDirectedGraphVertexPropertyCachePtr(const std::string& graphName, const std::string& propertyName);
    /**
     * Returns a host pointer to the memory which stores the device pointer to be included for the specified property in the dynamic header
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @throws exception::UnknownInternalError If the specified property is not registered
     */
    void* getEnvironmentDirectedGraphEdgePropertyCachePtr(const std::string& graphName, const std::string& propertyName);
    void setAgentVariableCount(const std::string& variableName, unsigned int count);
    void setMessageOutVariableCount(const std::string& variableName, unsigned int count);
    void setMessageInVariableCount(const std::string& variableName, unsigned int count);
    void setNewAgentVariableCount(const std::string& variableName, unsigned int count);
    /**
     * Set the number of vertices in the named buffer
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @param count The value to set
     * @throws exception::UnknownInternalError If the specified property is not registered
     */
    void setEnvironmentDirectedGraphVertexPropertyCount(const std::string& graphName, const std::string& propertyName, unsigned int count);
    /**
     * Set the number of edges in the named buffer
     * @param graphName The properties's graph's name
     * @param propertyName The properties's name
     * @param count The value to set
     * @throws exception::UnknownInternalError If the specified property is not registered
     */
    void setEnvironmentDirectedGraphEdgePropertyCount(const std::string& graphName, const std::string& propertyName, unsigned int count);
    /**
     * Specify an environment property to be included in the dynamic header
     * @param propertyName The property's name
     * @param offset The property's offset within the environment property map (EnvironmentManager)
     * @param type The name of the property's type (%std::type_index::name())
     * @param type_size The type size of the property's base type (sizeof()), this is the size of a single element if the property is an array property.
     * @param elements The number of elements in the property (1 unless the property is an array property)
     * @throws exception::UnknownInternalError If an environment property with the same name is already registered
     */
    void registerEnvVariable(const char* propertyName, ptrdiff_t offset, const char* type, size_t type_size, unsigned int elements = 1);
    /**
     * Unregister an environment property, so that it is nolonger included in the dynamic header
     * @param propertyName The property's name
     * @throws exception::UnknownInternalError If the specified property is not registered
     */
    void unregisterEnvVariable(const char* propertyName);
    /**
     * Specify an environment macro property to be included in the dynamic header
     * @param propertyName The property's name
     * @param d_ptr Pointer to the buffer in device memory
     * @param type The name of the property's type (%std::type_index::name())
     * @param type_size The type size of the property's base type (sizeof()), this is the size of a single element if the property is an array property.
     * @param dimensions The number of elements in the property (1 unless the property is an array property)
     * @throws exception::UnknownInternalError If an environment property with the same name is already registered
     */
    void registerEnvMacroProperty(const char* propertyName, void* d_ptr, const char* type, size_t type_size, const std::array<unsigned int, 4>& dimensions);
    /**
     * Unregister an environment property, so that it is nolonger included in the dynamic header
     * @param propertyName The property's name
     * @throws exception::UnknownInternalError If the specified property is not registered
     */
    void unregisterEnvMacroProperty(const char* propertyName);
    /**
     * Set the filename tagged in the file (goes into a #line statement)
     * @param filename Name to be used for the file in compile errors
     * @note Do not include quotes
     */
    void setFileName(const std::string& filename);
    /**
     * Generates and returns the dynamic header based on the currently registered variables and properties
     * @param env_buffer_len Length of the environment managers buffer
     * @return The dynamic Curve header
     */
    std::string getDynamicHeader(size_t env_buffer_len);
    /**
     * @return The identifier used for the environment property cache within the dynamic header
     */
    static std::string getVariableSymbolName();

    /**
     * Demangle a verbose type name (e.g. std::type_index.name().c_str()) into a user readable type
     * This is required as different compilers will perform name mangling in different way (or not at all).
     * @param verbose_name The verbose type name to be demangled
     * @return The demangled type name
     */
    static std::string demangle(const char* verbose_name);

    /**
     * Demangle from a std::type_index into a user readable type
     * This is required as different compilers will perform name mangling in different way (or not at all).
     * @param type The type to return the demangled name for
     * @return The demangled type name of the provided type
     */
    static std::string demangle(const std::type_index& type);
    /**
     * Copies the environment managers cache to the rtc header cache
     * @param d_env_ptr Device pointer to the Environment managers cache
     * @param bufferLen Length of the buffer
     */
    void updateEnvCache(const void* d_env_ptr, const size_t bufferLen);
    /**
     * Copy h_data_buffer to device
     * @param instance The compiled RTC agent function instance to copy the environment cache to
     * @param stream The CUDA stream used for the cuda memcpy
     * @note This is async, the stream is non synchronised
     */
    void updateDevice_async(const jitify::experimental::KernelInstantiation& instance, cudaStream_t stream);

 protected:
   /**
    * Utility method for replacing tokens within the dynamic header with dynamically computed strings
    * @param placeholder String to locate within the header
    * @param dst Replacement for the string located within the header
    * @throws exception::UnknownInternalError If placeholder could not be found within the header
    */
    void setHeaderPlaceholder(std::string placeholder, std::string dst);
    /**
     * Properties for a registered agent/message-in/message-out/agent-out variable
     */
    struct RTCVariableProperties {
        /**
         * Name of the variable's base type (e.g. type of an individual element if array variable)
         */
        std::string type;
        /**
         * True if the variable should be readable
        */
        bool read;
        /**
         * True if the variable should be writable
        */
        bool write;
        /**
         * Number of elements, 1 unless an array variable
         */
        unsigned int elements;
        /**
         * Size of the variable's base type (e.g. size of an individual element if array variable)
         */
        size_t type_size;
        /**
         * Pointer to a location in host memory where the device pointer to this variables buffer must be stored
         */
        void *h_data_ptr;
        /**
         * Index in the count buffer where the count is stored
         * Count being the number of agents/messages/vertices/edges etc in the buffer
         */
        unsigned int count_index;
    };
    /**
     * Properties for a registered environment property
     */
    struct RTCEnvVariableProperties {
        /**
         * Name of the property's base type (e.g. type of an individual element if array property)
         */
        std::string type;
        /**
         * Number of elements, 1 unless an array property
         */
        unsigned int elements;
        /**
         * Offset to the properties data inside the EnvironmentManager's RTC cache
         */
        ptrdiff_t offset;
        /**
         * Size of the property's base type (e.g. size of an individual element if array property)
         */
        size_t type_size;
    };
    /**
     * Properties for a registered environment macro property
     */
    struct RTCEnvMacroPropertyProperties {
        /**
         * Name of the property's base type (e.g. type of an individual element if array property)
         */
        std::string type;
        /**
         * Number of elemements in each dimension
         */
        std::array<unsigned int, 4> dimensions;
        /**
         * Size of the property's base type (e.g. size of an individual element if array property)
         */
        size_t type_size;
        /**
         * Copy of the device pointer
         * @note This assumes it will never be reallocated/resized after registration
         */
        void* d_ptr;
        /**
         * Pointer to a location in host memory where the device pointer to this variables buffer must be stored
         */
        void* h_data_ptr;
    };

 private:
    /**
     * Sub-method for setting up the Environment within the dynamic header
     * @param env_buffer_len Length of the environment managers buffer
     */
    void initHeaderEnvironment(size_t env_buffer_len);
    /**
     * Sub-method for setting up the variable/property set methods
     */
    void initHeaderSetters();
    /**
     * Sub-method for setting up the variable/property get methods
     */
    void initHeaderGetters();
    /**
     * Initialise all the variable h_data_ptr properties
     * This should only be called once during the init chain
     * @throws exception::InvalidOperation If this method has already been called
     */
    void initDataBuffer();
    /**
     * The dynamically generated header
     * Empty string until getDynamicHeader() has been called
     */
    std::string header;
    /**
     * The template used to build the dynamic header
     */
    static const char* curve_rtc_dynamic_h_template;
    /**
     * Offset into h_data_buffer where agent variable data begins
     */
    size_t agent_data_offset = 0;
    /**
     * Offset into h_data_buffer where output message variable data begins
     */
    size_t messageOut_data_offset = 0;
    /**
     * Offset into h_data_buffer where input message variable data begins
     */
    size_t messageIn_data_offset = 0;
    /**
     * Offset into h_data_buffer where output agent (device agent birth) variable data begins
     */
    size_t newAgent_data_offset = 0;
    /**
     * Offset into h_data_buffer where environment directed graph vertex property data begins
     */
    size_t directedGraphVertex_data_offset = 0;
    /**
     * Offset into h_data_buffer where environment directed graph edge property data begins
     */
    size_t directedGraphEdge_data_offset = 0;
    /**
     * Offset into h_data_buffer where output agent (device agent birth) variable data begins
     */
    size_t envMacro_data_offset = 0;
    /**
     * Offset into h_data_buffer where count buffer data begins
     */
    size_t count_data_offset = 0;
    /**
     * Size of the allocation pointed to by h_data_buffer
     */
    size_t data_buffer_size = 0;
    /**
     * Host copy of the RTC data buffer
     */
    char * h_data_buffer = nullptr;
    /**
     * Registered agent variable properties
     * <name, RTCVariableProperties>
     */
    std::map<std::string, RTCVariableProperties> agent_variables;
    /**
     * Registered output message variable properties
     * <name, RTCVariableProperties>
     */
    std::map<std::string, RTCVariableProperties> messageOut_variables;
    /**
     * Registered input message variable properties
     * <name, RTCVariableProperties>
     */
    std::map<std::string, RTCVariableProperties> messageIn_variables;
    /**
     * Registered agent out (device agent birth) variable properties
     * <name, RTCVariableProperties>
     */
    std::map<std::string, RTCVariableProperties> newAgent_variables;
    /**
     * Registered environment directed graph vertex properties
     * <<graphName, propertyName>, RTCVariableProperties>
     */
    util::StringPairMap<RTCVariableProperties> directedGraph_vertexProperties;
    /**
     * Registered environment directed graph edge properties
     * <<graphName, propertyName>, RTCVariableProperties>
     */
    util::StringPairMap<RTCVariableProperties> directedGraph_edgeProperties;
    /**
     * Registered environment property properties
     * <name, RTCVariableProperties>
     */
    std::map<std::string, RTCEnvVariableProperties> RTCEnvVariables;
    /**
     * Registered environment macro property properties
     * <name, RTCVariableProperties>
     */
    std::map<std::string, RTCEnvMacroPropertyProperties> RTCEnvMacroProperties;
    /**
     * Holds the number of agents/messages/vertices/edges/etc in the buffer which holds the index
     */
    std::vector<unsigned int> count_buffer;
};

}  // namespace curve
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_RTC_CUH_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_RTC_H_
#define INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_RTC_H_

#include <cstring>
#include <string>
#include <cstdio>
#include <typeindex>
#include <map>

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
     * Generates and returns the dynamic header based on the currently registered variables and properties
     * @return The dynamic Curve header
     */
    std::string getDynamicHeader();
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
     * Copies EnvironmentManager::MAX_BUFFER_SIZE bytes from env_ptr to h_data_buffer
     * This should be used to copy the EnvironmentManager's rtc_cache
     * @param env_ptr Pointer to the Environment managers rtc_cache
     */
    void updateEnvCache(const char* env_ptr);
    /**
     * Copy h_data_buffer to device
     * @param instance The compiled RTC agent function instance to copy the environment cache to
     */
    void updateDevice(const jitify::experimental::KernelInstantiation& instance);

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

 private:
    /**
     * Sub-method for setting up the Environment within the dynamic header
     */
    void initHeaderEnvironment();
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
    size_t msgOut_data_offset = 0;
    /**
     * Offset into h_data_buffer where input message variable data begins
     */
    size_t msgIn_data_offset = 0;
    /**
     * Offset into h_data_buffer where output agent (device agent birth) variable data begins
     */
    size_t newAgent_data_offset = 0;
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
     * Registered environment property properties
     * <name, RTCVariableProperties>
     */
    std::map<std::string, RTCEnvVariableProperties> RTCEnvVariables;
};

}  // namespace curve
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_RTC_H_

#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAMACROENVIRONMENT_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAMACROENVIRONMENT_H_

#include <cuda_runtime.h>

#include <map>
#include <utility>
#include <string>
#include <typeindex>
#include <array>
#include <vector>
#include <memory>

#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/runtime/detail/curve/HostCurve.cuh"
#include "flamegpu/runtime/environment/HostMacroProperty.cuh"
#include "flamegpu/detail/cuda.cuh"

// forward declare classes from other modules

namespace flamegpu {
struct SubEnvironmentData;
struct AgentFunctionData;
struct EnvironmentData;
class CUDASimulation;
namespace detail {
namespace curve {
class CurveRTCHost;
}  // namespace curve


/**
 * This class is CUDASimulation's internal handler for macro environment functionality
 */
class CUDAMacroEnvironment {
    /**
     * Used to group items required by properties
     */
    struct MacroEnvProp {
        /**
         * @param _type The type index of the base type (e.g. typeid(float))
         * @param _type_size The size of the base type (e.g. sizeof(float))
         * @param _elements Number of elements in each dimension
         */
        MacroEnvProp(const std::type_index& _type, const size_t _type_size, const std::array<unsigned int, 4> &_elements)
            : type(_type)
            , type_size(_type_size)
            , elements(_elements)
            , d_ptr(nullptr)
            , is_sub(false) { }
        ~MacroEnvProp() {
            if (d_ptr && !is_sub) {
                gpuErrchk(flamegpu::detail::cuda::cudaFree(d_ptr));
            }
        }
        MacroEnvProp(const MacroEnvProp& other) = delete;
        MacroEnvProp(MacroEnvProp&& other)
            : type(other.type)
            , type_size(other.type_size)
            , elements(other.elements)
            , d_ptr(other.d_ptr)
            , is_sub(other.is_sub) {
            other.d_ptr = nullptr;
        }
        std::type_index type;
        size_t type_size;
        std::array<unsigned int, 4> elements;
        void *d_ptr;
        // Denotes whether d_ptr is owned by this struct or not
        bool is_sub;
        // ptrdiff_t rtc_offset;  // This is set by buildRTCOffsets();
    };
    const CUDASimulation& cudaSimulation;
    std::map<std::string, MacroEnvProp> properties;
    std::map<std::string, std::weak_ptr<HostMacroProperty_MetaData>> host_cache;

 public:
    /**
     * Normal constructor
     * @param description Agent description of the agent
     * @param _cudaSimulation Parent CUDASimulation of the agent
     */
    CUDAMacroEnvironment(const EnvironmentData& description, const CUDASimulation& _cudaSimulation);
    CUDAMacroEnvironment(CUDAMacroEnvironment&) = delete;
    CUDAMacroEnvironment(CUDAMacroEnvironment&&) = delete;
    /**
     * Performs CUDA allocations, and registers CURVE variables
     */
    void init(cudaStream_t stream);
    /**
     * Performs CUDA allocations, and registers CURVE variables
     * Initialises submodel mappings too
     * @param mapping The SubEnvironment mapping info
     * @param master_macro_env The master model's macro env to map sub macro properties with
     * @param stream The CUDAStream to use for CUDA operations
     * @note This must be called after the master model CUDAMacroEnvironment has init
     */
    void init(const SubEnvironmentData& mapping, const CUDAMacroEnvironment& master_macro_env, cudaStream_t stream);
    /**
     * Release all CUDA allocations, and unregisters CURVE variables
     */
    void free();
    /**
     * Register the properties to the provided RTC header
     * @param curve_header The RTC header to act upon
     */
    void mapRTCVariables(detail::curve::CurveRTCHost& curve_header) const;
    /**
     * Release the properties to the provided RTC header
     * @param curve_header The RTC header to act upon
     */
    void unmapRTCVariables(detail::curve::CurveRTCHost& curve_header) const;
    /**
     * Register (and set) all owned macro properties within the specified curve instance
     * @param curve The curve instance to act upon
     */
    void registerCurveVariables(detail::curve::HostCurve &curve) const;

#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    /**
     * Reset the flags used by seatbelts to catch potential race conditions
     * @param streams Streams to async reset over
     */
    void resetFlagsAsync(const std::vector<cudaStream_t>& streams);
    /**
     * Returns the current state of the device read flag for the named macro property
     * @param property_name Name of the macro property to query
     */
    bool getDeviceReadFlag(const std::string& property_name);
    /**
     * Returns the current state of the device read flag for the named macro property
     * @param property_name Name of the macro property to query
     */
    bool getDeviceWriteFlag(const std::string& property_name);
    /**
     * Returns the raw flag variable used for read write flag for the named macro property
     * @param property_name Name of the macro property to query
     * @note This means both can be checked with a single memcpy
     * @note Bit 0 being set means read has occurred, bit 1 being set means write has occurred
     */
    unsigned int getDeviceRWFlags(const std::string& property_name);
#endif
    /**
     * Returns a HostAPI style direct accessor the macro property
     * @param name Name of the macro property
     * @tparam T Type of the macro property
     * @tparam I 1st dimension length
     * @tparam J 2nd dimension length
     * @tparam K 3rd dimension length
     * @tparam W 4th dimension length
     */
    template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
    HostMacroProperty<T, I, J, K, W> getProperty(const std::string& name);
#ifdef SWIG
    /**
     * Returns a HostAPI style direct accessor the macro property for SWIG
     * @param name Name of the macro property
     * @tparam T Type of the macro property
     */
    template<typename T>
    HostMacroProperty_swig<T> getProperty_swig(const std::string& name);
#endif
};

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W> CUDAMacroEnvironment::getProperty(const std::string& name) {
    // Validation
    auto prop = properties.find(name);
    if (prop == properties.end()) {
        THROW flamegpu::exception::InvalidEnvProperty("Environment macro property with name '%s' not found, "
            "in HostEnvironment::getMacroProperty()\n",
            name.c_str());
    } else if (prop->second.type != std::type_index(typeid(T))) {
        THROW flamegpu::exception::InvalidEnvProperty("Environment macro property '%s' type mismatch '%s' != '%s', "
            "in HostEnvironment::getMacroProperty()\n",
            name.c_str(), std::type_index(typeid(T)).name(), prop->second.type.name());
    } else if (prop->second.elements != std::array<unsigned int, 4>{I, J, K, W}) {
        THROW flamegpu::exception::InvalidEnvProperty("Environment macro property '%s' dimensions mismatch (%u, %u, %u, %u) != (%u, %u, %u, %u), "
            "in HostEnvironment::getMacroProperty()\n",
            name.c_str(), I, J, K, W, prop->second.elements[0], prop->second.elements[1], prop->second.elements[2], prop->second.elements[3]);
    }
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int flags = getDeviceWriteFlag(name);
    if (flags & (1 << 1)) {
        THROW flamegpu::exception::InvalidOperation("Environment macro property '%s' was written to by an agent function in the same layer, "
            "accessing it with a host function in the same layer could cause a race condition, in CUDAMacroEnvironment::getProperty().",
            name.c_str());
    }
    const bool read_flag = flags & (1 << 0);
#else
    const bool read_flag = false;
#endif
    // See if there is a live metadata in cache
    auto cache = host_cache.find(name);
    if (cache != host_cache.end()) {
        if (cache->second.lock()) {
            return HostMacroProperty<T, I, J, K, W>(cache->second.lock());
        }
        host_cache.erase(cache);
    }
    auto ret = std::make_shared<HostMacroProperty_MetaData>(prop->second.d_ptr, prop->second.elements, sizeof(T), read_flag, name);
    host_cache.emplace(name, ret);
    return HostMacroProperty<T, I, J, K, W>(ret);
}

#ifdef SWIG
template<typename T>
HostMacroProperty_swig<T> CUDAMacroEnvironment::getProperty_swig(const std::string& name) {
    // Validation
    auto prop = properties.find(name);
    if (prop == properties.end()) {
        THROW flamegpu::exception::InvalidEnvProperty("Environment macro property with name '%s' not found, "
            "in HostEnvironment::getMacroProperty()\n",
            name.c_str());
    } else if (prop->second.type != std::type_index(typeid(T))) {
        THROW flamegpu::exception::InvalidEnvProperty("Environment macro property '%s' type mismatch '%s' != '%s', "
            "in HostEnvironment::getMacroProperty()\n",
            name.c_str(), std::type_index(typeid(T)).name(), prop->second.type.name());
    }
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int flags = getDeviceWriteFlag(name);
    if (flags & (1 << 1)) {
        THROW flamegpu::exception::InvalidOperation("Environment macro property '%s' was written to by an agent function in the same layer, "
            "accessing it with a host function in the same layer could cause a race condition, in CUDAMacroEnvironment::getProperty().",
            name.c_str());
    }
    const bool read_flag = flags & (1 << 0);
#else
    const bool read_flag = false;
#endif
    // See if there is a live metadata in cache
    auto cache = host_cache.find(name);
    if (cache != host_cache.end()) {
        if (cache->second.lock()) {
            return HostMacroProperty_swig<T>(cache->second.lock());
        }
        host_cache.erase(cache);
    }
    auto ret = std::make_shared<HostMacroProperty_MetaData>(prop->second.d_ptr, prop->second.elements, sizeof(T), read_flag, name);
    host_cache.emplace(name, ret);
    return HostMacroProperty_swig<T>(ret);
}
#endif

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAMACROENVIRONMENT_H_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEENVIRONMENT_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEENVIRONMENT_CUH_

#include <cuda_runtime.h>

#include <string>
#include <cassert>

#include "flamegpu/runtime/utility/HostEnvironment.cuh"

namespace flamegpu_internal {
    /**
     * Managed by HostEnvironment, returned whenever a failure state is reached
     */
    extern __constant__ uint64_t c_deviceEnvErrorPattern;
}

/**
 * Utility for accessing environmental properties
 * These can only be read within agent functions
 * They can be set and updated within host functions
 */
class DeviceEnvironment {
    /**
     * Constructs the object
     */
    friend class FLAMEGPU_DEVICE_API;
    /**
     * Performs runtime validation that CURVE_NAMESPACE_HASH matches host value
     */
    friend class EnvironmentManager;
    /**
     * Device accessible copy of curve namespace hash, this is precomputed from EnvironmentManager::CURVE_NAMESPACE_HASH
     * EnvironmentManager::EnvironmentManager() validates that this value matches
     */
    __host__ __device__ static constexpr unsigned int CURVE_NAMESPACE_HASH() { return 0X1428F902u; }
    /**
     * Hash of the model's name, this is added to CURVE_NAMESPACE_HASH and variable name hash to find curve hash
     */
    const Curve::NamespaceHash &modelname_hash;
    /**
     * Constructor, requires the model name hash to init modelname_hash
     * @param _modelname_hash Hash of model name generated by curveGetVariableHash()
     */
    __device__ __forceinline__ DeviceEnvironment(const Curve::NamespaceHash &_modelname_hash)
        : modelname_hash(_modelname_hash) { }

 public:
    /**
     * Recognisable error pattern returned by methods on failure
     */
    __host__ __device__ static constexpr uint64_t ERROR_PATTERN() { return 0XDEADBEEFDEAFBABEllu; }
    /**
     * Gets an environment property
     * @param name name used for accessing the property, this value should be a string literal e.g. "foobar"
     * @tparam T Type of the environment property being accessed
     * @tparam N Length of variable name, this should always be implicit if passing a string literal
     */
    template<typename T, unsigned int N>
    __device__ __forceinline__ const T &get(const char(&name)[N]) const;
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property, this value should be a string literal e.g. "foobar"
     * @tparam T Type of the environment property being accessed
     * @tparam N Length of variable name, this should always be implicit if passing a string literal
     */
    template<typename T, unsigned int N>
    __device__ __forceinline__ const T &get(const char(&name)[N], const EnvironmentManager::size_type &index) const;
    /**
     * Returns whether the named env property exists
     * @param name name used for accessing the property, this value should be a string literal e.g. "foobar"
     * @tparam N Length of variable name, this should always be implicit if passing a string literal
     */
    template<unsigned int N>
    __device__ __forceinline__ bool contains(const char(&name)[N]) const;
};


/**
 * Getters
 */
template<typename T, unsigned int N>
__device__ __forceinline__ const T &DeviceEnvironment::get(const char(&name)[N]) const {
    Curve::VariableHash cvh = CURVE_NAMESPACE_HASH() + modelname_hash + Curve::variableHash(name);
    // Error checking is internal to Curve::getVariablePtrByHash, returns nullptr on fail
    // get a pointer to the specific variable by offsetting by the provided index
    T *value_ptr = reinterpret_cast<T*>(Curve::getVariablePtrByHash(cvh, 0));
    if (value_ptr) {
        return *value_ptr;
    } else {
        curve_internal::d_curve_error = Curve::DEVICE_ERROR_UNKNOWN_VARIABLE;
        assert(false);
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(&flamegpu_internal::c_deviceEnvErrorPattern));
    }
}
template<typename T, unsigned int N>
__device__ __forceinline__ const T &DeviceEnvironment::get(const char(&name)[N], const EnvironmentManager::size_type &index) const {
    Curve::VariableHash cvh = CURVE_NAMESPACE_HASH() + modelname_hash + Curve::variableHash(name);
    // Error checking is internal to Curve::getVariablePtrByHash, returns nullptr on fail
    size_t offset = index * sizeof(T);
    // get a pointer to the specific variable by offsetting by the provided index
    T *value_ptr = reinterpret_cast<T*>(Curve::getVariablePtrByHash(cvh, offset));
    if (value_ptr) {
        return *value_ptr;
    } else {
        curve_internal::d_curve_error = Curve::DEVICE_ERROR_UNKNOWN_VARIABLE;
        assert(false);
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(&flamegpu_internal::c_deviceEnvErrorPattern));
    }
}

/**
 * Util
 */
template<unsigned int N>
__device__ __forceinline__ bool DeviceEnvironment::contains(const char(&name)[N]) const {
    Curve::VariableHash cvh = CURVE_NAMESPACE_HASH() + modelname_hash + Curve::variableHash(name);
    return Curve::getVariable(cvh) != Curve::UNKNOWN_VARIABLE;
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEENVIRONMENT_CUH_

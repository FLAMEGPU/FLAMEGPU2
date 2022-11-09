#ifndef INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_DEVICECURVE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_DEVICECURVE_CUH_

#include "flamegpu/runtime/detail/SharedBlock.h"
#include "flamegpu/runtime/detail/curve/Curve.cuh"
#include "flamegpu/exception/FLAMEGPUDeviceException_device.cuh"
#include "flamegpu/util/type_decode.h"

#ifdef USE_GLM
#ifdef __CUDACC__
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#else
#pragma diag_suppress = esa_on_defaulted_function_ignored
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#endif  // __CUDACC__
#include <glm/glm.hpp>
#endif  // USE_GLM

namespace flamegpu {
namespace detail {
namespace curve {

/** @brief    The DeviceAPI for accessing a Curve table stored on the device.
 *
 * cuRVE is a C library and this singleton class acts as a mechanism to ensure that any reference to the library is handled correctly.
 * For example multiple objects may which to request that curve is initialised. This class will ensure that this function call is only made once the first time that a cuRVEInstance is required.
 */
class DeviceCurve {
 public:
    typedef Curve::Variable          Variable;                    // !< Typedef for cuRVE variable handle
    typedef Curve::VariableHash      VariableHash;                // !< Typedef for cuRVE variable name string hash
    typedef Curve::NamespaceHash     NamespaceHash;               // !< Typedef for cuRVE variable namespace string hash
    static const int MAX_VARIABLES = Curve::MAX_VARIABLES;        // !< Default maximum number of cuRVE variables (must be a power of 2)static const VariableHash EMPTY_FLAG = Curve::EMPTY_FLAG;
    static const VariableHash EMPTY_FLAG = Curve::EMPTY_FLAG;
    static const int UNKNOWN_VARIABLE = -1;                       // !< value returned as a Variable if an API function encounters an error

 private:
    ////
    //// These are the two most central CURVE methods
    ////
    /**
     * Retrieve the index of the given hash within the cuRVE hashtable.
     *
     * @param variable_hash A cuRVE variable string hash from variableHash.
     * @return The index of the specified variable within the hash table or UNKNOWN_VARIABLE on failure.
     */
    __device__ __forceinline__ static Variable getVariableIndex(VariableHash variable_hash);
    /**
     * Retrieve a pointer to the variable of given name and offset
     *
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable.
     * @param offset an offset into the variable's buffer in bytes (offset is normally variable index * sizeof(T) * N).
     * @tparam T The return type requested of the variable (only used for type-checking when SEATBELTS==ON).
     * @tparam N The variable array length, 1 for non array variables (only used for type-checking when SEATBELTS==ON).
     * @tparam M The length of the string literal passed to variableName. This parameter should always be implicit, and does not need to be provided.
     * @return A generic pointer to the variable value. Will be nullptr if there is an error and a DeviceError has been raised.
     * @throws exception::DeviceError (Only when SEATBELTS==ON) If the specified variable is not found in the cuRVE hashtable, or it's details are invalid.
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static char* getVariablePtr(const char(&variableName)[M], VariableHash namespace_hash, unsigned int offset);
    ////
    //// These are the CURVE middle layer functions
    //// It is assumed, that the use of literals in the non-array methods, will be solved as const at compile time to make them free
    ////
    /**
     * Retrieve a single typed value from the cuRVE hashtable using a specified name and index
     *
     * Returns a single value of specified type from a variableHash using the given index position.
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable.
     * @param agent_index The index of the variable in the named variable buffer. This corresponds to the agent/message/? index within the agent/message/? population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T The return type requested of the variable.
     * @tparam N The variable array length, 1 for non array variables.
     * @tparam M The length of the string literal passed to variableName. This parameter should always be implicit, and does not need to be provided.
     * @throws exception::DeviceError (Only when SEATBELTS==ON) If the specified variable is not found in the cuRVE hashtable, or it's details are invalid.
     */
    template <typename T, unsigned int N = 1, unsigned int M>
    __device__ __forceinline__ static T getVariable(const char(&variableName)[M], VariableHash namespace_hash, unsigned int agent_index = 0, unsigned int array_index = 0);
    /**
     * @copydoc DeviceCurve::getVariable()
     * @note This uses the __ldg() intrinsic to access the variable via the read-only cache.
     */
    template <typename T, unsigned int N = 1, unsigned int M>
    __device__ __forceinline__ static T getVariable_ldg(const char(&variableName)[M], VariableHash namespace_hash, unsigned int agent_index = 0, unsigned int array_index = 0);
    /**
     * Set a single typed value from the cuRVE hashtable using a specified name and index.
     *
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable.
     * @param value The value to be stored.
     * @param agent_index The index of the variable in the named variable buffer. This corresponds to the agent/message/? index within the agent/message/? population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T The type variable to be stored.
     * @tparam N The variable array length, 1 for non array variables.
     * @tparam M The length of the string literal passed to variableName. This parameter should always be implicit, and does not need to be provided.
     * @throws exception::DeviceError (Only when SEATBELTS==ON) If the specified variable is not found in the cuRVE hashtable, or it's details are invalid.
     */
    template <typename T, unsigned int N = 1, unsigned int M>
    __device__ __forceinline__ static void setVariable(const char(&variableName)[M], VariableHash namespace_hash, T value, unsigned int agent_index = 0, unsigned int array_index = 0);

 public:
    /**
     * Fill the shared memory curve table from the supplied device pointer
     */
     __device__ __forceinline__ static void init(const CurveTable* __restrict__ d_curve_table) {
         using detail::sm;
         for (int idx = threadIdx.x; idx < Curve::MAX_VARIABLES; idx += blockDim.x) {
             sm()->curve_variables[idx] = d_curve_table->variables[idx];
             sm()->curve_hashes[idx] = d_curve_table->hashes[idx];
#if !defined(SEATBELTS) || SEATBELTS
             sm()->curve_type_size[idx] = d_curve_table->type_size[idx];
             sm()->curve_elements[idx] = d_curve_table->elements[idx];
             sm()->curve_count[idx] = d_curve_table->count[idx];
#endif
         }
    }
    ////
    //// These are the public CURVE API
    ////
    /**
     * Retrieve the specified agent/message (array) variable from the cuRVE hashtable.
     *
     * @param variableName A constant char array (C string) variable name.
     * @param index The index of the variable in the named variable buffer. This corresponds to the agent/message agent/message index within the agent/message population.
     * @tparam T The return type requested of the variable.
     * @tparam M The length of the string literal passed to variableName. This parameter should always be implicit, and does not need to be provided.
     * @return The requested variable
     * @throws exception::DeviceError (Only when SEATBELTS==ON) If the specified variable is not found in the cuRVE hashtable, or it's details are invalid.
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static T getAgentVariable(const char(&variableName)[M], unsigned int index);
    /**
     * @copydoc DeviceCurve::getAgentVariable()
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static T getMessageVariable(const char(&variableName)[M], unsigned int index);
    /**
     * @copydoc DeviceCurve::getAgentVariable()
     * @note This uses the __ldg() intrinsic to access the variable via the read-only cache.
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static T getAgentVariable_ldg(const char(&variableName)[M], unsigned int index);
    /**
     * @copydoc DeviceCurve::getAgentVariable()
     * @note This uses the __ldg() intrinsic to access the variable via the read-only cache.
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static T getMessageVariable_ldg(const char(&variableName)[M], unsigned int index);
    /**
     * @copydoc DeviceCurve::getAgentVariable()
     * @param array_index The index of the element in the named variable array.
     * @tparam N Length of the array variable specified by variableName
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable(const char(&variableName)[M], unsigned int variable_index, unsigned int array_index);
    /**
     * @copydoc DeviceCurve::getAgentArrayVariable()
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getMessageArrayVariable(const char(&variableName)[M], unsigned int variable_index, unsigned int array_index);
    /**
     * @copydoc DeviceCurve::getAgentArrayVariable()
     * @note This uses the __ldg() intrinsic to access the variable via the read-only cache.
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable_ldg(const char(&variableName)[M], unsigned int variable_index, unsigned int array_index);
    /**
     * @copydoc DeviceCurve::getAgentArrayVariable()
     * @note This uses the __ldg() intrinsic to access the variable via the read-only cache.
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getMessageArrayVariable_ldg(const char(&variableName)[M], unsigned int variable_index, unsigned int array_index);
    /**
     * Set the specified agent/message (array) variable from the cuRVE hashtable.
     *
     * @param variableName A constant char array (C string) variable name.
     * @param variable The value to be stored.
     * @param index The index of the variable in the named variable vector. This corresponds to the agent/message agent/message/new-agent index within the agent/message/new-agent population.
     * @tparam T The return type requested of the variable.
     * @tparam M The length of the string literal passed to variableName. This parameter should always be implicit, and does not need to be provided.
     * @throws exception::DeviceError (Only when SEATBELTS==ON) If the specified variable is not found in the cuRVE hashtable, or it's details are invalid.
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static void setAgentVariable(const char(&variableName)[M], T variable, unsigned int index);
    /**
     * @copydoc DeviceCurve::setAgentVariable()
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static void setMessageVariable(const char(&variableName)[M], T variable, unsigned int index);
    /**
     * @copydoc DeviceCurve::setAgentVariable()
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static void setNewAgentVariable(const char(&variableName)[M], T variable, unsigned int index);
    /**
     * @copydoc DeviceCurve::setAgentVariable()
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setAgentArrayVariable(const char(&variableName)[M], T variable, unsigned int variable_index, unsigned int array_index);
    /**
     * @copydoc DeviceCurve::setAgentVariable()
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setMessageArrayVariable(const char(&variableName)[M], T variable, unsigned int variable_index, unsigned int array_index);
    /**
     * @copydoc DeviceCurve::setAgentVariable()
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setNewAgentArrayVariable(const char(&variableName)[M], T variable, unsigned int variable_index, unsigned int array_index);

    /**
     * Retrieve the specified environment (array) property from the cuRVE hashtable.
     *
     * @param propertyName A constant char array (C string) property name.
     * @tparam T The return type requested of the property.
     * @tparam M The length of the string literal passed to variableName. This parameter should always be implicit, and does not need to be provided.
     * @return The requested property
     * @throws exception::DeviceError (Only when SEATBELTS==ON) If the specified property is not found in the cuRVE hashtable, or it's details are invalid.
     */
    template <typename T, unsigned int M>
    __device__ __forceinline__ static T getEnvironmentProperty(const char(&propertyName)[M]);
    /**
     * @copydoc DeviceCurve::getEnvironmentProperty()
     * @param array_index The index of the element in the named variable array.
     * @tparam N (Optional) Length of the array variable specified by variableName, available for parity with other APIs, checked if provided  (flamegpu must be built with SEATBELTS enabled for device error checking).
     */
    template <typename T, unsigned int N = 0, unsigned int M>
    __device__ __forceinline__ static T getEnvironmentArrayProperty(const char(&propertyName)[M], unsigned int array_index);

    /**
     * Retrieve the specified environment macro property buffer's pointer from the cuRVE hashtable.
     *
     * @param name A constant char array (C string) environment macro property name.
     * @tparam T The type of the requested environment macro property.
     * @tparam I The length of the 1st dimension of the environment macro property, default 1.
     * @tparam J The length of the 2nd dimension of the environment macro property, default 1.
     * @tparam K The length of the 3rd dimension of the environment macro property, default 1.
     * @tparam W The length of the 4th dimension of the environment macro property, default 1.
     * @tparam M The length of the string literal passed to variableName. This parameter should always be implicit, and does not need to be provided.
     * @throws exception::DeviceError (Only when SEATBELTS==ON) If the specified variable is not found in the cuRVE hashtable, or it's details are invalid.
     */
    template<typename T, unsigned int I = 1, unsigned int J = 1, unsigned int K = 1, unsigned int W = 1, unsigned int M>
    __device__ __forceinline__ static char *getEnvironmentMacroProperty(const char(&name)[M]);
};

////
//// Core CURVE API
////
__device__ __forceinline__ DeviceCurve::Variable DeviceCurve::getVariableIndex(const VariableHash variable_hash) {
    using detail::sm;
    // loop unrolling of hash collision detection
    // (This may inflate register usage based on the max number of iterations in some cases)
    for (unsigned int x = 0; x< MAX_VARIABLES; x++) {
        const Variable i = (variable_hash + x) & (MAX_VARIABLES - 1);
        if (sm()->curve_hashes[i] == variable_hash)
            return i;
    }
    return UNKNOWN_VARIABLE;
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ char* DeviceCurve::getVariablePtr(const char(&variableName)[M], const VariableHash namespace_hash, const unsigned int offset) {
    using detail::sm;
    const Variable cv = getVariableIndex(Curve::variableHash(variableName) + namespace_hash);
#if !defined(SEATBELTS) || SEATBELTS
    if (cv == UNKNOWN_VARIABLE) {
        DTHROW("Curve variable with name '%s' was not found.\n", variableName);
        return nullptr;
    } else if (sm()->curve_type_size[cv] != sizeof(typename type_decode<T>::type_t)) {
        DTHROW("Curve variable with name '%s', type size mismatch %u != %llu.\n", variableName, sm()->curve_type_size[cv], sizeof(typename type_decode<T>::type_t));
        return nullptr;
    } else if (!(sm()->curve_elements[cv] == type_decode<T>::len_t * N || (namespace_hash == Curve::variableHash("_environment") && N == 0))) {  // Special case, environment can avoid specifying N
        DTHROW("Curve variable with name '%s', variable array length mismatch %u != %u.\n", variableName, sm()->curve_elements[cv], type_decode<T>::len_t);
        return nullptr;
    } else if (offset >= sm()->curve_type_size[cv] * sm()->curve_elements[cv] * sm()->curve_count[cv]) {  // Note : offset is basically index * sizeof(T)
        DTHROW("Curve variable with name '%s', offset exceeds buffer length  %u >= %u.\n", offset, sm()->curve_type_size[cv] * sm()->curve_elements[cv] * sm()->curve_count[cv]);
        return nullptr;
    }
#endif
    // return a generic pointer to variable address for given offset
    return sm()->curve_variables[cv] + offset;
}
////
//// Middle Layer CURVE API
////
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getVariable(const char(&variableName)[M], const VariableHash namespace_hash, const unsigned int agent_index, const unsigned int array_index) {
    using detail::sm;
    const unsigned int buffer_offset = agent_index * static_cast<unsigned int>(sizeof(T)) * N + array_index * sizeof(typename type_decode<T>::type_t);
    T *value_ptr = reinterpret_cast<T*>(getVariablePtr<T, N>(variableName, namespace_hash, buffer_offset));

#if !defined(SEATBELTS) || SEATBELTS
    if (!value_ptr)
        return {};
#endif
    return *value_ptr;
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getVariable_ldg(const char(&variableName)[M], const VariableHash namespace_hash, const unsigned int agent_index, const unsigned int array_index) {
    const unsigned int buffer_offset = agent_index * static_cast<unsigned int>(sizeof(T)) * N + array_index * sizeof(typename type_decode<T>::type_t);
    T *value_ptr = reinterpret_cast<T*>(getVariablePtr<T, N>(variableName, namespace_hash, buffer_offset));

#if !defined(SEATBELTS) || SEATBELTS
    if (!value_ptr)
        return {};
#endif
    return __ldg(value_ptr);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setVariable(const char(&variableName)[M], const VariableHash namespace_hash, const T variable, const unsigned int agent_index, const unsigned int array_index) {
    const unsigned int buffer_offset = agent_index * static_cast<unsigned int>(sizeof(T)) * N + array_index * sizeof(typename type_decode<T>::type_t);
    T* value_ptr = reinterpret_cast<T*>(getVariablePtr<T, N>(variableName, namespace_hash, buffer_offset));

#if !defined(SEATBELTS) || SEATBELTS
    if (!value_ptr)
        return;
#endif
    *value_ptr = variable;
}
////
//// Public CURVE API
////
template <typename T, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentVariable(const char (&variableName)[M], unsigned int index) {
    using detail::sm;
    return getVariable<T>(variableName, 0, index);
}
template <typename T, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageVariable(const char (&variableName)[M], unsigned int index) {
    return getVariable<T>(variableName, Curve::variableHash("_message_in"), index);
}
template <typename T, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentVariable_ldg(const char (&variableName)[M], unsigned int index) {
    return getVariable_ldg<T>(variableName, 0, index);
}
template <typename T, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageVariable_ldg(const char (&variableName)[M], unsigned int index) {
    return getVariable_ldg<T>(variableName, Curve::variableHash("_message_in"), index);
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentArrayVariable(const char(&variableName)[M], unsigned int agent_index, unsigned int array_index) {
    return getVariable<T, N>(variableName, 0, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageArrayVariable(const char(&variableName)[M], unsigned int message_index, unsigned int array_index) {
    return getVariable<T, N>(variableName, Curve::variableHash("_message_in"), message_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentArrayVariable_ldg(const char(&variableName)[M], unsigned int agent_index, unsigned int array_index) {
    return getVariable_ldg<T, N>(variableName, 0, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageArrayVariable_ldg(const char(&variableName)[M], unsigned int message_index, unsigned int array_index) {
    return getVariable_ldg<T, N>(variableName, Curve::variableHash("_message_in"), message_index, array_index);
}

template <typename T, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setAgentVariable(const char(&variableName)[M], T variable, unsigned int index) {
    setVariable<T>(variableName, 0, variable, index);
}
template <typename T, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setMessageVariable(const char(&variableName)[M], T variable, unsigned int index) {
    setVariable<T>(variableName, Curve::variableHash("_message_out"), variable, index);
}
template <typename T, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setNewAgentVariable(const char(&variableName)[M], T variable, unsigned int index) {
    setVariable<T>(variableName, Curve::variableHash("_agent_birth"), variable, index);
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setAgentArrayVariable(const char(&variableName)[M], T variable, unsigned int agent_index, unsigned int array_index) {
    setVariable<T, N>(variableName, 0, variable, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setMessageArrayVariable(const char(&variableName)[M], T variable, unsigned int message_index, unsigned int array_index) {
    setVariable<T, N>(variableName, Curve::variableHash("_message_out"), variable, message_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setNewAgentArrayVariable(const char(&variableName)[M], T variable, unsigned int agent_index, unsigned int array_index) {
    setVariable<T, N>(variableName, Curve::variableHash("_agent_birth"), variable, agent_index, array_index);
}

template <typename T, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getEnvironmentProperty(const char(&propertyName)[M]) {
    using detail::sm;
    return  *reinterpret_cast<const T*>(sm()->env_buffer + reinterpret_cast<ptrdiff_t>(getVariablePtr<T, 1>(propertyName, Curve::variableHash("_environment"), 0)));
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getEnvironmentArrayProperty(const char(&propertyName)[M], unsigned int array_index) {
    using detail::sm;
    return *reinterpret_cast<const T*>(sm()->env_buffer + reinterpret_cast<ptrdiff_t>(getVariablePtr<T, N>(propertyName, Curve::variableHash("_environment"), array_index * sizeof(T))));
}

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W, unsigned int M>
__device__ __forceinline__ char* DeviceCurve::getEnvironmentMacroProperty(const char(&name)[M]) {
    return getVariablePtr<T, I*J*K*W>(name, Curve::variableHash("_macro_environment"), 0);
}
}  // namespace curve
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_DEVICECURVE_CUH_

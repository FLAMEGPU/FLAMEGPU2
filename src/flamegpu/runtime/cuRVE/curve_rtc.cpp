#include <sstream>

#include "flamegpu/runtime/cuRVE/curve_rtc.h"
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/runtime/utility/EnvironmentManager.cuh"

// jitify include for demangle
#ifdef _MSC_VER
#pragma warning(push, 2)
#include "jitify/jitify.hpp"
#pragma warning(pop)
#else
#include "jitify/jitify.hpp"
#endif

namespace flamegpu {


const char* CurveRTCHost::curve_rtc_dynamic_h_template = R"###(dynamic/curve_rtc_dynamic.h
#ifndef CURVE_RTC_DYNAMIC_H_
#define CURVE_RTC_DYNAMIC_H_

#include "flamegpu/exception/FGPUDeviceException.h"

namespace flamegpu {

template <unsigned int N, unsigned int I> struct StringCompare {
    __device__ inline static bool strings_equal_loop(const char(&a)[N], const char(&b)[N]) {
        return a[N - I] == b[N - I] && StringCompare<N, I - 1>::strings_equal_loop(a, b);
    }
};

template <unsigned int N> struct StringCompare<N, 1> {
    __device__ inline static bool strings_equal_loop(const char(&a)[N], const char(&b)[N]) {
        return a[0] == b[0];
    }
};

template <unsigned int N>
__device__ bool strings_equal(const char(&a)[N], const char(&b)[N]) {
    return StringCompare<N, N>::strings_equal_loop(a, b);
}

template <unsigned int N, unsigned int M>
__device__ bool strings_equal(const char(&a)[N], const char(&b)[M]) {
    return false;
}

/**
 * Dynamically generated version of Curve without hashing
 * Both environment data, and curve variable ptrs are stored in this buffer
 * Order: Env Data, Agent, MsgOut, MsgIn, NewAgent
 * EnvData size must be a multiple of 8 bytes
 */
$DYNAMIC_VARIABLES

class Curve {
    public:
    static const int UNKNOWN_VARIABLE = -1;

    typedef int                      Variable;
    typedef unsigned int             VariableHash;
    typedef unsigned int             NamespaceHash;

    enum DeviceError {
        DEVICE_ERROR_NO_ERRORS,
        DEVICE_ERROR_UNKNOWN_VARIABLE,
        DEVICE_ERROR_VARIABLE_DISABLED,
        DEVICE_ERROR_UNKNOWN_TYPE,
        DEVICE_ERROR_UNKNOWN_LENGTH
    };

    enum HostError {
        ERROR_NO_ERRORS,
        ERROR_UNKNOWN_VARIABLE,
        ERROR_TOO_MANY_VARIABLES
    };
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable(const char(&name)[N], VariableHash namespace_hash, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable(const char(&name)[N], VariableHash namespace_hash, unsigned int index);
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable_ldg(const char(&name)[N], VariableHash namespace_hash, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable_ldg(const char(&name)[N], VariableHash namespace_hash, unsigned int index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable(const char(&name)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable_ldg(const char(&name)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getArrayVariable_ldg(const char(&name)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setAgentVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setMessageVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setNewAgentVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setAgentArrayVariable(const char(&name)[M], VariableHash namespace_hash, T variable, unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setNewAgentArrayVariable(const char(&name)[M], VariableHash namespace_hash, T variable, unsigned int variable_index, unsigned int array_index);

};

template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getAgentVariable(const char (&name)[N], VariableHash namespace_hash, unsigned int index) {
$DYNAMIC_GETAGENTVARIABLE_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getMessageVariable(const char (&name)[N], VariableHash namespace_hash, unsigned int index) {
$DYNAMIC_GETMESSAGEVARIABLE_IMPL
}

template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getAgentVariable_ldg(const char (&name)[N], VariableHash namespace_hash, unsigned int index) {
$DYNAMIC_GETAGENTVARIABLE_LDG_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getMessageVariable_ldg(const char (&name)[N], VariableHash namespace_hash, unsigned int index) {
$DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T Curve::getAgentArrayVariable(const char(&name)[M], VariableHash namespace_hash, unsigned int index, unsigned int array_index) {
$DYNAMIC_GETAGENTARRAYVARIABLE_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T Curve::getAgentArrayVariable_ldg(const char(&name)[M], VariableHash namespace_hash, unsigned int index, unsigned int array_index) {
$DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL
}

template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setAgentVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index) {
$DYNAMIC_SETAGENTVARIABLE_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setMessageVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index) {
$DYNAMIC_SETMESSAGEVARIABLE_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setNewAgentVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index) {
$DYNAMIC_SETNEWAGENTVARIABLE_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void Curve::setAgentArrayVariable(const char(&name)[M], VariableHash namespace_hash, T variable, unsigned int index, unsigned int array_index) {
$DYNAMIC_SETAGENTARRAYVARIABLE_IMPL    
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void Curve::setNewAgentArrayVariable(const char(&name)[M], VariableHash namespace_hash, T variable, unsigned int index, unsigned int array_index) {
$DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL    
}

}  // namespace flamegpu 

// has to be included after definition of curve namespace
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"

namespace flamegpu {

template<typename T, unsigned int N>
__device__ __forceinline__ T DeviceEnvironment::getProperty(const char(&name)[N]) const {
$DYNAMIC_ENV_GETVARIABLE_IMPL
}

template<typename T, unsigned int N>
__device__ __forceinline__ T DeviceEnvironment::getProperty(const char(&name)[N], const unsigned int &index) const {
$DYNAMIC_ENV_GETARRAYVARIABLE_IMPL
}

template<unsigned int N>
__device__ __forceinline__ bool DeviceEnvironment::containsProperty(const char(&name)[N]) const {
$DYNAMIC_ENV_CONTAINTS_IMPL
}

}  // namespace flamegpu

#endif  // CURVE_RTC_DYNAMIC_H_
)###";


CurveRTCHost::CurveRTCHost() : header(CurveRTCHost::curve_rtc_dynamic_h_template) {
}

CurveRTCHost::~CurveRTCHost() {
    free(h_data_buffer);
}

void CurveRTCHost::registerAgentVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    if (!agent_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerAgentVariable()", variableName);
    }
}
void CurveRTCHost::registerMessageInVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    if (!messageIn_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerMessageInVariable()", variableName);
    }
}
void CurveRTCHost::registerMessageOutVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    if (!messageOut_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerMessageOutVariable()", variableName);
    }
}
void CurveRTCHost::registerNewAgentVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    if (!newAgent_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerNewAgentVariable()", variableName);
    }
}

void CurveRTCHost::unregisterAgentVariable(const char* variableName) {
    auto i = agent_variables.find(variableName);
    if (i != agent_variables.end()) {
        agent_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterAgentVariable()", variableName);
    }
}
void CurveRTCHost::unregisterMessageOutVariable(const char* variableName) {
    auto i = messageOut_variables.find(variableName);
    if (i != messageOut_variables.end()) {
        messageOut_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterMessageOutVariable()", variableName);
    }
}
void CurveRTCHost::unregisterMessageInVariable(const char* variableName) {
    auto i = messageIn_variables.find(variableName);
    if (i != messageIn_variables.end()) {
        messageIn_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterMessageInVariable()", variableName);
    }
}
void CurveRTCHost::unregisterNewAgentVariable(const char* variableName) {
    auto i = newAgent_variables.find(variableName);
    if (i != newAgent_variables.end()) {
        newAgent_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterNewAgentVariable()", variableName);
    }
}


void* CurveRTCHost::getAgentVariableCachePtr(const char* variableName) {
    const auto i = agent_variables.find(variableName);
    if (i != agent_variables.end()) {
        return i->second.h_data_ptr;
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getAgentVariableCachePtr()", variableName);
}
void* CurveRTCHost::getMessageOutVariableCachePtr(const char* variableName) {
    const auto i = messageOut_variables.find(variableName);
    if (i != messageOut_variables.end()) {
        return i->second.h_data_ptr;
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getMessageOutVariableCachePtr()", variableName);
}
void* CurveRTCHost::getMessageInVariableCachePtr(const char* variableName) {
    const auto i = messageIn_variables.find(variableName);
    if (i != messageIn_variables.end()) {
        return i->second.h_data_ptr;
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getMessageInVariableCachePtr()", variableName);
}
void* CurveRTCHost::getNewAgentVariableCachePtr(const char* variableName) {
    const auto i = newAgent_variables.find(variableName);
    if (i != newAgent_variables.end()) {
        return i->second.h_data_ptr;
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getNewAgentVariableCachePtr()", variableName);
}

void CurveRTCHost::registerEnvVariable(const char* propertyName, ptrdiff_t offset, const char* type, size_t type_size, unsigned int elements) {
    RTCEnvVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.elements = elements;
    props.offset = offset;
    props.type_size = type_size;
    if (!RTCEnvVariables.emplace(propertyName, props).second) {
        THROW exception::UnknownInternalError("Environment property with name '%s' is already registered, in CurveRTCHost::registerEnvVariable()", propertyName);
    }
}

void CurveRTCHost::unregisterEnvVariable(const char* propertyName) {
    auto i = RTCEnvVariables.find(propertyName);
    if (i != RTCEnvVariables.end()) {
        RTCEnvVariables.erase(propertyName);
    } else {
        THROW exception::UnknownInternalError("Environment property '%s' not found when removing environment property, in CurveRTCHost::unregisterEnvVariable()", propertyName);
    }
}


void CurveRTCHost::initHeaderEnvironment() {
    // Calculate size of, and generate dynamic variables buffer
    std::stringstream variables;
    data_buffer_size = EnvironmentManager::MAX_BUFFER_SIZE;
    if (data_buffer_size % sizeof(void*) != 0) {
        THROW exception::UnknownInternalError("EnvironmentManager::MAX_BUFFER_SIZE should be a multiple of %llu!", sizeof(void*));
    }
    agent_data_offset = data_buffer_size;     data_buffer_size += agent_variables.size() * sizeof(void*);
    msgOut_data_offset = data_buffer_size;    data_buffer_size += messageOut_variables.size() * sizeof(void*);
    msgIn_data_offset = data_buffer_size;     data_buffer_size += messageIn_variables.size() * sizeof(void*);
    newAgent_data_offset = data_buffer_size;  data_buffer_size += newAgent_variables.size() * sizeof(void*);
    variables << "__constant__  char " << getVariableSymbolName() << "[" << data_buffer_size << "];\n";
    setHeaderPlaceholder("$DYNAMIC_VARIABLES", variables.str());
    // generate Environment::get func implementation ($DYNAMIC_ENV_GETVARIABLE_IMPL)
    {
        std::stringstream getEnvVariableImpl;
        for (std::pair<std::string, RTCEnvVariableProperties> element : RTCEnvVariables) {
            RTCEnvVariableProperties props = element.second;
            if (props.elements == 1) {
                getEnvVariableImpl <<   "    if (strings_equal(name, \"" << element.first << "\")) {\n";
                getEnvVariableImpl <<   "#if !defined(SEATBELTS) || SEATBELTS\n";
                getEnvVariableImpl <<   "        if(sizeof(T) != " << element.second.type_size << ") {\n";
                getEnvVariableImpl <<   "            DTHROW(\"Environment property '%s' type mismatch.\\n\", name);\n";
                getEnvVariableImpl <<   "            return 0;\n";
                getEnvVariableImpl <<   "        }\n";
                getEnvVariableImpl <<   "#endif\n";
                getEnvVariableImpl <<   "        return *reinterpret_cast<T*>(reinterpret_cast<void*>(" << getVariableSymbolName() <<" + " << props.offset << "));\n";
                getEnvVariableImpl <<   "    };\n";
            }
        }
        getEnvVariableImpl <<           "#if !defined(SEATBELTS) || SEATBELTS\n";
        getEnvVariableImpl <<           "    DTHROW(\"Environment property '%s' was not found.\\n\", name);\n";
        getEnvVariableImpl <<           "#endif\n";
        getEnvVariableImpl <<           "    return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_ENV_GETVARIABLE_IMPL", getEnvVariableImpl.str());
    }
    // generate Environment::get func implementation for array variables ($DYNAMIC_ENV_GETARRAYVARIABLE_IMPL)
    {
        std::stringstream getEnvArrayVariableImpl;
        for (std::pair<std::string, RTCEnvVariableProperties> element : RTCEnvVariables) {
            RTCEnvVariableProperties props = element.second;
            if (props.elements > 1) {
                getEnvArrayVariableImpl << "    if (strings_equal(name, \"" << element.first << "\")) {\n";
                getEnvArrayVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                getEnvArrayVariableImpl << "        if(sizeof(T) != " << element.second.type_size << ") {\n";
                getEnvArrayVariableImpl << "            DTHROW(\"Environment array property '%s' type mismatch.\\n\", name);\n";
                getEnvArrayVariableImpl << "            return 0;\n";
                // Env var doesn't currently require user to specify length
                // getEnvArrayVariableImpl << "        } else if (N != " << element.second.elements << ") {\n";
                // getEnvArrayVariableImpl << "            DTHROW(\"Environment array property '%s' length mismatch.\\n\", name);\n";
                // getEnvArrayVariableImpl << "            return 0;\n";
                getEnvArrayVariableImpl << "        } else if (index >= " << element.second.elements << ") {\n";
                getEnvArrayVariableImpl << "            DTHROW(\"Environment array property '%s', index %d is out of bounds.\\n\", name, index);\n";
                getEnvArrayVariableImpl << "            return 0;\n";
                getEnvArrayVariableImpl << "        }\n";
                getEnvArrayVariableImpl << "#endif\n";
                getEnvArrayVariableImpl << "        return reinterpret_cast<T*>(reinterpret_cast<void*>(" << getVariableSymbolName() <<" + " << props.offset << "))[index];\n";
                getEnvArrayVariableImpl << "    };\n";
            }
        }
        getEnvArrayVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        getEnvArrayVariableImpl <<         "    DTHROW(\"Environment array property '%s' was not found.\\n\", name);\n";
        getEnvArrayVariableImpl <<         "#endif\n";
        getEnvArrayVariableImpl <<         "    return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_ENV_GETARRAYVARIABLE_IMPL", getEnvArrayVariableImpl.str());
    }
    // generate Environment::contains func implementation ($DYNAMIC_ENV_CONTAINTS_IMPL)
    {
        std::stringstream containsEnvVariableImpl;
        for (std::pair<std::string, RTCEnvVariableProperties> element : RTCEnvVariables) {
            RTCEnvVariableProperties props = element.second;
            if (props.elements == 1) {
                containsEnvVariableImpl <<   "    if (strings_equal(name, \"" << element.first << "\"))\n";
                containsEnvVariableImpl <<   "        return true;\n";
            }
        }
        containsEnvVariableImpl <<           "    return false;\n";
        setHeaderPlaceholder("$DYNAMIC_ENV_CONTAINTS_IMPL", containsEnvVariableImpl.str());
    }
}
void CurveRTCHost::initHeaderSetters() {
    // generate setAgentVariable func implementation ($DYNAMIC_SETAGENTVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setAgentVariableImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements == 1) {
                setAgentVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setAgentVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                setAgentVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                setAgentVariableImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setAgentVariableImpl << "                    return;\n";
                setAgentVariableImpl << "                }\n";
                setAgentVariableImpl << "#endif\n";
                setAgentVariableImpl << "              (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[index] = (T) variable;\n";
                setAgentVariableImpl << "              return;\n";
                setAgentVariableImpl << "          }\n";
            } else { ++ct; }
        }
        setAgentVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        setAgentVariableImpl <<         "          DTHROW(\"Agent variable '%s' was not found during setVariable().\\n\", name);\n";
        setAgentVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETAGENTVARIABLE_IMPL", setAgentVariableImpl.str());
    }
    // generate setMessageVariable func implementation ($DYNAMIC_SETMESSAGEVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setMessageVariableImpl;
        for (const auto &element : messageOut_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements == 1) {
                setMessageVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setMessageVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                setMessageVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                setMessageVariableImpl << "                    DTHROW(\"Message variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setMessageVariableImpl << "                    return;\n";
                setMessageVariableImpl << "                }\n";
                setMessageVariableImpl << "#endif\n";
                setMessageVariableImpl << "              (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << msgOut_data_offset + (ct++ * sizeof(void*)) << ")))[index] = (T) variable;\n";
                setMessageVariableImpl << "              return;\n";
                setMessageVariableImpl << "          }\n";
            } else { ++ct; }
        }
        setMessageVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        setMessageVariableImpl <<         "          DTHROW(\"Message variable '%s' was not found during setVariable().\\n\", name);\n";
        setMessageVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETMESSAGEVARIABLE_IMPL", setMessageVariableImpl.str());
    }
    // generate setNewAgentVariable func implementation ($DYNAMIC_SETNEWAGENTVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setNewAgentVariableImpl;
        for (const auto &element : newAgent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements == 1) {
                setNewAgentVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setNewAgentVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                setNewAgentVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                setNewAgentVariableImpl << "                    DTHROW(\"New agent variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setNewAgentVariableImpl << "                    return;\n";
                setNewAgentVariableImpl << "                }\n";
                setNewAgentVariableImpl << "#endif\n";
                setNewAgentVariableImpl << "              (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << newAgent_data_offset + (ct++ * sizeof(void*)) << ")))[index] = (T) variable;\n";
                setNewAgentVariableImpl << "              return;\n";
                setNewAgentVariableImpl << "          }\n";
            } else { ++ct; }
        }
        setNewAgentVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        setNewAgentVariableImpl <<         "          DTHROW(\"New agent variable '%s' was not found during setVariable().\\n\", name);\n";
        setNewAgentVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETNEWAGENTVARIABLE_IMPL", setNewAgentVariableImpl.str());
    }
    // generate setAgentArrayVariable func implementation ($DYNAMIC_SETAGENTARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setAgentArrayVariableImpl;
        if (!agent_variables.empty())
            setAgentArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setAgentArrayVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                setAgentArrayVariableImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                setAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setAgentArrayVariableImpl << "                  return;\n";
                setAgentArrayVariableImpl << "              } else if (N != " << element.second.elements << ") {\n";
                setAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setAgentArrayVariableImpl << "                  return;\n";
                setAgentArrayVariableImpl << "              } else if (array_index >= " << element.second.elements << ") {\n";
                setAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during setVariable().\\n\", name, array_index);\n";
                setAgentArrayVariableImpl << "                  return;\n";
                setAgentArrayVariableImpl << "              }\n";
                setAgentArrayVariableImpl << "#endif\n";
                setAgentArrayVariableImpl << "              (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[i] = (T) variable;\n";
                setAgentArrayVariableImpl << "              return;\n";
                setAgentArrayVariableImpl << "          }\n";
            } else { ++ct; }
        }
        setAgentArrayVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        setAgentArrayVariableImpl <<         "          DTHROW(\"Agent array variable '%s' was not found during setVariable().\\n\", name);\n";
        setAgentArrayVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETAGENTARRAYVARIABLE_IMPL", setAgentArrayVariableImpl.str());
    }
    // generate setNewAgentArrayVariable func implementation ($DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setNewAgentArrayVariableImpl;
        if (!newAgent_variables.empty())
            setNewAgentArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (const auto &element : newAgent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setNewAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setNewAgentArrayVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                setNewAgentArrayVariableImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                setNewAgentArrayVariableImpl << "                  DTHROW(\"New agent array variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setNewAgentArrayVariableImpl << "                  return;\n";
                setNewAgentArrayVariableImpl << "              } else if (N != " << element.second.elements << ") {\n";
                setNewAgentArrayVariableImpl << "                  DTHROW(\"New agent array variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setNewAgentArrayVariableImpl << "                  return;\n";
                setNewAgentArrayVariableImpl << "              } else if (array_index >= " << element.second.elements << ") {\n";
                setNewAgentArrayVariableImpl << "                  DTHROW(\"New agent array variable '%s', index %d is out of bounds during setVariable().\\n\", name, array_index);\n";
                setNewAgentArrayVariableImpl << "                  return;\n";
                setNewAgentArrayVariableImpl << "              }\n";
                setNewAgentArrayVariableImpl << "#endif\n";
                setNewAgentArrayVariableImpl << "              (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << newAgent_data_offset + (ct++ * sizeof(void*)) << ")))[i] = (T) variable;\n";
                setNewAgentArrayVariableImpl << "              return;\n";
                setNewAgentArrayVariableImpl << "          }\n";
            } else { ++ct; }
        }
        setNewAgentArrayVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        setNewAgentArrayVariableImpl <<         "          DTHROW(\"New agent array variable '%s' was not found during setVariable().\\n\", name);\n";
        setNewAgentArrayVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL", setNewAgentArrayVariableImpl.str());
    }
}
void CurveRTCHost::initHeaderGetters() {
    // generate getAgentVariable func implementation ($DYNAMIC_GETAGENTVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentVariableImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getAgentVariableImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                getAgentVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getAgentVariableImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentVariableImpl << "                    return 0;\n";
                getAgentVariableImpl << "                }\n";
                getAgentVariableImpl << "#endif\n";
                getAgentVariableImpl << "                return (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[index];\n";
                getAgentVariableImpl << "            }\n";
            } else { ++ct; }
        }
        getAgentVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        getAgentVariableImpl <<         "            DTHROW(\"Agent variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentVariableImpl <<         "#endif\n";
        getAgentVariableImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTVARIABLE_IMPL", getAgentVariableImpl.str());
    }
    // generate getMessageVariable func implementation ($DYNAMIC_GETMESSAGEVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMessageVariableImpl;
        for (const auto &element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getMessageVariableImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                getMessageVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getMessageVariableImpl << "                    DTHROW(\"Message variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageVariableImpl << "                    return 0;\n";
                getMessageVariableImpl << "                }\n";
                getMessageVariableImpl << "#endif\n";
                getMessageVariableImpl << "                return (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << msgIn_data_offset + (ct++ * sizeof(void*)) << ")))[index];\n";
                getMessageVariableImpl << "            }\n";
            } else { ++ct; }
        }
        getMessageVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        getMessageVariableImpl <<         "            DTHROW(\"Message variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageVariableImpl <<         "#endif\n";
        getMessageVariableImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEVARIABLE_IMPL", getMessageVariableImpl.str());
    }
    // generate getAgentVariable func implementation ($DYNAMIC_GETAGENTVARIABLE_LDG_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentVariableLDGImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getAgentVariableLDGImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentVariableLDGImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                getAgentVariableLDGImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getAgentVariableLDGImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentVariableLDGImpl << "                    return 0;\n";
                getAgentVariableLDGImpl << "                }\n";
                getAgentVariableLDGImpl << "#endif\n";
                getAgentVariableLDGImpl << "                return (T) __ldg((*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << "))) + index);\n";
                getAgentVariableLDGImpl << "            }\n";
            } else { ++ct; }
        }
        getAgentVariableLDGImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        getAgentVariableLDGImpl <<         "            DTHROW(\"Agent variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentVariableLDGImpl <<         "#endif\n";
        getAgentVariableLDGImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTVARIABLE_LDG_IMPL", getAgentVariableLDGImpl.str());
    }
    // generate getMessageVariable func implementation ($DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMessageVariableLDGImpl;
        for (const auto &element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getMessageVariableLDGImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageVariableLDGImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                getMessageVariableLDGImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getMessageVariableLDGImpl << "                    DTHROW(\"Message variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageVariableLDGImpl << "                    return 0;\n";
                getMessageVariableLDGImpl << "                }\n";
                getMessageVariableLDGImpl << "#endif\n";
                getMessageVariableLDGImpl << "                return (T) __ldg((*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << msgIn_data_offset + (ct++ * sizeof(void*)) << "))) + index);\n";
                getMessageVariableLDGImpl << "            }\n";
            } else { ++ct; }
        }
        getMessageVariableLDGImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        getMessageVariableLDGImpl <<         "            DTHROW(\"Message variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageVariableLDGImpl <<         "#endif\n";
        getMessageVariableLDGImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL", getMessageVariableLDGImpl.str());
    }
    // generate getArrayVariable func implementation ($DYNAMIC_GETAGENTARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentArrayVariableImpl;
        if (!agent_variables.empty())
            getAgentArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentArrayVariableImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                getAgentArrayVariableImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                getAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableImpl << "                  return 0;\n";
                getAgentArrayVariableImpl << "              } else if (N != " << element.second.elements << ") {\n";
                getAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableImpl << "                  return 0;\n";
                getAgentArrayVariableImpl << "              } else if (array_index >= " << element.second.elements << ") {\n";
                getAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during getVariable().\\n\", name, array_index);\n";
                getAgentArrayVariableImpl << "                  return 0;\n";
                getAgentArrayVariableImpl << "              }\n";
                getAgentArrayVariableImpl << "#endif\n";
                getAgentArrayVariableImpl << "              return (*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[i];\n";
                getAgentArrayVariableImpl << "           };\n";
            } else { ++ct; }
        }
        getAgentArrayVariableImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        getAgentArrayVariableImpl <<         "           DTHROW(\"Agent array variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentArrayVariableImpl <<         "#endif\n";
        getAgentArrayVariableImpl <<         "           return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTARRAYVARIABLE_IMPL", getAgentArrayVariableImpl.str());
    }
    // generate getArrayVariable func implementation ($DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentArrayVariableLDGImpl;
        if (!agent_variables.empty())
            getAgentArrayVariableLDGImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (std::pair<std::string, RTCVariableProperties> element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getAgentArrayVariableLDGImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentArrayVariableLDGImpl << "#if !defined(SEATBELTS) || SEATBELTS\n";
                getAgentArrayVariableLDGImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                getAgentArrayVariableLDGImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableLDGImpl << "                  return 0;\n";
                getAgentArrayVariableLDGImpl << "              } else if (N != " << element.second.elements << ") {\n";
                getAgentArrayVariableLDGImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableLDGImpl << "                  return 0;\n";
                getAgentArrayVariableLDGImpl << "              } else if (array_index >= " << element.second.elements << ") {\n";
                getAgentArrayVariableLDGImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during getVariable().\\n\", name, array_index);\n";
                getAgentArrayVariableLDGImpl << "                  return 0;\n";
                getAgentArrayVariableLDGImpl << "              }\n";
                getAgentArrayVariableLDGImpl << "#endif\n";
                getAgentArrayVariableLDGImpl << "              return (T) __ldg((*static_cast<T**>(static_cast<void*>(" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << "))) + i);\n";
                getAgentArrayVariableLDGImpl << "           };\n";
            } else { ++ct; }
        }
        getAgentArrayVariableLDGImpl <<         "#if !defined(SEATBELTS) || SEATBELTS\n";
        getAgentArrayVariableLDGImpl <<         "           DTHROW(\"Agent array variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentArrayVariableLDGImpl <<         "#endif\n";
        getAgentArrayVariableLDGImpl <<         "           return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL", getAgentArrayVariableLDGImpl.str());
    }
}
void CurveRTCHost::initDataBuffer() {
    if (data_buffer_size == 0 || h_data_buffer) {
        THROW exception::InvalidOperation("CurveRTCHost::initDataBuffer() should only be called once, during the init chain.\n");
    }
    // Alloc buffer
    h_data_buffer = static_cast<char*>(malloc(data_buffer_size));
    // Notify all variables of their ptr to store data in cache
    size_t ct = 0;
    for (auto &element : agent_variables) {
        element.second.h_data_ptr = h_data_buffer + agent_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto &element : messageOut_variables) {
        element.second.h_data_ptr = h_data_buffer + msgOut_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto &element : messageIn_variables) {
        element.second.h_data_ptr = h_data_buffer + msgIn_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto &element : newAgent_variables) {
        element.second.h_data_ptr = h_data_buffer + newAgent_data_offset + (ct++ * sizeof(void*));
    }
}

std::string CurveRTCHost::getDynamicHeader() {
    initHeaderEnvironment();
    initHeaderSetters();
    initHeaderGetters();
    initDataBuffer();
    return header;
}

void CurveRTCHost::setHeaderPlaceholder(std::string placeholder, std::string dst) {
    // replace placeholder with dynamically generated variables string
    size_t pos = header.find(placeholder);
    if (pos != std::string::npos) {
        header.replace(pos, placeholder.length(), dst);
    } else {
        THROW exception::UnknownInternalError("String (%s) not found when creating dynamic version of curve for RTC: in CurveRTCHost::setHeaderPlaceholder", placeholder.c_str());
    }
}

std::string CurveRTCHost::getVariableSymbolName() {
    std::stringstream name;
    name << "rtc_env_data_curve";
    return name.str();
}

std::string CurveRTCHost::demangle(const char* verbose_name) {
#ifndef _MSC_VER
    std::string s = jitify::reflection::detail::demangle_cuda_symbol(verbose_name);
#else
    // Jitify removed the required demangle function, this is a basic clone of what was being done in earlier version
    // It's possible jitify::reflection::detail::demangle_native_type() would work, however that requires type_info, not type_index
    size_t index = 0;
    std::string s = verbose_name;
    while (true) {
        /* Locate the substring to replace. */
        index = s.find("class", index);
        if (index == std::string::npos) break;

        /* Make the replacement. */
        s.replace(index, 5, "     ");

        /* Advance index forward so the next iteration doesn't pick it up as well. */
        index += 5;
    }
#endif
    // Lambda function for trimming whitesapce as jitify demangle does not remove this
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
        }));
#ifdef _MSC_VER
    // int64_t is the only known problematic type in windows as it has a typeid().name() of __int64.
    // This can be manually replaced
    std::string int64_type = "__int64";
    std::string int64_type_fixed = "long long int";
    size_t start_pos = s.find(int64_type);
    if (!(start_pos == std::string::npos))
        s.replace(start_pos, int64_type.length(), int64_type_fixed);
#endif

    // map known basic types in
    return s;
}

std::string CurveRTCHost::demangle(const std::type_index& type) {
    return demangle(type.name());
}
void CurveRTCHost::updateEnvCache(const char *env_ptr) {
    if (env_ptr) {
        memcpy(h_data_buffer, env_ptr, EnvironmentManager::MAX_BUFFER_SIZE);
    }
}
void CurveRTCHost::updateDevice(const jitify::experimental::KernelInstantiation& instance) {
    // The namespace is required here, but not in other uses of getVariableSymbolName.
    std::string cache_var_name = std::string("flamegpu::") + getVariableSymbolName();
    CUdeviceptr d_var_ptr = instance.get_global_ptr(cache_var_name.c_str());
    gpuErrchkDriverAPI(cuMemcpyHtoD(d_var_ptr, h_data_buffer, data_buffer_size));
}

}  // namespace flamegpu

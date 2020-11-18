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


const char* CurveRTCHost::curve_rtc_dynamic_h_template = R"###(dynamic/curve_rtc_dynamic.h
#ifndef CURVE_RTC_DYNAMIC_H_
#define CURVE_RTC_DYNAMIC_H_

#include "flamegpu/exception/FGPUDeviceException.h"

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
*/

$DYNAMIC_VARIABLES

$DYNAMIC_ENV_VARIABLES

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

// has to be included after definition of curve namespace
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"

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

#endif  // CURVE_RTC_DYNAMIC_H_
)###";


CurveRTCHost::CurveRTCHost() : header(CurveRTCHost::curve_rtc_dynamic_h_template) {
}


void CurveRTCHost::registerAgentVariable(const char* variableName, unsigned int namespace_hash, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    if (agent_namespace == 0)
        agent_namespace = namespace_hash;
    if (namespace_hash != agent_namespace) {
        THROW UnknownInternalError("A different namespace hash (%d) is already registered to the one provided (%d): in CurveRTCHost::registerAgentVariable", agent_namespace, namespace_hash);
    }
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    agent_variables.emplace(variableName, props);
}
void CurveRTCHost::registerMessageInVariable(const char* variableName, unsigned int namespace_hash, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    if (messageIn_namespace == 0)
        messageIn_namespace = namespace_hash;
    if (namespace_hash != messageIn_namespace) {
        THROW UnknownInternalError("A different namespace hash (%d) is already registered to the one provided (%d): in CurveRTCHost::registerMessageInVariable", messageIn_namespace, namespace_hash);
    }
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    messageIn_variables.emplace(variableName, props);
}
void CurveRTCHost::registerMessageOutVariable(const char* variableName, unsigned int namespace_hash, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    if (messageOut_namespace == 0)
        messageOut_namespace = namespace_hash;
    if (namespace_hash != messageOut_namespace) {
        THROW UnknownInternalError("A different namespace hash (%d) is already registered to the one provided (%d): in CurveRTCHost::registerMessageOutVariable", messageOut_namespace, namespace_hash);
    }
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    messageOut_variables.emplace(variableName, props);
}
void CurveRTCHost::registerNewAgentVariable(const char* variableName, unsigned int namespace_hash, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    if (newAgent_namespace == 0)
        newAgent_namespace = namespace_hash;
    if (namespace_hash != newAgent_namespace) {
        THROW UnknownInternalError("A different namespace hash (%d) is already registered to the one provided (%d): in CurveRTCHost::registerNewAgentVariable", newAgent_namespace, namespace_hash);
    }
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    newAgent_variables.emplace(variableName, props);
}

void CurveRTCHost::unregisterAgentVariable(const char* variableName, unsigned int namespace_hash) {
    if (namespace_hash != agent_namespace) {
        THROW UnknownInternalError("Namespace hash (%d) not found when removing variable: in CurveRTCHost::unregisterAgentVariable", namespace_hash);
    }
    auto i = agent_variables.find(variableName);
    if (i != agent_variables.end()) {
        agent_variables.erase(variableName);
    } else {
        THROW UnknownInternalError("Variable '%s' not found when removing variable: in CurveRTCHost::unregisterAgentVariable", variableName);
    }
    // Clear namespace on last var
    if (agent_variables.empty())
        agent_namespace = 0;
}
void CurveRTCHost::unregisterMessageOutVariable(const char* variableName, unsigned int namespace_hash) {
    if (namespace_hash != messageOut_namespace) {
        THROW UnknownInternalError("Namespace hash (%d) not found when removing variable: in CurveRTCHost::unregisterMessageOutVariable", namespace_hash);
    }
    auto i = messageOut_variables.find(variableName);
    if (i != messageOut_variables.end()) {
        messageOut_variables.erase(variableName);
    } else {
        THROW UnknownInternalError("Variable '%s' not found when removing variable: in CurveRTCHost::unregisterMessageOutVariable", variableName);
    }
    // Clear namespace on last var
    if (messageOut_variables.empty())
        messageOut_namespace = 0;
}
void CurveRTCHost::unregisterMessageInVariable(const char* variableName, unsigned int namespace_hash) {
    if (namespace_hash != messageIn_namespace) {
        THROW UnknownInternalError("Namespace hash (%d) not found when removing variable: in CurveRTCHost::unregisterMessageInVariable", namespace_hash);
    }
    auto i = messageIn_variables.find(variableName);
    if (i != messageIn_variables.end()) {
        messageIn_variables.erase(variableName);
    } else {
        THROW UnknownInternalError("Variable '%s' not found when removing variable: in CurveRTCHost::unregisterMessageInVariable", variableName);
    }
    // Clear namespace on last var
    if (messageIn_variables.empty())
        messageIn_namespace = 0;
}
void CurveRTCHost::unregisterNewAgentVariable(const char* variableName, unsigned int namespace_hash) {
    if (namespace_hash != newAgent_namespace) {
        THROW UnknownInternalError("Namespace hash (%d) not found when removing variable: in CurveRTCHost::unregisterNewAgentVariable", namespace_hash);
    }
    auto i = newAgent_variables.find(variableName);
    if (i != newAgent_variables.end()) {
        newAgent_variables.erase(variableName);
    } else {
        THROW UnknownInternalError("Variable '%s' not found when removing variable: in CurveRTCHost::unregisterNewAgentVariable", variableName);
    }
    // Clear namespace on last var
    if (newAgent_variables.empty())
        newAgent_namespace = 0;
}

void CurveRTCHost::registerEnvVariable(const char* variableName, unsigned int namespace_hash, ptrdiff_t offset, const char* type, size_t type_size, unsigned int elements) {
    // check to see if namespace key already exists
    auto i = RTCEnvVariables.find(namespace_hash);
    RTCEnvVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.elements = elements;
    props.offset = offset;
    props.type_size = type_size;
    if (i != RTCEnvVariables.end()) {
        // emplace into existing namespace key
        i->second.emplace(variableName, props);
    } else {
        std::unordered_map<std::string, RTCEnvVariableProperties> inner;
        inner.emplace(variableName, props);
        RTCEnvVariables.emplace(namespace_hash, inner);
    }
}

void CurveRTCHost::unregisterEnvVariable(const char* variableName, unsigned int namespace_hash) {
    auto i = RTCEnvVariables.find(namespace_hash);
    if (i != RTCEnvVariables.end()) {
        i->second.erase(variableName);
    } else {
        THROW UnknownInternalError("Namespace hash (%d) not found when removing environment variable: in CurveRTCHost::unregisterEnvVariable", namespace_hash);
    }
}


void CurveRTCHost::initHeaderEnvironment() {
    // generate dynamic variables ($DYNAMIC_VARIABLES)
    std::stringstream variables;
    for (const auto &element : agent_variables) {
        variables << "__device__ " << element.second.type << "* " << "curve_rtc_ptr_" << agent_namespace << "_" << element.first << ";\n";
    }
    for (const auto &element : messageIn_variables) {
        variables << "__device__ " << element.second.type << "* " << "curve_rtc_ptr_" << messageIn_namespace << "_" << element.first << ";\n";
    }
    for (const auto &element : messageOut_variables) {
        variables << "__device__ " << element.second.type << "* " << "curve_rtc_ptr_" << messageOut_namespace << "_" << element.first << ";\n";
    }
    for (const auto &element : newAgent_variables) {
        variables << "__device__ " << element.second.type << "* " << "curve_rtc_ptr_" << newAgent_namespace << "_" << element.first << ";\n";
    }
    setHeaderPlaceholder("$DYNAMIC_VARIABLES", variables.str());

    // generate dynamic environment variables ($DYNAMIC_ENV_VARIABLES)
    std::stringstream envVariables;
    envVariables << "__constant__  char " << getEnvVariableSymbolName() <<"[" << EnvironmentManager::MAX_BUFFER_SIZE << "];\n";
    setHeaderPlaceholder("$DYNAMIC_ENV_VARIABLES", envVariables.str());
    // generate Environment::get func implementation ($DYNAMIC_ENV_GETVARIABLE_IMPL)
    {
        std::stringstream getEnvVariableImpl;
        getEnvVariableImpl <<               "    switch(modelname_hash){\n";
        for (auto key_pair : RTCEnvVariables) {
            unsigned int namespace_hash = key_pair.first;
            getEnvVariableImpl <<           "      case(" << namespace_hash << "):\n";
            for (std::pair<std::string, RTCEnvVariableProperties> element : key_pair.second) {
                RTCEnvVariableProperties props = element.second;
                if (props.elements == 1) {
                    getEnvVariableImpl <<   "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                    getEnvVariableImpl <<   "#ifndef NO_SEATBELTS\n";
                    getEnvVariableImpl <<   "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                    getEnvVariableImpl <<   "                    DTHROW(\"Environment property '%s' type mismatch.\\n\", name);\n";
                    getEnvVariableImpl <<   "                    return 0;\n";
                    getEnvVariableImpl <<   "                }\n";
                    getEnvVariableImpl <<   "#endif\n";
                    getEnvVariableImpl <<   "                return *reinterpret_cast<T*>(reinterpret_cast<void*>(" << getEnvVariableSymbolName() <<" + " << props.offset << "));\n";
                    getEnvVariableImpl <<   "            };\n";
                }
            }
            getEnvVariableImpl <<           "#ifndef NO_SEATBELTS\n";
            getEnvVariableImpl <<           "            DTHROW(\"Environment property '%s' was not found.\\n\", name);\n";
            getEnvVariableImpl <<           "#endif\n";
            getEnvVariableImpl <<           "            return 0;\n";
        }
        getEnvVariableImpl <<               "      default:\n";
        getEnvVariableImpl <<               "#ifndef NO_SEATBELTS\n";
        getEnvVariableImpl <<               "          DTHROW(\"Unexpected modelname hash %d for environment property '%s'.\\n\", modelname_hash, name);\n";
        getEnvVariableImpl <<               "#endif\n";
        getEnvVariableImpl <<               "          return 0;\n";
        getEnvVariableImpl <<               "    }\n";
        getEnvVariableImpl <<               "    return 0;\n";    // if namespace is not recognised
        setHeaderPlaceholder("$DYNAMIC_ENV_GETVARIABLE_IMPL", getEnvVariableImpl.str());
    }
    // generate Environment::get func implementation for array variables ($DYNAMIC_ENV_GETARRAYVARIABLE_IMPL)
    {
        std::stringstream getEnvArrayVariableImpl;
        getEnvArrayVariableImpl <<             "    switch(modelname_hash){\n";
        for (auto key_pair : RTCEnvVariables) {
            unsigned int namespace_hash = key_pair.first;
            getEnvArrayVariableImpl <<         "      case(" << namespace_hash << "):\n";
            for (std::pair<std::string, RTCEnvVariableProperties> element : key_pair.second) {
                RTCEnvVariableProperties props = element.second;
                if (props.elements > 1) {
                    getEnvArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                    getEnvArrayVariableImpl << "#ifndef NO_SEATBELTS\n";
                    getEnvArrayVariableImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                    getEnvArrayVariableImpl << "                  DTHROW(\"Environment array property '%s' type mismatch.\\n\", name);\n";
                    getEnvArrayVariableImpl << "                  return 0;\n";
                    // Env var doesn't currently require user to specify length
                    // getEnvArrayVariableImpl << "              } else if (N != " << element.second.elements << ") {\n";
                    // getEnvArrayVariableImpl << "                  DTHROW(\"Environment array property '%s' length mismatch.\\n\", name);\n";
                    // getEnvArrayVariableImpl << "                  return 0;\n";
                    getEnvArrayVariableImpl << "              } else if (index >= " << element.second.elements << ") {\n";
                    getEnvArrayVariableImpl << "                  DTHROW(\"Environment array property '%s', index %d is out of bounds.\\n\", name, index);\n";
                    getEnvArrayVariableImpl << "                  return 0;\n";
                    getEnvArrayVariableImpl << "              }\n";
                    getEnvArrayVariableImpl << "#endif\n";
                    getEnvArrayVariableImpl << "              return reinterpret_cast<T*>(reinterpret_cast<void*>(" << getEnvVariableSymbolName() <<" + " << props.offset << "))[index];\n";
                    getEnvArrayVariableImpl << "          };\n";
                }
            }
            getEnvArrayVariableImpl <<         "#ifndef NO_SEATBELTS\n";
            getEnvArrayVariableImpl <<         "          DTHROW(\"Environment array property '%s' was not found.\\n\", name);\n";
            getEnvArrayVariableImpl <<         "#endif\n";
            getEnvArrayVariableImpl <<         "          return 0;\n";
        }
        getEnvArrayVariableImpl <<             "      default:\n";
        getEnvArrayVariableImpl <<             "#ifndef NO_SEATBELTS\n";
        getEnvArrayVariableImpl <<             "          DTHROW(\"Unexpected modelname hash %d for environment array property '%s'.\\n\", modelname_hash, name);\n";
        getEnvArrayVariableImpl <<             "#endif\n";
        getEnvArrayVariableImpl <<             "          return 0;\n";
        getEnvArrayVariableImpl <<             "    }\n";
        getEnvArrayVariableImpl <<             "    return 0;\n";   // if namespace is not recognised
        setHeaderPlaceholder("$DYNAMIC_ENV_GETARRAYVARIABLE_IMPL", getEnvArrayVariableImpl.str());
    }
    // generate Environment::contains func implementation ($DYNAMIC_ENV_CONTAINTS_IMPL)
    {
        std::stringstream containsEnvVariableImpl;
        containsEnvVariableImpl <<               "    switch(modelname_hash){\n";
        for (auto key_pair : RTCEnvVariables) {
            unsigned int namespace_hash = key_pair.first;
            containsEnvVariableImpl <<           "      case(" << namespace_hash << "):\n";
            unsigned int count = 0;
            for (std::pair<std::string, RTCEnvVariableProperties> element : key_pair.second) {
                RTCEnvVariableProperties props = element.second;
                if (props.elements == 1) {
                    containsEnvVariableImpl <<   "            if (strings_equal(name, \"" << element.first << "\"))\n";
                    containsEnvVariableImpl <<   "                return true;\n";
                    count++;
                }
            }
            if (count > 0) {
                containsEnvVariableImpl <<       "            else\n";
            }
            containsEnvVariableImpl <<           "                return false;\n";
        }
        containsEnvVariableImpl <<               "      default:\n";
        containsEnvVariableImpl <<               "          return false;\n";
        containsEnvVariableImpl <<               "    }\n";
        containsEnvVariableImpl <<               "    return false;\n";    // if namespace is not recognised
        setHeaderPlaceholder("$DYNAMIC_ENV_CONTAINTS_IMPL", containsEnvVariableImpl.str());
    }
}
void CurveRTCHost::initHeaderSetters() {
    // generate setAgentVariable func implementation ($DYNAMIC_SETAGENTVARIABLE_IMPL)
    {
        std::stringstream setAgentVariableImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements == 1) {
                setAgentVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setAgentVariableImpl << "#ifndef NO_SEATBELTS\n";
                setAgentVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                setAgentVariableImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setAgentVariableImpl << "                    return;\n";
                setAgentVariableImpl << "                }\n";
                setAgentVariableImpl << "#endif\n";
                setAgentVariableImpl << "              curve_rtc_ptr_" << agent_namespace << "_" << element.first << "[index] = (T) variable;\n";
                setAgentVariableImpl << "              return;\n";
                setAgentVariableImpl << "          }\n";
            }
        }
        setAgentVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        setAgentVariableImpl <<         "          DTHROW(\"Agent variable '%s' was not found during setVariable().\\n\", name);\n";
        setAgentVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETAGENTVARIABLE_IMPL", setAgentVariableImpl.str());
    }
    // generate setMessageVariable func implementation ($DYNAMIC_SETMESSAGEVARIABLE_IMPL)
    {
        std::stringstream setMessageVariableImpl;
        for (const auto &element : messageOut_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements == 1) {
                setMessageVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setMessageVariableImpl << "#ifndef NO_SEATBELTS\n";
                setMessageVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                setMessageVariableImpl << "                    DTHROW(\"Message variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setMessageVariableImpl << "                    return;\n";
                setMessageVariableImpl << "                }\n";
                setMessageVariableImpl << "#endif\n";
                setMessageVariableImpl << "              curve_rtc_ptr_" << messageOut_namespace << "_" << element.first << "[index] = (T) variable;\n";
                setMessageVariableImpl << "              return;\n";
                setMessageVariableImpl << "          }\n";
            }
        }
        setMessageVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        setMessageVariableImpl <<         "          DTHROW(\"Message variable '%s' was not found during setVariable().\\n\", name);\n";
        setMessageVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETMESSAGEVARIABLE_IMPL", setMessageVariableImpl.str());
    }
    // generate setNewAgentVariable func implementation ($DYNAMIC_SETNEWAGENTVARIABLE_IMPL)
    {
        std::stringstream setNewAgentVariableImpl;
        for (const auto &element : newAgent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements == 1) {
                setNewAgentVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setNewAgentVariableImpl << "#ifndef NO_SEATBELTS\n";
                setNewAgentVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                setNewAgentVariableImpl << "                    DTHROW(\"New agent variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setNewAgentVariableImpl << "                    return;\n";
                setNewAgentVariableImpl << "                }\n";
                setNewAgentVariableImpl << "#endif\n";
                setNewAgentVariableImpl << "              curve_rtc_ptr_" << newAgent_namespace << "_" << element.first << "[index] = (T) variable;\n";
                setNewAgentVariableImpl << "              return;\n";
                setNewAgentVariableImpl << "          }\n";
            }
        }
        setNewAgentVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        setNewAgentVariableImpl <<         "          DTHROW(\"New agent variable '%s' was not found during setVariable().\\n\", name);\n";
        setNewAgentVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETNEWAGENTVARIABLE_IMPL", setNewAgentVariableImpl.str());
    }
    // generate setAgentArrayVariable func implementation ($DYNAMIC_SETAGENTARRAYVARIABLE_IMPL)
    {
        std::stringstream setAgentArrayVariableImpl;
        if (!agent_variables.empty())
            setAgentArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setAgentArrayVariableImpl << "#ifndef NO_SEATBELTS\n";
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
                setAgentArrayVariableImpl << "              curve_rtc_ptr_" << agent_namespace << "_" << element.first << "[i] = (T) variable;\n";
                setAgentArrayVariableImpl << "              return;\n";
                setAgentArrayVariableImpl << "          }\n";
            }
        }
        setAgentArrayVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        setAgentArrayVariableImpl <<         "          DTHROW(\"Agent array variable '%s' was not found during setVariable().\\n\", name);\n";
        setAgentArrayVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETAGENTARRAYVARIABLE_IMPL", setAgentArrayVariableImpl.str());
    }
    // generate setNewAgentArrayVariable func implementation ($DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL)
    {
        std::stringstream setNewAgentArrayVariableImpl;
        if (!newAgent_variables.empty())
            setNewAgentArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (const auto &element : newAgent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setNewAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setNewAgentArrayVariableImpl << "#ifndef NO_SEATBELTS\n";
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
                setNewAgentArrayVariableImpl << "              curve_rtc_ptr_" << newAgent_namespace << "_" << element.first << "[i] = (T) variable;\n";
                setNewAgentArrayVariableImpl << "              return;\n";
                setNewAgentArrayVariableImpl << "          }\n";
            }
        }
        setNewAgentArrayVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        setNewAgentArrayVariableImpl <<         "          DTHROW(\"New agent array variable '%s' was not found during setVariable().\\n\", name);\n";
        setNewAgentArrayVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL", setNewAgentArrayVariableImpl.str());
    }
}
void CurveRTCHost::initHeaderGetters() {
    // generate getAgentVariable func implementation ($DYNAMIC_GETAGENTVARIABLE_IMPL)
    {
        std::stringstream getAgentVariableImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getAgentVariableImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentVariableImpl << "#ifndef NO_SEATBELTS\n";
                getAgentVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getAgentVariableImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentVariableImpl << "                    return 0;\n";
                getAgentVariableImpl << "                }\n";
                getAgentVariableImpl << "#endif\n";
                getAgentVariableImpl << "                return (T) " << "curve_rtc_ptr_" << agent_namespace << "_" << element.first << "[index];\n";
                getAgentVariableImpl << "            }\n";
            }
        }
        getAgentVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        getAgentVariableImpl <<         "            DTHROW(\"Agent variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentVariableImpl <<         "#endif\n";
        getAgentVariableImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTVARIABLE_IMPL", getAgentVariableImpl.str());
    }
    // generate getMessageVariable func implementation ($DYNAMIC_GETMESSAGEVARIABLE_IMPL)
    {
        std::stringstream getMessageVariableImpl;
        for (const auto &element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getMessageVariableImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageVariableImpl << "#ifndef NO_SEATBELTS\n";
                getMessageVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getMessageVariableImpl << "                    DTHROW(\"Message variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageVariableImpl << "                    return 0;\n";
                getMessageVariableImpl << "                }\n";
                getMessageVariableImpl << "#endif\n";
                getMessageVariableImpl << "                return (T) " << "curve_rtc_ptr_" << messageIn_namespace << "_" << element.first << "[index];\n";
                getMessageVariableImpl << "            }\n";
            }
        }
        getMessageVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        getMessageVariableImpl <<         "            DTHROW(\"Message variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageVariableImpl <<         "#endif\n";
        getMessageVariableImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEVARIABLE_IMPL", getMessageVariableImpl.str());
    }
    // generate getAgentVariable func implementation ($DYNAMIC_GETAGENTVARIABLE_LDG_IMPL)
    {
        std::stringstream getAgentVariableLDGImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getAgentVariableLDGImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentVariableLDGImpl << "#ifndef NO_SEATBELTS\n";
                getAgentVariableLDGImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getAgentVariableLDGImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentVariableLDGImpl << "                    return 0;\n";
                getAgentVariableLDGImpl << "                }\n";
                getAgentVariableLDGImpl << "#endif\n";
                getAgentVariableLDGImpl << "                return (T) " << "__ldg(&curve_rtc_ptr_" << agent_namespace << "_" << element.first << "[index]);\n";
                getAgentVariableLDGImpl << "            }\n";
            }
        }
        getAgentVariableLDGImpl <<         "#ifndef NO_SEATBELTS\n";
        getAgentVariableLDGImpl <<         "            DTHROW(\"Agent variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentVariableLDGImpl <<         "#endif\n";
        getAgentVariableLDGImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTVARIABLE_LDG_IMPL", getAgentVariableLDGImpl.str());
    }
    // generate getMessageVariable func implementation ($DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL)
    {
        std::stringstream getMessageVariableLDGImpl;
        for (const auto &element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getMessageVariableLDGImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageVariableLDGImpl << "#ifndef NO_SEATBELTS\n";
                getMessageVariableLDGImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getMessageVariableLDGImpl << "                    DTHROW(\"Message variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageVariableLDGImpl << "                    return 0;\n";
                getMessageVariableLDGImpl << "                }\n";
                getMessageVariableLDGImpl << "#endif\n";
                getMessageVariableLDGImpl << "                return (T) " << "curve_rtc_ptr_" << messageIn_namespace << "_" << element.first << "[index];\n";
                getMessageVariableLDGImpl << "            }\n";
            }
        }
        getMessageVariableLDGImpl <<         "#ifndef NO_SEATBELTS\n";
        getMessageVariableLDGImpl <<         "            DTHROW(\"Message variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageVariableLDGImpl <<         "#endif\n";
        getMessageVariableLDGImpl <<         "            return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL", getMessageVariableLDGImpl.str());
    }
    // generate getArrayVariable func implementation ($DYNAMIC_GETAGENTARRAYVARIABLE_IMPL)
    {
        std::stringstream getAgentArrayVariableImpl;
        if (!agent_variables.empty())
            getAgentArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentArrayVariableImpl << "#ifndef NO_SEATBELTS\n";
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
                getAgentArrayVariableImpl << "              return (T) " << "curve_rtc_ptr_" << agent_namespace << "_" << element.first << "[i];\n";
                getAgentArrayVariableImpl << "           };\n";
            }
        }
        getAgentArrayVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        getAgentArrayVariableImpl <<         "           DTHROW(\"Agent array variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentArrayVariableImpl <<         "#endif\n";
        getAgentArrayVariableImpl <<         "           return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTARRAYVARIABLE_IMPL", getAgentArrayVariableImpl.str());
    }
    // generate getArrayVariable func implementation ($DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL)
    {
        std::stringstream getAgentArrayVariableLDGImpl;
        if (!agent_variables.empty())
            getAgentArrayVariableLDGImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (std::pair<std::string, RTCVariableProperties> element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getAgentArrayVariableLDGImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentArrayVariableLDGImpl << "#ifndef NO_SEATBELTS\n";
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
                getAgentArrayVariableLDGImpl << "              return (T) " << "__ldg(&curve_rtc_ptr_" << agent_namespace << "_" << element.first << "[i]);\n";
                getAgentArrayVariableLDGImpl << "           };\n";
            }
        }
        getAgentArrayVariableLDGImpl <<         "#ifndef NO_SEATBELTS\n";
        getAgentArrayVariableLDGImpl <<         "           DTHROW(\"Agent array variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentArrayVariableLDGImpl <<         "#endif\n";
        getAgentArrayVariableLDGImpl <<         "           return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL", getAgentArrayVariableLDGImpl.str());
    }
}


std::string CurveRTCHost::getDynamicHeader() {
    initHeaderEnvironment();
    initHeaderSetters();
    initHeaderGetters();
    return header;
}

void CurveRTCHost::setHeaderPlaceholder(std::string placeholder, std::string dst) {
    // replace placeholder with dynamically generated variables string
    size_t pos = header.find(placeholder);
    if (pos != std::string::npos) {
        header.replace(pos, placeholder.length(), dst);
    } else {
        THROW UnknownInternalError("String (%s) not found when creating dynamic version of curve for RTC: in CurveRTCHost::setHeaderPlaceholder", placeholder.c_str());
    }
}


std::string CurveRTCHost::getVariableSymbolName(const char* variableName, unsigned int namespace_hash) {
    std::stringstream name;
    name << "curve_rtc_ptr_" << namespace_hash << "_" << variableName;
    return name.str();
}

std::string CurveRTCHost::getEnvVariableSymbolName() {
    std::stringstream name;
    name << "curve_env_rtc_ptr";
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

#include <sstream>

#include "flamegpu/runtime/cuRVE/curve_rtc.h"
#include "flamegpu/exception/FGPUException.h"


const char* CurveRTCHost::curve_rtc_dynamic_h_template = R"###(dynamic/curve_rtc_dynamic.h
#ifndef CURVE_RTC_DYNAMIC_H_
#define CURVE_RTC_DYNAMIC_H_


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

$DYNAMIC_AGENT_VARIBALES

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
    __device__ __forceinline__ static T getVariable(const char(&name)[N], VariableHash namespace_hash, unsigned int index);

    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getArrayVariable(const char(&name)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    

    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index);

    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setArrayVariable(const char(&name)[M], VariableHash namespace_hash, T variable, unsigned int variable_index, unsigned int array_index);

};

template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getVariable(const char (&name)[N], VariableHash namespace_hash, unsigned int index) {
$DYNAMIC_GETVARIABLE_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T Curve::getArrayVariable(const char(&name)[M], VariableHash namespace_hash, unsigned int index, unsigned int array_index) {
$DYNAMIC_GETARRAYVARIABLE_IMPL
}

template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index) {
$DYNAMIC_SETVARIABLE_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void Curve::setArrayVariable(const char(&name)[M], VariableHash namespace_hash, T variable, unsigned int index, unsigned int array_index) {
$DYNAMIC_SETARRAYVARIABLE_IMPL    
}

#endif  // CURVE_RTC_DYNAMIC_H_
)###";


CurveRTCHost::CurveRTCHost() : header(CurveRTCHost::curve_rtc_dynamic_h_template) {
}


void CurveRTCHost::registerVariable(const char* variableName, unsigned int namespace_hash, const char* type) {
    // check to see if namespace key already exists
    auto i = RTCVariables.find(namespace_hash);
    if (i != RTCVariables.end()) {
        // emplace into existing namespace key
        i->second.emplace(variableName, type);
    } else {
        std::unordered_map<std::string, std::string> inner;
        inner.emplace(variableName, type);
        RTCVariables.emplace(namespace_hash, inner);
    }
}


void CurveRTCHost::unregisterVariable(const char* variableName, unsigned int namespace_hash) {
    auto i = RTCVariables.find(namespace_hash);
    if (i != RTCVariables.end()) {
        i->second.erase(variableName);
    } else {
        THROW UnknownInternalError("Namespace hash (%d) not found when removing variable: in CurveRTCHost::unregisterVariable", namespace_hash);
    }
}

std::string CurveRTCHost::getDynamicHeader() {
    // generate dynamic variables ($DYNAMIC_AGENT_VARIBALES)
    std::stringstream variables;
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        for (std::pair<std::string, std::string> element : key_pair.second) {
            variables << "__device__ " << element.second << "* " << "curve_rtc_ptr_" << namespace_hash << "_" << element.first << ";\n";
        }
    }
    setHeaderPlaceholder("$DYNAMIC_AGENT_VARIBALES", variables.str());

    // generate getVariable func implementation ($DYNAMIC_GETVARIABLE_IMPL)
    std::stringstream getVariableImpl;
    getVariableImpl <<         "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        getVariableImpl <<     "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, std::string> element : key_pair.second) {
            getVariableImpl << "            if (strings_equal(name, \"" << element.first << "\"))\n";
            getVariableImpl << "                return (T) " << "curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[index];\n";
        }
        getVariableImpl <<     "            else\n";
        getVariableImpl <<     "                return 0;\n";
    }
    getVariableImpl <<         "      default:\n";
    getVariableImpl <<         "          return 0;\n";
    getVariableImpl <<         "    }\n";

    setHeaderPlaceholder("$DYNAMIC_GETVARIABLE_IMPL", getVariableImpl.str());

    // generate setVariable func implementation ($DYNAMIC_SETTVARIABLE_IMPL)
    std::stringstream setVariableImpl;
    setVariableImpl <<         "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        setVariableImpl <<     "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, std::string> element : key_pair.second) {
            setVariableImpl << "          if (strings_equal(name, \"" << element.first << "\"))\n";
            setVariableImpl << "              curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[index] = (T) variable;\n";
        }
    }
    setVariableImpl <<         "      default:\n";
    setVariableImpl <<         "          return;\n";
    setVariableImpl <<         "    }\n";
    setHeaderPlaceholder("$DYNAMIC_SETVARIABLE_IMPL", setVariableImpl.str());

    // generate getArrayVariable func implementation ($DYNAMIC_GETARRAYVARIABLE_IMPL)
    std::stringstream getArrayVariableImpl;
    getArrayVariableImpl <<         "    const size_t i = (index * N) + array_index;\n";
    getArrayVariableImpl <<         "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        getArrayVariableImpl <<     "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, std::string> element : key_pair.second) {
            getArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\"))\n";
            getArrayVariableImpl << "              return (T) " << "curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[i];\n";
        }
    }
    getArrayVariableImpl <<         "          else\n";
    getArrayVariableImpl <<         "              return 0;\n";
    getArrayVariableImpl <<         "      default:\n";
    getArrayVariableImpl <<         "          return 0;\n";
    getArrayVariableImpl <<         "    }\n";
    setHeaderPlaceholder("$DYNAMIC_GETARRAYVARIABLE_IMPL", getArrayVariableImpl.str());

    // generate setArrayVariable func implementation ($DYNAMIC_SETARRAYVARIABLE_IMPL)
    std::stringstream setArrayVariableImpl;
    setArrayVariableImpl <<         "    const size_t i = (index * N) + array_index;\n";
    setArrayVariableImpl <<         "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        setArrayVariableImpl <<     "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, std::string> element : key_pair.second) {
            setArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\"))\n";
            setArrayVariableImpl << "              curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[i] = (T) variable;\n";
        }
    }
    setArrayVariableImpl <<         "      default:\n";
    setArrayVariableImpl <<         "          return;\n";
    setArrayVariableImpl <<         "    }\n";
    setHeaderPlaceholder("$DYNAMIC_SETARRAYVARIABLE_IMPL", setArrayVariableImpl.str());

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

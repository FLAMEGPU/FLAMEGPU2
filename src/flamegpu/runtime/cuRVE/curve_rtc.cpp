#include <sstream>

#include "flamegpu/runtime/cuRVE/curve_rtc.h"


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

    return 0;
}

template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setVariable(const char(&name)[N], VariableHash namespace_hash, T variable, unsigned int index) {
    
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void Curve::setArrayVariable(const char(&name)[M], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index) {
    
}

#endif  // CURVE_RTC_DYNAMIC_H_
)###";


CurveRTCHost::CurveRTCHost() : header(CurveRTCHost::curve_rtc_dynamic_h_template) {
}


void CurveRTCHost::registerVariable(const char* variableName, const char* type) {
    RTCVariables.emplace(variableName, type);
}


void CurveRTCHost::unregisterVariable(const char* variableName) {
    RTCVariables.erase(variableName);
}

std::string CurveRTCHost::getDynamicHeader() {
    // generate dynamic variables ($DYNAMIC_AGENT_VARIBALES)
    std::stringstream variables;
    for (std::pair<std::string, std::string> element : RTCVariables) {
        variables << "__device__ " << element.second << "* " << "curve_rtc_ptr_" << element.first << ";\n";
    }
    setHeaderPlaceholder("$DYNAMIC_AGENT_VARIBALES", variables.str());

    // generate getVariable func implementation ($DYNAMIC_GETVARIABLE_IMPL)
    std::stringstream getVariableImpl;
    for (std::pair<std::string, std::string> element : RTCVariables) {
        getVariableImpl << "    if (strings_equal(name, \"" << element.first << "\"))\n";
        getVariableImpl << "        return (T) " << "curve_rtc_ptr_" << element.first << "[index];\n";
    }
    getVariableImpl << "    else\n";
    getVariableImpl << "        return 0;\n";
    setHeaderPlaceholder("$DYNAMIC_GETVARIABLE_IMPL", getVariableImpl.str());

    return header;
}

void CurveRTCHost::setHeaderPlaceholder(std::string placeholder, std::string dst) {
    // replace placeholder with dynamically generated variables string
    size_t pos = header.find(placeholder);
    header.replace(pos, placeholder.length(), dst);
}

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

// has to be included after definition of curve namespace
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"

template<typename T, unsigned int N>
__device__ __forceinline__ T DeviceEnvironment::get(const char(&name)[N]) const {
$DYNAMIC_ENV_GETVARIABLE_IMPL
}

template<typename T, unsigned int N>
__device__ __forceinline__ T DeviceEnvironment::get(const char(&name)[N], const unsigned int &index) const {
$DYNAMIC_ENV_GETARRAYVARIABLE_IMPL
}

template<unsigned int N>
__device__ __forceinline__ bool DeviceEnvironment::contains(const char(&name)[N]) const {
$DYNAMIC_ENV_CONTAINTS_IMPL
}

#endif  // CURVE_RTC_DYNAMIC_H_
)###";


CurveRTCHost::CurveRTCHost() : header(CurveRTCHost::curve_rtc_dynamic_h_template) {
}


void CurveRTCHost::registerVariable(const char* variableName, unsigned int namespace_hash, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    // check to see if namespace key already exists
    auto i = RTCVariables.find(namespace_hash);
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    if (i != RTCVariables.end()) {
        // emplace into existing namespace key
        i->second.emplace(variableName, props);
    } else {
        std::unordered_map<std::string, RTCVariableProperties> inner;
        inner.emplace(variableName, props);
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

std::string CurveRTCHost::getDynamicHeader() {
    // generate dynamic variables ($DYNAMIC_VARIABLES)
    std::stringstream variables;
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        for (std::pair<std::string, RTCVariableProperties> element : key_pair.second) {
            RTCVariableProperties props = element.second;
            variables << "__device__ " << props.type << "* " << "curve_rtc_ptr_" << namespace_hash << "_" << element.first << ";\n";
        }
    }
    setHeaderPlaceholder("$DYNAMIC_VARIABLES", variables.str());

    // generate dynamic environment variables ($DYNAMIC_ENV_VARIABLES)
    std::stringstream envVariables;
    envVariables << "__constant__  char " << getEnvVariableSymbolName() <<"[" << EnvironmentManager::MAX_BUFFER_SIZE << "];\n";
    setHeaderPlaceholder("$DYNAMIC_ENV_VARIABLES", envVariables.str());

    // generate getVariable func implementation ($DYNAMIC_GETVARIABLE_IMPL)
    std::stringstream getVariableImpl;
    getVariableImpl <<             "    // Switch over agent\n";
    getVariableImpl <<             "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        getVariableImpl <<         "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, RTCVariableProperties> element : key_pair.second) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {
                getVariableImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getVariableImpl << "#ifndef NO_SEATBELTS\n";
                getVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getVariableImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getVariableImpl << "                    return 0;\n";
                getVariableImpl << "                }\n";
                getVariableImpl << "#endif\n";
                getVariableImpl << "                return (T) " << "curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[index];\n";
                getVariableImpl << "            }\n";
            }
        }
        getVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        getVariableImpl <<         "            DTHROW(\"Agent variable '%s' was not found during getVariable().\\n\", name);\n";
        getVariableImpl <<         "#endif\n";
        getVariableImpl <<         "            return 0;\n";
    }
    getVariableImpl <<             "      default:{\n";
    getVariableImpl <<             "#ifndef NO_SEATBELTS\n";
    getVariableImpl <<             "          DTHROW(\"Unexpected namespace hash %d for agent variable '%s' during getVariable().\\n\", namespace_hash, name);\n";
    getVariableImpl <<             "#endif\n";
    getVariableImpl <<             "          return 0;\n";
    getVariableImpl <<             "      }\n";
    getVariableImpl <<             "    }\n";
    getVariableImpl <<             "    return 0;\n";    // if namespace is not recognised
    setHeaderPlaceholder("$DYNAMIC_GETVARIABLE_IMPL", getVariableImpl.str());

    // generate setVariable func implementation ($DYNAMIC_SETTVARIABLE_IMPL)
    std::stringstream setVariableImpl;
    setVariableImpl <<             "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        setVariableImpl <<         "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, RTCVariableProperties> element : key_pair.second) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements == 1) {
                setVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setVariableImpl << "#ifndef NO_SEATBELTS\n";
                setVariableImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                setVariableImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setVariableImpl << "                    break;\n";
                setVariableImpl << "                }\n";
                setVariableImpl << "#endif\n";
                setVariableImpl << "              curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[index] = (T) variable;\n";
                setVariableImpl << "              break;\n";
                setVariableImpl << "          }\n";
            }
        }
        setVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        setVariableImpl <<         "          DTHROW(\"Agent variable '%s' was not found during setVariable().\\n\", name);\n";
        setVariableImpl <<         "#endif\n";
        setVariableImpl <<         "          break;\n";
    }
    setVariableImpl <<             "      default:\n";
    setVariableImpl <<             "#ifndef NO_SEATBELTS\n";
    setVariableImpl <<             "          DTHROW(\"Unexpected namespace hash %d for agent variable '%s' during setVariable().\\n\", namespace_hash, name);\n";
    setVariableImpl <<             "#endif\n";
    setVariableImpl <<             "          return;\n";
    setVariableImpl <<             "    }\n";
    setHeaderPlaceholder("$DYNAMIC_SETVARIABLE_IMPL", setVariableImpl.str());

    // generate getArrayVariable func implementation ($DYNAMIC_GETARRAYVARIABLE_IMPL)
    std::stringstream getArrayVariableImpl;
    getArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
    getArrayVariableImpl <<             "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        getArrayVariableImpl <<         "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, RTCVariableProperties> element : key_pair.second) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getArrayVariableImpl << "#ifndef NO_SEATBELTS\n";
                getArrayVariableImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                getArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getArrayVariableImpl << "                  return 0;\n";
                getArrayVariableImpl << "              } else if (N != " << element.second.elements << ") {\n";
                getArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getArrayVariableImpl << "                  return 0;\n";
                getArrayVariableImpl << "              } else if (array_index >= " << element.second.elements << "|| array_index < 0) {\n";
                getArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during getVariable().\\n\", name, array_index);\n";
                getArrayVariableImpl << "                  return 0;\n";
                getArrayVariableImpl << "              }\n";
                getArrayVariableImpl << "#endif\n";
                getArrayVariableImpl << "              return (T) " << "curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[i];\n";
                getArrayVariableImpl << "           };\n";
            }
        }
        getArrayVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        getArrayVariableImpl <<         "           DTHROW(\"Agent array variable '%s' was not found during getVariable().\\n\", name);\n";
        getArrayVariableImpl <<         "#endif\n";
        getArrayVariableImpl <<         "           return 0;\n";
    }
    getArrayVariableImpl <<             "      default:\n";
    getArrayVariableImpl <<             "#ifndef NO_SEATBELTS\n";
    getArrayVariableImpl <<             "          DTHROW(\"Unexpected namespace hash %d for agent array variable '%s' during getVariable().\\n\", namespace_hash, name);\n";
    getArrayVariableImpl <<             "#endif\n";
    getArrayVariableImpl <<             "          return 0;\n";
    getArrayVariableImpl <<             "    }\n";
    getArrayVariableImpl <<             "    return 0;\n";   // if namespace is not recognised
    setHeaderPlaceholder("$DYNAMIC_GETARRAYVARIABLE_IMPL", getArrayVariableImpl.str());

    // generate setArrayVariable func implementation ($DYNAMIC_SETARRAYVARIABLE_IMPL)
    std::stringstream setArrayVariableImpl;
    setArrayVariableImpl <<             "    const size_t i = (index * N) + array_index;\n";
    setArrayVariableImpl <<             "    switch(namespace_hash){\n";
    for (auto key_pair : RTCVariables) {
        unsigned int namespace_hash = key_pair.first;
        setArrayVariableImpl <<         "      case(" << namespace_hash << "):\n";
        for (std::pair<std::string, RTCVariableProperties> element : key_pair.second) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setArrayVariableImpl << "#ifndef NO_SEATBELTS\n";
                setArrayVariableImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                setArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setArrayVariableImpl << "                  break;\n";
                setArrayVariableImpl << "              } else if (N != " << element.second.elements << ") {\n";
                setArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setArrayVariableImpl << "                  break;\n";
                setArrayVariableImpl << "              } else if (array_index >= " << element.second.elements << "|| array_index < 0) {\n";
                setArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during setVariable().\\n\", name, array_index);\n";
                setArrayVariableImpl << "                  break;\n";
                setArrayVariableImpl << "              }\n";
                setArrayVariableImpl << "#endif\n";
                setArrayVariableImpl << "              curve_rtc_ptr_" << namespace_hash << "_" << element.first << "[i] = (T) variable;\n";
                setArrayVariableImpl << "              break;\n";
                setArrayVariableImpl << "          }\n";
            }
        }
        setArrayVariableImpl <<         "#ifndef NO_SEATBELTS\n";
        setArrayVariableImpl <<         "          DTHROW(\"Agent array variable '%s' was not found during setVariable().\\n\", name);\n";
        setArrayVariableImpl <<         "#endif\n";
        setArrayVariableImpl <<         "          break;\n";
    }
    setArrayVariableImpl <<             "      default:\n";
    setArrayVariableImpl <<             "#ifndef NO_SEATBELTS\n";
    setArrayVariableImpl <<             "          DTHROW(\"Unexpected namespace hash %d for agent array variable '%s' during setVariable().\\n\", namespace_hash, name);\n";
    setArrayVariableImpl <<             "#endif\n";
    setArrayVariableImpl <<             "          return;\n";
    setArrayVariableImpl <<             "    }\n";
    setHeaderPlaceholder("$DYNAMIC_SETARRAYVARIABLE_IMPL", setArrayVariableImpl.str());

    // generate Environment::get func implementation ($DYNAMIC_ENV_GETVARIABLE_IMPL)
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

    // generate Environment::get func implementation for array variables ($DYNAMIC_ENV_GETARRAYVARIABLE_IMPL)
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
                getEnvArrayVariableImpl << "              } else if (index >= " << element.second.elements << "|| index < 0) {\n";
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

    // generate Environment::contains func implementation ($DYNAMIC_ENV_CONTAINTS_IMPL)
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

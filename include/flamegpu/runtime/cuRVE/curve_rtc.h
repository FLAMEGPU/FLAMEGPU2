#ifndef INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_RTC_H_
#define INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_RTC_H_

#include <cstring>
#include <string>
#include <cstdio>
#include <typeindex>
#include <unordered_map>


/** @brief    A cuRVE instance.
 *
 * The Curve RTC host is a class for dynamically building a header file for use in RTC functions. Rather than providing a hashmap of string variable names it will dynamically create a header with agent variables directly accessible via compile time string comparisons.
 */
class CurveRTCHost {
 public:
    CurveRTCHost();

    void registerVariable(const char* variableName, unsigned int namespace_hash, const char* type, unsigned int elements = 1, bool read = true, bool write = true);

    void unregisterVariable(const char* variableName, unsigned int namespace_hash);

    void registerEnvVariable(const char* variableName, unsigned int namespace_hash, const char* type, unsigned int elements = 1);

    void unregisterEnvVariable(const char* variableName, unsigned int namespace_hash);

    std::string getDynamicHeader();

    static std::string getVariableSymbolName(const char* variableName, unsigned int namespace_hash);

    static std::string getEnvVariableSymbolName(const char* variableName, unsigned int namespace_hash);

    /**
     * Demangle a verbose type name (e.g. std::type_index.name().c_str()) into a user readable type
     * This is required as different compilers will perform name mangling in different way (or not at all).
     */
    static std::string demangle(const char* verbose_name);

    /**
     * Demangle  from a std::type_index into a user readable type
     * This is required as different compilers will perform name mangling in different way (or not at all).
     */
    static std::string demangle(const std::type_index& type);

 protected:
    void setHeaderPlaceholder(std::string placeholder, std::string dst);

    struct RTCVariableProperties {
        std::string type;
        bool read;
        bool write;
        unsigned int elements;
    };

    struct RTCEnvVariableProperties {
        std::string type;
        unsigned int elements;
    };

 private:
    std::string header;
    static const char* curve_rtc_dynamic_h_template;
    std::unordered_map<unsigned int, std::unordered_map<std::string, RTCVariableProperties>> RTCVariables;     // <namespace, <name, RTCVariableProperties>>
    std::unordered_map<unsigned int, std::unordered_map<std::string, RTCEnvVariableProperties>> RTCEnvVariables;     // <namespace, <name, RTCEnvVariableProperties>>
};
#endif  // INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_RTC_H_

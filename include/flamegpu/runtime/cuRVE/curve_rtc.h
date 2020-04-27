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

    std::string getDynamicHeader();

 protected:
    void setHeaderPlaceholder(std::string placeholder, std::string dst);

    typedef struct {
        std::string type;
        bool read;
        bool write;
        unsigned int elements;
    } RTCVariableProperties;

 private:
    std::string header;
    static const char* curve_rtc_dynamic_h_template;
    std::unordered_map<unsigned int, std::unordered_map<std::string, RTCVariableProperties>> RTCVariables;     // <namespace, <name, RTCVariableProperties>>
};
#endif  // INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_RTC_H_

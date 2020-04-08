#ifndef INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_RTC_H_
#define INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_RTC_H_

#include <cstring>
#include <string>
#include <cstdio>
#include <typeindex>
#include <unordered_map>


/** @brief    A cuRVE instance.
 *
 * The Curve RTC host is a class for dynamically building a header file for use in RTC functions. Rather than providing a hashmap of string variable names it will dynamically create a header with agent varibales directly accessible via compile time string comparisons.
 */
class CurveRTCHost {
 public:
    CurveRTCHost();

    void registerVariable(const char* variableName, const char* type);

    void unregisterVariable(const char* variableName);

    std::string getDynamicHeader();

 protected:
    void setHeaderPlaceholder(std::string placeholder, std::string dst);

 private:
    std::string header;
    static const char* curve_rtc_dynamic_h_template;
    std::unordered_map<std::string, std::string> RTCVariables;
};
#endif  // INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_RTC_H_

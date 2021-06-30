#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_CXXNAME_HPP_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_CXXNAME_HPP_

#include <string>

namespace flamegpu {
namespace util {
namespace detail {
namespace cxxname {

/**
 * Get the unqualified name from a qualified (or unqualified) name. 
 * This is a very simple approach, which essentially just returns the final component of a qualified name, so should not be relied upon and is only provided to provided more user-friendly error messages.
 * @param qualified a (un)qualified class name
 * @return the unqualified class name
 */ 
inline std::string getUnqualifiedName(std::string qualified) {
    const char * SCOPE_RESOLUTION_OPERATIOR = "::";
    size_t lastOccurence = qualified.find_last_of(SCOPE_RESOLUTION_OPERATIOR);
    if (lastOccurence == std::string::npos) {
        return qualified;
    } else {
        return qualified.substr(lastOccurence + 1);
    }
}

}  // namespace cxxname
}  // namespace detail
}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_CXXNAME_HPP_

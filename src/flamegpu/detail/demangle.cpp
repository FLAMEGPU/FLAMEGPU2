#include "flamegpu/detail/demangle.h"

#include <algorithm>
#include <memory>
#include <string>
#include <typeindex>
#ifndef _MSC_VER
// abi::__cxa_demangle()
#include <cxxabi.h>
#endif

#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {
namespace detail {
namespace demangle {

std::string demangle(const char* verbose_name) {
#ifndef _MSC_VER
    // Implementation from Jitify1
    size_t bufsize = 0;
    char* buf = nullptr;
    std::string s;
    int status;
    auto demangled_ptr = std::unique_ptr<char, decltype(free)*>(
        abi::__cxa_demangle(verbose_name, buf, &bufsize, &status), free);
    if (status == 0) {
        s = demangled_ptr.get();  // all worked as expected
    } else if (status == -2) {
        s = verbose_name;  // we interpret this as plain C name
    } else if (status == -1) {
        THROW exception::UnknownInternalError("memory allocation failure in __cxa_demangle");
    } else if (status == -3) {
        THROW exception::UnknownInternalError("invalid argument to __cxa_demangle");
    }
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

std::string demangle(const std::type_index& type) {
    return demangle(type.name());
}

}  // namespace demangle
}  // namespace detail
}  // namespace flamegpu

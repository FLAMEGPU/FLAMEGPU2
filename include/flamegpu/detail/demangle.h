#ifndef INCLUDE_FLAMEGPU_DETAIL_DEMANGLE_H_
#define INCLUDE_FLAMEGPU_DETAIL_DEMANGLE_H_

#include <string>
#include <typeindex>

namespace flamegpu {
namespace detail {
namespace demangle {


/**
 * Demangle a verbose type name (e.g. std::type_index.name().c_str()) into a user readable type
 * This is required as different compilers will perform name mangling in different way (or not at all).
 * @param verbose_name The verbose type name to be demangled
 * @return The demangled type name
 */
std::string demangle(const char* verbose_name);

/**
 * Demangle from a std::type_index into a user readable type
 * This is required as different compilers will perform name mangling in different way (or not at all).
 * @param type The type to return the demangled name for
 * @return The demangled type name of the provided type
 */
std::string demangle(const std::type_index& type);

}  // namespace demangle
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_DEMANGLE_H_

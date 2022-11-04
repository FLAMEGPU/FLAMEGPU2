#ifndef INCLUDE_FLAMEGPU_DEFINES_H_
#define INCLUDE_FLAMEGPU_DEFINES_H_

namespace flamegpu {

// Definitions class, for macros and so on.
/**
 * Type used for generic identifiers, primarily used for Agent ids
 */
typedef unsigned int id_t;
/**
 * Internal variable name used for IDs
 */
constexpr const char* ID_VARIABLE_NAME = "_id";
/**
 * Internal value used when IDs have not be set
 * If this value is changed, things may break
 */
constexpr id_t ID_NOT_SET = 0;
/**
* Typedef for verbosity level of the API
*/
enum class Verbosity {Quiet, Default, Verbose};

typedef unsigned int size_type;

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DEFINES_H_

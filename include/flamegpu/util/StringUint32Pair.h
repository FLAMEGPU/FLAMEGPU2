#ifndef INCLUDE_FLAMEGPU_UTIL_STRINGUINT32PAIR_H_
#define INCLUDE_FLAMEGPU_UTIL_STRINGUINT32PAIR_H_

#include <string>
#include <utility>
#include <map>
#include <unordered_map>

namespace flamegpu {
namespace util {
/**
 * Pair of String and unsigned integer.
 */
typedef std::pair<std::string, unsigned int> StringUint32Pair;

/**
 * Hash function so that StringUint32Pair can be used as a key in a map
 */
struct StringUint32PairHash {
    std::size_t operator()(const std::pair<std::string, unsigned int>& k) const noexcept {
        return ((std::hash<std::string>()(k.first) ^ (std::hash<unsigned int>()(k.second) << 1)) >> 1);
    }
};

/**
 * Ordered map with StringUint32Pair as the key type
 */
template<typename T>
using StringUint32PairMap = std::map<StringUint32Pair, T, StringUint32PairHash>;

/**
 * Unordered map with StringUint32Pair as the key type
 */
template<typename T>
using StringUint32PairUnorderedMap = std::unordered_map<StringUint32Pair, T, StringUint32PairHash>;

}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_STRINGUINT32PAIR_H_

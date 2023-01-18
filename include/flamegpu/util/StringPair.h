#ifndef INCLUDE_FLAMEGPU_UTIL_STRINGPAIR_H_
#define INCLUDE_FLAMEGPU_UTIL_STRINGPAIR_H_

#include <string>
#include <utility>
#include <map>
#include <unordered_map>

namespace flamegpu {
namespace util {
/**
 * Pair of strings
 */
typedef std::pair<std::string, std::string> StringPair;

/**
 * Compare function so that StringPair can be used in a map
 */
template<typename T1, typename T2>
struct PairCompare {
    bool operator() (const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const {
        if (lhs.first == rhs.first)
            return lhs.second < rhs.second;
        return lhs.first < rhs.first;
    }
};
typedef PairCompare<std::string, std::string> StringPairCompare;
/**
 * Hash function so that StringPair can be used as a key in an unordered map
 */
template<typename T1, typename T2>
struct PairHash {
    size_t operator()(const std::pair<T1, T2>& k) const {
        return std::hash<T1>()(k.first) ^
            (std::hash<T2>()(k.second) << 1);
    }
};
typedef PairHash<std::string, std::string> StringPairHash;

/**
 * Ordered map with StringPair as the key type
 */
template<typename A, typename B, typename T>
using PairMap = std::map<std::pair<A, B>, T, PairCompare<A, B>>;
template<typename T>
using StringPairMap = PairMap<std::string, std::string, T>;

/**
 * Unordered map with StringPair as the key type
 */
template<typename A, typename B, typename T>
using PairUnorderedMap = std::unordered_map<std::pair<A, B>, T, PairHash<A, B>>;
template<typename T>
using StringPairUnorderedMap = PairUnorderedMap<std::string, std::string, T>;

}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_STRINGPAIR_H_

#ifndef INCLUDE_FLAMEGPU_UTIL_STRINGPAIR_H_
#define INCLUDE_FLAMEGPU_UTIL_STRINGPAIR_H_

#include <string>
#include <utility>
#include <map>
#include <unordered_map>

typedef std::pair<std::string, std::string> StringPair;

struct StringPairHash {
    size_t operator()(const std::pair<std::string, std::string>& k) const {
        return std::hash<std::string>()(k.first) ^
            (std::hash<std::string>()(k.second) << 1);
    }
};

template<typename T>
using StringPairMap = std::map<StringPair, T, StringPairHash>;

template<typename T>
using StringPairUnorderedMap = std::unordered_map<StringPair, T, StringPairHash>;

#endif  // INCLUDE_FLAMEGPU_UTIL_STRINGPAIR_H_

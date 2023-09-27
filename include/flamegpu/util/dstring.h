#ifndef INCLUDE_FLAMEGPU_UTIL_DSTRING_H_
#define INCLUDE_FLAMEGPU_UTIL_DSTRING_H_

#include <cuda_runtime.h>

namespace flamegpu {
namespace util {
/**
 * Device implementations of required string.h functionality
 */


/**
 * strcmp() - Compare two strings, return 0 if equal, otherwise return suggests order
 *
 * @param s1 First string to be compared
 * @param s2 Second string to be compared
 *
 * @note Implementation based on https://stackoverflow.com/a/34873763/1646387
 */
__device__ __forceinline__ int dstrcmp(const char *s1, const char *s2) {
    const unsigned char *p1 = (const unsigned char *)s1;
    const unsigned char *p2 = (const unsigned char *)s2;

    while (*p1 && *p1 == *p2) ++p1, ++p2;

    return (*p1 > *p2) - (*p2  > *p1);
}

}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DSTRING_H_

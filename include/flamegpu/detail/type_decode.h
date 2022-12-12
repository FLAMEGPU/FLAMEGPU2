#ifndef INCLUDE_FLAMEGPU_DETAIL_TYPE_DECODE_H_
#define INCLUDE_FLAMEGPU_DETAIL_TYPE_DECODE_H_

#if defined(FLAMEGPU_USE_GLM) || defined(GLM_VERSION)
#ifndef GLM_VERSION
#ifdef __CUDACC__
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#else
#pragma diag_suppress = esa_on_defaulted_function_ignored
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#endif  // __CUDACC__
#include <glm/glm.hpp>
#endif
#endif

namespace flamegpu {
namespace detail {
/**
 * This struct allows us to natively decode GLM types to their type + length
 */
template <typename T>
struct type_decode {
    // Length of the decoded type
    static constexpr unsigned int len_t = 1;
    // Type of the decoded type
    typedef T type_t;
};

#if defined(FLAMEGPU_USE_GLM) || defined(GLM_VERSION)
/**
 * GLM specialisation, only enabled if GLM is present
 */
template <int N, typename T, glm::qualifier Q>
struct type_decode<glm::vec<N, T, Q>> {
    static constexpr unsigned int len_t = N;
    typedef T type_t;
};
#endif

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_TYPE_DECODE_H_

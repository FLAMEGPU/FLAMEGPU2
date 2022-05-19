#ifndef INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_CUH_

#ifndef __CUDACC_RTC__
#include <string>
#endif
/**
 * The main cuRVE header file for the CUDA Runtime Variable Environment (cuRVE)
 * Based off the following article http:// www.gamasutra.com/view/news/127915/InDepth_Quasi_CompileTime_String_Hashing.php
 * This file contains definitions common to HostCurve and DeviceCurve
 */
namespace flamegpu {
namespace detail {
namespace curve {

class Curve {
 public:
    typedef int                      Variable;           // !< Typedef for cuRVE variable handle
    typedef unsigned int             VariableHash;       // !< Typedef for cuRVE variable name string hash
    typedef unsigned int             NamespaceHash;      // !< Typedef for cuRVE variable namespace string hash
    static const int MAX_VARIABLES = 512;                // !< Default maximum number of cuRVE variables (must be a power of 2)
    static const VariableHash EMPTY_FLAG = 0;
    /**
     * Main cuRVE variable hashing function
     *
     * Calls recursive hashing functions, this should be processed at compile time (effectively constexpr).
     * @param str String to be hashed
     * @return a 32 bit cuRVE string variable hash.
     */
    template <unsigned int N>
    __device__ __host__ __forceinline__ static VariableHash variableHash(const char(&str)[N]);
#ifndef __CUDACC_RTC__
    /**
     * Main cuRVE variable hashing function for strings of length determined at runtime and not compile time
     * Should only be used for registered variables as this will be much slower than the compile time alternative.
     * @param str String to be hashed
     * @return a 32 bit cuRVE string variable hash.
     */
    __host__ static VariableHash variableRuntimeHash(const std::string &str);
    /**
     * Main cuRVE variable hashing function for unsigned integers
     * @param num Unsigned int to be hashed
     * @return a 32 bit cuRVE string variable hash.
     */
    __host__ static VariableHash variableRuntimeHash(unsigned int num);
#endif
};
/**
 * Representation of Curve's hashtable
 */
struct CurveTable {
    Curve::VariableHash hashes[Curve::MAX_VARIABLES];            // Device array of the hash values of registered variables
    char* variables[Curve::MAX_VARIABLES];                // Device array of pointer to device memory addresses for variable storage
    unsigned int type_size[Curve::MAX_VARIABLES];         // Device array of the types of registered variables
    unsigned int elements[Curve::MAX_VARIABLES];
    unsigned int count[Curve::MAX_VARIABLES];
};

/* TEMPLATE HASHING FUNCTIONS */

/** @brief Non terminal template structure has function for a constant char array
 *     Use of template meta-programming ensures the compiler can evaluate string hashes at compile time. This reduces constant string variable names to a single 32 bit value. Hashing is based on 'Quasi Compile Time String Hashing' at http:// www.altdevblogaday.com/2011/10/27/quasi-compile-time-string-hashing/
 *     Code uses compilation flags for both the host and the CUDA device.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N, unsigned int I> struct CurveStringHash {
    __device__ __host__ inline static Curve::VariableHash Hash(const char(&str)[N]) {
        return (CurveStringHash<N, I - 1>::Hash(str) ^ str[I - 1]) * 16777619u;
    }
};
/** @brief Terminal template structure has function for a constant char array
 *     Function within a template structure allows partial template specialisation for terminal case.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N> struct CurveStringHash<N, 1> {
    __device__ __host__ inline static Curve::VariableHash Hash(const char(&str)[N]) {
        return (2166136261u ^ str[0]) * 16777619u;
    }
};

template <unsigned int N>
__device__ __host__ __forceinline__ Curve::VariableHash Curve::variableHash(const char(&str)[N]) {
    return CurveStringHash<N, N>::Hash(str);
}


}  // namespace curve
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_DETAIL_CURVE_CURVE_CUH_

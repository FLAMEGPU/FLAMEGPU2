#ifndef HASHING_H_
#define HASHING_H_

#include <string.h>

#include "cuda_runtime.h"

#define EMPTY_HASH_VALUE 0x00

/* TEMPLATE HASHING FUNCTIONS */

/** @brief Non terminal template structure has function for a constant char array
 * 	Use of template meta-programming ensures the compiler can evaluate string hashes at compile time. This reduces constant string variable names to a single 32 bit value. Hashing is based on 'Quasi Compile Time String Hashing' at http://www.altdevblogaday.com/2011/10/27/quasi-compile-time-string-hashing/
 * 	Code uses compilation flags for both the host and the CUDA device.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N, unsigned int I> struct StringHash
{
	__device__ __host__ inline static unsigned int Hash(const char (&str)[N])
	{
		return (StringHash<N, I-1>::Hash(str) ^ str[I-1])*16777619u;
	}
};
/** @brief Terminal template structure has function for a constant char array
 * 	Function within a template structure allows partial template specialisation for terminal case.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N> struct StringHash<N, 1>
{
	__device__ __host__ inline static unsigned int Hash(const char (&str)[N])
	{
	    return (2166136261u ^ str[0])*16777619u;
	}
};
/** @brief Main cuRVE variable hashing function
 *  Calls recursive hashing functions
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N> __device__ __host__ inline static unsigned int VariableHash(const char (&str)[N])
{
	return StringHash<N, N>::Hash(str);
}


/**
 * Non compile time hashing function
 */

unsigned int VariableHash(const char* str){

	const size_t length = strlen(str) + 1;
	unsigned int hash = 2166136261u;
	for (size_t i=0; i<length; ++i){
		hash ^= *str++;
		hash *= 16777619u;
	}
	return hash;
}
#endif

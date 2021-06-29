#ifndef INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_H_

/**
 * @file   curve.h
 * @author Paul Richmond (p.richmond@sheffield.ac.uk) http:// www.paulrichmond.staff.shef.ac.uk/
 * @date   Feb 2017
 * @brief  Main cuRVE header file
 *
 * The main cuRVE header file for the CUDA Runtime Variable Environment (cuRVE)
 * Based off the following article http:// www.gamasutra.com/view/news/127915/InDepth_Quasi_CompileTime_String_Hashing.php
 * \todo Requires vector length table for each variable (or namespace) which is registered. For now no safe checking of vector length is done.
 */

// #include <cuda_runtime.h>

#include <cstring>
#include <cstdio>
#ifndef __CUDACC_RTC__
#include <mutex>
#include <shared_mutex>
#endif

#include "flamegpu/exception/FLAMEGPUDeviceException.h"

namespace flamegpu {

/** @brief    A cuRVE instance.
 *
 * cuRVE is a C library and this singleton class acts as a mechanism to ensure that any reference to the library is handled correctly.
 * For example multiple objects may which to request that curve is initialised. This class will ensure that this function call is only made once the first time that a cuRVEInstance is required.
 */
class Curve {
 public:
    static const int UNKNOWN_VARIABLE = -1;              // !< value returned as a Variable if an API function encounters an error

    typedef int                      Variable;           // !< Typedef for cuRVE variable handle
    typedef unsigned int             VariableHash;       // !< Typedef for cuRVE variable name string hash
    typedef unsigned int             NamespaceHash;      // !< Typedef for cuRVE variable namespace string hash

    /**
     * Main cuRVE variable hashing function for strings of length determined at runtime and not compile time
     * Should only be used for registered variables as this will be much slower than the compile time alternative.
     * @param str String to be hashed
     * @return a 32 bit cuRVE string variable hash.
     */
    __host__ static VariableHash variableRuntimeHash(const char* str);
    /**
     * Main cuRVE variable hashing function for unsigned integers
     * @param num Unsigned int to be hashed
     * @return a 32 bit cuRVE string variable hash.
     */
    __host__ static VariableHash variableRuntimeHash(unsigned int num);

    /**
     * Main cuRVE variable hashing function
     *
     * Calls recursive hashing functions, this should be processed at compile time (effectively constexpr).
     * @param str String to be hashed
     * @return a 32 bit cuRVE string variable hash.
     */
    template <unsigned int N>
    __device__ __host__ __forceinline__ static VariableHash variableHash(const char(&str)[N]);
    /**
     *  Function for getting a handle (hash table index) to a cuRVE variable from a variable string hash
     *
     *  Function performs hash collision avoidance using linear probing.
     *
     *  @param variable_hash A cuRVE variable string hash from variableHash.
     *  @return Variable Handle for the cuRVE variable.
     */
    __host__ Variable getVariableHandle(VariableHash variable_hash);
    /**
     * Function for registering a variable by a VariableHash
     *
     * Registers a variable by insertion in a hash table. 
     * @param variable_hash A cuRVE variable string hash from variableHash.
     * @param d_ptr a pointer to the vector which holds the hashed variable of give name
     * @param size Size of the data type (this should be the size of a single element if an array variable)
     * @param length Number of elements (1 unless the variable is an array)
     * @return Variable Handle of registered variable or UNKNOWN_VARIABLE if an error is encountered.
     * @note It is recommend that you instead use the appropriate registerVariable() template function.
     */
    __host__ Variable registerVariableByHash(VariableHash variable_hash, void* d_ptr, size_t size, unsigned int length);
    /**
     * Template function for registering a constant string
     *
     * Registers a constant string variable name by hashing and then inserting into a hash table.
     * @param variableName A constant char array (C string) variable name.
     * @param d_ptr a pointer to the vector which holds the variable of give name
     * @param length Number of elements (1 unless the variable is an array)
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     * @return Variable Handle of registered variable or UNKNOWN_VARIABLE if an error is encountered.
     * @note This function is nolonger used
     */
    template <unsigned int N, typename T>
    __host__ Variable registerVariable(const char(&variableName)[N], void* d_ptr, unsigned int length);
    /**
     * Check how many items are in the hash table
     *
     * @return The number of items currently stored in the hash table
     */
    __host__ int size() const;
    /**
     * Copy host structures to device
     *
     * This function copies the host hash table to the device, it must be used prior to launching agent functions (and agent function conditions) if Curve has been updated.
     * 4 memcpys to device are always performed, CURVE does not track whether it has been changed internally.
     */
    __host__ void updateDevice();
    /**
     * Function for un-registering a variable by a VariableHash
     *
     *  Un-registers a variable by removing it from the internal hash table.
     *  @note It is recommend to instead use provided unregisterVariable() template function.
     *  @param variable_hash A cuRVE variable string hash from variableHash.
     */
    __host__ void unregisterVariableByHash(VariableHash variable_hash);
    /**
     * Template function for un-registering a constant string
     *
     * Un-registers a constant string variable name by hashing and then removing from the hash table.
     * @param variableName A constant char array (C string) variable name.
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     */
    template <unsigned int N>
    __host__ void unregisterVariable(const char(&variableName)[N]);

    /**
     * Device function for getting the index of a variable of given name within the Curve hashtable buffers
     *
     * Returns the index of the hashed variable within the hash table
     * @param variable_hash A cuRVE variable string hash from variableHash.
     * @return The index of the specified variable within the hash table
     */
    __device__ __forceinline__ static Variable getVariable(const VariableHash variable_hash);
    /**
     * Device function for getting the type size of elements of a variable of given name
     *
     * Gets the size of the cuRVE variable type given the variable hash
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @return A size_t which is the size of the variable or 0 otherwise
     */
    __device__ __forceinline__ static size_t getVariableSize(const VariableHash variable_hash);
    /**
     * Device function for getting the number of elements of a variable of given name
     *
     * Gets the length of the cuRVE variable given the variable hash
     * This will be 1 unless the variable is an array
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @return An unsigned int which is the number of elements within the curve variable (1 unless it's an array)
     */
    __device__ __forceinline__ static unsigned int getVariableLength(const VariableHash variable_hash);
    /**
     * Device function for getting a pointer to a variable of given name
     *
     * Returns a generic pointer to a variable of given name at a specific offset in bytes from the start of the variable array.
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @param offset an offset into the variable array in bytes (offset is variable index * sizeof(variable type))
     * @return A generic pointer to the variable value. Will be nullptr if there is an error.
     * @note Device errors are never thrown from here, instead it is assumed DeviceAPI will detect the error itself so that more detail an be returned to the user
     */
    __device__ __forceinline__ static void* getVariablePtrByHash(const VariableHash variable_hash, size_t offset);
    /**
     * Device function for getting a single typed value from a VariableHash at a given index
     *
     * Returns a single value of specified type from a variableHash using the given index position.
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @return T A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     * @note Device errors are never thrown from here, instead it is assumed DeviceAPI will detect the error itself so that more detail an be returned to the user
     */
    template <typename T>
    __device__ __forceinline__ static T getVariableByHash(const VariableHash variable_hash, unsigned int index);
    /**
     * Device function for getting a single typed value from a VariableHash at a given index via the read-only cache (__ldg())
     *
     * Returns a single value of specified type from a variableHash using the given index position.
     * This uses the __ldg() intrinsic to access the variable via the read-only cache.
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @return T A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     * @note Device errors are never thrown from here, instead it is assumed DeviceAPI will detect the error itself so that more detail an be returned to the user
     * @see getVariableByHash()
     */
    template <typename T>
    __device__ __forceinline__ static T getVariableByHash_ldg(const VariableHash variable_hash, unsigned int index);
    /**
     * Device function for getting an element from an array variable specified by a VariableHash at a given index
     *
     * Returns a single value of specified type from a variableHash using the given index position.
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @param agent_index The index of the variable in the named variable array vector. This corresponds to the agent's index within the agent population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T Type of variable array
     * @tparam N Length of variable array
     * @return T A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     * @note Device errors are never thrown from here, instead it is assumed DeviceAPI will detect the error itself so that more detail an be returned to the user
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getArrayVariableByHash(const VariableHash variable_hash, unsigned int agent_index, unsigned int array_index);
    /**
     * Device function for getting an element from an array variable specified by a VariableHash at a given index via the read-only cache (__ldg())
     *
     * Returns a single value of specified type from a variableHash using the given index position.
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @param agent_index The index of the variable in the named variable array vector. This corresponds to the agent's index within the agent population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T Type of variable array
     * @tparam N Length of variable array
     * @return T A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     * @note Device errors are never thrown from here, instead it is assumed DeviceAPI will detect the error itself so that more detail an be returned to the user
     * @see getArrayVariableByHash()
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getArrayVariableByHash_ldg(const VariableHash variable_hash, unsigned int agent_index, unsigned int array_index);
    /**
     * This forwards directly to Curve::getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index) 
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index);
    /**
     * This forwards directly to Curve::getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index) 
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing message variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index) 
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index);
    /**
     * This forwards directly to Curve::getVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::getVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index)
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index);
    /**
     * This forwards directly to Curve::getVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing message variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::getVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index)
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index);
    /**
     * This forwards directly to Curve::getArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param variable_index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of the array variable specified by variableName
     * @tparam M Length of variableName, this should always be implicit if passing a string literal
     * @see Curve::getArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index)
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    /**
     * This forwards directly to Curve::getArrayVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param variable_index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of the array variable specified by variableName
     * @tparam M Length of variableName, this should always be implicit if passing a string literal
     * @see Curve::getArrayVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index)
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable_ldg(const char(&variableName)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    /**
     * Device function for setting a single typed value from a VariableHash
     *
     * Sets a single value from a variableHash using the given index position.
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @param value The typed value to set at the given index.
     * @param index The index of the variable in the named variable vector.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     */
    template <typename T>
    __device__ __forceinline__ static void setVariableByHash(const VariableHash variable_hash, T value, unsigned int index);
    /**
     * Device function for setting a single typed value from a VariableHash
     *
     * Sets the value of the specified element within the specified agent variable
     * @param variable_hash A cuRVE variable string hash from VariableHash.
     * @param value The typed value to set at the given index.
     * @param agent_index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T Type of variable array
     * @tparam N Length of variable array
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setArrayVariableByHash(const VariableHash variable_hash, T value, unsigned int agent_index, unsigned int array_index);
    /**
     * This forwards directly to Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param variable The value to set the specified variable
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index)
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setAgentVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index);
    /**
     * This forwards directly to Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing message variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param variable The value to set the specified variable
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index)
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setMessageVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index);
    /**
     * This forwards directly to Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing new agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param variable The value to set the specified variable
     * @param index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index)
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setNewAgentVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index);
    /**
     * This forwards directly to Curve::setArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param variable The value to set the specified variable
     * @param variable_index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of the array variable specified by variableName
     * @tparam M Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::setArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index)
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setAgentArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, T variable, unsigned int variable_index, unsigned int array_index);
    /**
     * This forwards directly to Curve::setArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index)
     *
     * The proxy method allows RTC's dynamic Curve headers to remove otherwise redundant branches when accessing new agent variables
     * @param variableName A constant char array (C string) variable name.
     * @param namespace_hash Curve namespace hash for the variable
     * @param variable The value to set the specified variable
     * @param variable_index The index of the variable in the named variable vector. This corresponds to the agent's index within the agent population.
     * @param array_index The index of the element in the named variable array.
     * @tparam T Type of the variable (This should be type of an element, if variable is an array).
     * @tparam N Length of the array variable specified by variableName
     * @tparam M Length of variableName, this should always be implicit if passing a string literal
     *
     * @see Curve::setArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index)
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setNewAgentArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, T variable, unsigned int variable_index, unsigned int array_index);

    static const int MAX_VARIABLES = 1024;          // !< Default maximum number of cuRVE variables (must be a power of 2)
    static const VariableHash EMPTY_FLAG = 0;
    static const VariableHash DELETED_FLAG = 1;

 private:
    /**
     * Check how many items are in the hash table
     * @return The number of items in the hash table
     * @note This private version assumes you have already locked mutex
     * @see Curve::size()
     */
    __host__ int _size() const;
    /**
     * Private version of Curve::registerVariableByHash(VariableHash, void*, size_t, unsigned int)
     *
     * Function for registering a variable by a VariableHash
     * This version assumes mutex has been handled externally
     * Registers a variable by insertion in a hash table.
     * @param variable_hash A cuRVE variable string hash from variableHash.
     * @param d_ptr a pointer to the vector which holds the hashed variable of give name
     * @param size Size of the data type (this should be the size of a single element if an array variable)
     * @param length Number of elements (1 unless the variable is an array)
     * @return Variable Handle of registered variable or UNKNOWN_VARIABLE if an error is encountered.
     * @see Curve::registerVariableByHash(VariableHash, void*, size_t, unsigned int)
     */
    __host__ Variable _registerVariableByHash(VariableHash variable_hash, void* d_ptr, size_t size, unsigned int length);
    /**
     * Private common implementation for mutex reasons
     * @see unregisterVariableByHash(VariableHash)
     */
    /**
     * Private version of Curve::unregisterVariableByHash(VariableHash)
     *
     * Function for un-registering a variable by a VariableHash
     * This version assumes mutex has been handled externally
     * Un-registers a variable by removing it from the internal hash table.
     * @param variable_hash A cuRVE variable string hash from variableHash.
     * @see unregisterVariableByHash(VariableHash)
     */
    __host__ void _unregisterVariableByHash(VariableHash variable_hash);
    /**
     * Device template function for getting a setting a single typed value from a constant string variable name
     *
     * Returns a single value from a constant string expression using the given index position
     * @param variableName A constant string variable name which must have been registered as a curve variable.
     * @param namespace_hash Curve namespace hash for the variable
     * @param index The index of the variable in the named variable vector
     * @param variable The typed value to set at the given index.
     * @tparam T Type of the variable being accessed
     * @tparam N variableName length, this should be ignored as it is implicitly set
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index);
    /**
     * Template device function for getting a single typed value from a constant string variable name
     *
     * Returns a single value from a constant string expression using the given index position
     * @param variableName A constant string variable name which must have been registered as a cuRVE variable.
     * @param namespace_hash Curve namespace hash for the variable
     * @param index The index of the variable in the named variable vector
     * @tparam T Type of the variable being accessed
     * @tparam N variableName length, this should be ignored as it is implicitly set
     * @return A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index);
    /**
     * Curve::getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index) via the read-only cache using __ldg() intrinsic
     *
     * Returns a single value from a constant string expression using the given index position
     * @param variableName A constant string variable name which must have been registered as a cuRVE variable.
     * @param namespace_hash Curve namespace hash for the variable
     * @param index The index of the variable in the named variable vector
     * @tparam T Type of the variable being accessed
     * @tparam N variable_name length, this should be ignored as it is implicitly set
     * @return A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     * @see Curve::getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index)
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getVariable_ldg(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index);
    /**
     * Returns a value from a position of a variable array
     * @param variableName Name of the variable
     * @param namespace_hash Curve hash of the agentfn or similar
     * @param variable_index Index of the agent or similar
     * @param array_index Index within the variable array
     * @tparam T Type of the variable array
     * @tparam N Length of the variable array
     * @tparam M variableName length, this should be ignored as it is implicitly set
     * @note Returns 0 on failure
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    /**
     * Curve::getArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index) via the read-only cache using __ldg() intrinsic
     *
     * Returns a value from a position of a variable array
     * @param variableName Name of the variable
     * @param namespace_hash Curve hash of the agentfn or similar
     * @param variable_index Index of the agent or similar
     * @param array_index Index within the variable array
     * @tparam T Type of the variable array
     * @tparam N Length of the variable array
     * @tparam M variableName length, this should be ignored as it is implicitly set
     * @note Returns 0 on failure
     * @see Curve::getArrayVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index)
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getArrayVariable_ldg(const char(&variableName)[M], VariableHash namespace_hash, unsigned int variable_index, unsigned int array_index);
    /**
     * Stores a value at a position of a variable array
     * @param variableName Name of the variable
     * @param namespace_hash Curve hash of the agentfn or similar
     * @param variable The value to store
     * @param variable_index Index of the agent or similar
     * @param array_index Index within the variable array
     * @note Returns 0 on failure
     */
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, T variable, unsigned int variable_index, unsigned int array_index);
    VariableHash h_hashes[MAX_VARIABLES];         // Host array of the hash values of registered variables
    void* h_d_variables[MAX_VARIABLES];           // Host array of pointer to device memory addresses for variable storage
    size_t h_sizes[MAX_VARIABLES];                // Host array of the sizes of registered variable types (Note: RTTI not supported in CUDA so this is the best we can do for now)
    unsigned int h_lengths[MAX_VARIABLES];        // Host array of the length of registered variables (i.e: vector length)
    bool deviceInitialised;                       // Flag indicating that curve has/hasn't been initialised yet on a device.

#ifndef __CUDACC_RTC__
    /**
     * Managed multi-threaded access to the internal storage
     * All read-only methods take a shared-lock
     * All methods which modify the internals require a unique-lock
     * Some private methods expect a lock to be gained before calling (to prevent the same thread attempting to lock the mutex twice)
     */
    mutable std::shared_timed_mutex mutex;
    /**
     * Returns a %std::shared_lock for accessing the host Curve object
     *
     * This should be called before calling const methods
     * @see Curve::getUniqueLock()
     */
    std::shared_lock<std::shared_timed_mutex> getSharedLock() const { return std::shared_lock<std::shared_timed_mutex>(mutex); }
    /**
     * Returns a %std::unique_lock for accessing the host Curve object
     *
     * This should be called before calling non-const (or const) methods
     * @see Curve::getUniqueLock()
     */
    std::unique_lock<std::shared_timed_mutex> getUniqueLock() const { return std::unique_lock<std::shared_timed_mutex>(mutex); }
#endif
    /**
     * Initialises cuRVE on the currently active device.
     * 
     * @note Not yet aware if the device has been reset.
     * @todo Need to add a device-side check for initialisation.
     */
    void initialiseDevice();
    /**
     * Has access to call purge
     */
    friend class CUDASimulation;
    /**
     * Wipes out host mirrors of device memory
     * Only really to be used after calls to cudaDeviceReset()
     * @note Only currently used after some tests
     */
    __host__ void purge();

 protected:
    /**
     * Default constructor.
     *
     * Private destructor to prevent this singleton being created more than once. Classes requiring a cuRVEInstance object should instead use the getInstance() method.
     * This ensure that curveInit is only ever called once by the program.
     * This will initialise the internal storage used for hash tables.
     */
    Curve();

 public:
#ifndef __CUDACC_RTC__
    /**
     * @brief    Gets the instance.
     *
     * @return    A new instance if this is the first request for an instance otherwise an existing instance.
     */
    static Curve& getInstance();
    static std::mutex instance_mutex;
#endif
};


namespace curve {
namespace detail {
    extern __constant__ Curve::VariableHash d_hashes[Curve::MAX_VARIABLES];   // Device array of the hash values of registered variables
    extern __device__ char* d_variables[Curve::MAX_VARIABLES];                // Device array of pointer to device memory addresses for variable storage
    extern __constant__ size_t d_sizes[Curve::MAX_VARIABLES];                // Device array of the types of registered variables
    extern __constant__ unsigned int d_lengths[Curve::MAX_VARIABLES];
}  // namespace detail
}  // namespace curve


/* TEMPLATE HASHING FUNCTIONS */

/** @brief Non terminal template structure has function for a constant char array
 *     Use of template meta-programming ensures the compiler can evaluate string hashes at compile time. This reduces constant string variable names to a single 32 bit value. Hashing is based on 'Quasi Compile Time String Hashing' at http:// www.altdevblogaday.com/2011/10/27/quasi-compile-time-string-hashing/
 *     Code uses compilation flags for both the host and the CUDA device.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N, unsigned int I> struct CurveStringHash {
    __device__ __host__ inline static Curve::VariableHash Hash(const char (&str)[N]) {
        return (CurveStringHash<N, I-1>::Hash(str) ^ str[I-1])*16777619u;
    }
};
/** @brief Terminal template structure has function for a constant char array
 *     Function within a template structure allows partial template specialisation for terminal case.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N> struct CurveStringHash<N, 1> {
    __device__ __host__ inline static Curve::VariableHash Hash(const char (&str)[N]) {
        return (2166136261u ^ str[0])*16777619u;
    }
};

#ifndef __CUDACC_RTC__
/**
 * Host side template class implementation
 */
template <unsigned int N, typename T>
__host__ Curve::Variable Curve::registerVariable(const char(&variableName)[N], void* d_ptr, unsigned int length) {
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
    VariableHash variable_hash = variableHash(variableName);
    size_t size = sizeof(T);
    return _registerVariableByHash(variable_hash, d_ptr, size, length);  // the const func can get const and non const argument (for 3rd argument)
}
template <unsigned int N>
__host__ void Curve::unregisterVariable(const char(&variableName)[N]) {
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
    VariableHash variable_hash = variableHash(variableName);
    _unregisterVariableByHash(variable_hash);
}
#endif

/**
* Device side class implementation
*/
/* loop unrolling of hash collision detection */
__device__ __forceinline__ Curve::Variable Curve::getVariable(const VariableHash variable_hash) {
    for (unsigned int x = 0; x< MAX_VARIABLES; x++) {
        const Variable i = ((variable_hash + x) & (MAX_VARIABLES - 1));
        const VariableHash h = curve::detail::d_hashes[i];
        if (h == variable_hash)
            return i;
    }
    return UNKNOWN_VARIABLE;
}


template <unsigned int N>
__device__ __host__ __forceinline__ Curve::VariableHash Curve::variableHash(const char(&str)[N]) {
    return CurveStringHash<N, N>::Hash(str);
}
__device__ __forceinline__ size_t Curve::getVariableSize(const VariableHash variable_hash) {
    Variable cv;

    cv = getVariable(variable_hash);

    return curve::detail::d_sizes[cv];
}
__device__ __forceinline__ unsigned int Curve::getVariableLength(const VariableHash variable_hash) {
    Variable cv;

    cv = getVariable(variable_hash);

    return curve::detail::d_lengths[cv];
}
__device__ __forceinline__ void* Curve::getVariablePtrByHash(const VariableHash variable_hash, size_t offset) {
    Variable cv;

    cv = getVariable(variable_hash);
#if !defined(SEATBELTS) || SEATBELTS
    // error checking
    if (cv == UNKNOWN_VARIABLE) {
        return nullptr;
    }

    // check vector length
    if (offset > curve::detail::d_sizes[cv] * curve::detail::d_lengths[cv]) {  // Note : offset is basicly index * sizeof(T)
        return nullptr;
    }
#endif
    // return a generic pointer to variable address for given offset (no bounds checking here!)
    return curve::detail::d_variables[cv] + offset;
}
template <typename T>
__device__ __forceinline__ T Curve::getVariableByHash(const VariableHash variable_hash, unsigned int index) {
    size_t offset = index *sizeof(T);

#if !defined(SEATBELTS) || SEATBELTS
    // do a check on the size as otherwise the value_ptr may eb out of bounds.
    size_t size = getVariableSize(variable_hash);

    // error checking
    if (size != sizeof(T)) {
        return 0;
    }
#endif
    // get a pointer to the specific variable by offsetting by the provided index
    T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));

#if !defined(SEATBELTS) || SEATBELTS
    if (!value_ptr)
        return 0;
#endif
    return *value_ptr;
}
template <typename T>
__device__ __forceinline__ T Curve::getVariableByHash_ldg(const VariableHash variable_hash, unsigned int index) {
    size_t offset = index *sizeof(T);

#if !defined(SEATBELTS) || SEATBELTS
    // do a check on the size as otherwise the value_ptr may eb out of bounds.
    size_t size = getVariableSize(variable_hash);

    // error checking
    if (size != sizeof(T)) {
        return NULL;
    }
#endif
    // get a pointer to the specific variable by offsetting by the provided index
    T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));

#if !defined(SEATBELTS) || SEATBELTS
    if (!value_ptr)
        return 0;
#endif
    return __ldg(value_ptr);
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getArrayVariableByHash(const VariableHash variable_hash, unsigned int agent_index, unsigned int array_index) {
    // do a check on the size as otherwise the value_ptr may eb out of bounds.
    const size_t var_size = N * sizeof(T);
    // error checking
#if !defined(SEATBELTS) || SEATBELTS
    const size_t size = getVariableSize(variable_hash);
    if (size != var_size) {
        return NULL;
    }
#endif
    const size_t offset = (agent_index * var_size) + (array_index * sizeof(T));
    // get a pointer to the specific variable by offsetting by the provided index
    T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));

#if !defined(SEATBELTS) || SEATBELTS
    if (!value_ptr)
        return 0;
#endif
    return *value_ptr;
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getArrayVariableByHash_ldg(const VariableHash variable_hash, unsigned int agent_index, unsigned int array_index) {
    // do a check on the size as otherwise the value_ptr may eb out of bounds.
    const size_t var_size = N * sizeof(T);
    // error checking
#if !defined(SEATBELTS) || SEATBELTS
    const size_t size = getVariableSize(variable_hash);
    if (size != var_size) {
        return NULL;
    }
#endif
    const size_t offset = (agent_index * var_size) + (array_index * sizeof(T));
    // get a pointer to the specific variable by offsetting by the provided index
    T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));

#if !defined(SEATBELTS) || SEATBELTS
    if (!value_ptr)
        return 0;
#endif
    return __ldg(value_ptr);
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getAgentVariable(const char (&variableName)[N], VariableHash namespace_hash, unsigned int index) {
    return getVariable<T>(variableName, namespace_hash, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getMessageVariable(const char (&variableName)[N], VariableHash namespace_hash, unsigned int index) {
    return getVariable<T>(variableName, namespace_hash, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getVariable(const char (&variableName)[N], VariableHash namespace_hash, unsigned int index) {
    VariableHash variable_hash = variableHash(variableName);
#if !defined(SEATBELTS) || SEATBELTS
    {
        const auto cv = getVariable(variable_hash+namespace_hash);
        if (cv ==  UNKNOWN_VARIABLE) {
            DTHROW("Curve variable with name '%s' was not found.\n", variableName);
        } else if (curve::detail::d_sizes[cv] != sizeof(T)) {
            DTHROW("Curve variable with name '%s' type size mismatch %llu != %llu.\n", variableName, curve::detail::d_sizes[cv], sizeof(T));
        }
    }
#endif
    return getVariableByHash<T>(variable_hash+namespace_hash, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getAgentVariable_ldg(const char (&variableName)[N], VariableHash namespace_hash, unsigned int index) {
    return getVariable_ldg<T>(variableName, namespace_hash, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getMessageVariable_ldg(const char (&variableName)[N], VariableHash namespace_hash, unsigned int index) {
    return getVariable_ldg<T>(variableName, namespace_hash, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ T Curve::getVariable_ldg(const char (&variableName)[N], VariableHash namespace_hash, unsigned int index) {
    VariableHash variable_hash = variableHash(variableName);
#if !defined(SEATBELTS) || SEATBELTS
    {
        const auto cv = getVariable(variable_hash+namespace_hash);
        if (cv ==  UNKNOWN_VARIABLE) {
            DTHROW("Curve variable with name '%s' was not found.\n", variableName);
        } else if (curve::detail::d_sizes[cv] != sizeof(T)) {
            DTHROW("Curve variable with name '%s' type size mismatch %llu != %llu.\n", variableName, curve::detail::d_sizes[cv], sizeof(T));
        }
    }
#endif
    return getVariableByHash_ldg<T>(variable_hash+namespace_hash, index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T Curve::getAgentArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index) {
    return getArrayVariable<T, N>(variableName, namespace_hash, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T Curve::getArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index) {
    VariableHash variable_hash = variableHash(variableName);
#if !defined(SEATBELTS) || SEATBELTS
    {
        const auto cv = getVariable(variable_hash+namespace_hash);
        if (cv ==  UNKNOWN_VARIABLE) {
            DTHROW("Curve variable array with name '%s' was not found.\n", variableName);
        } else if (curve::detail::d_sizes[cv] != sizeof(T) * N) {
            DTHROW("Curve variable array with name '%s', type size mismatch %llu != %llu.\n", variableName, curve::detail::d_sizes[cv], sizeof(T) * N);
        }
    }
    if (array_index >= N) {
        DTHROW("Curve array index %u is out of bounds for variable with name '%s'.\n", array_index, variableName);
        return 0;
    }
#endif
    // Curve currently doesn't store whether a variable is an array
    // Curve stores M * sizeof(T), so this is checked instead
    return getArrayVariableByHash<T, N>(variable_hash + namespace_hash, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T Curve::getAgentArrayVariable_ldg(const char(&variableName)[M], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index) {
    return getArrayVariable_ldg<T, N>(variableName, namespace_hash, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T Curve::getArrayVariable_ldg(const char(&variableName)[M], VariableHash namespace_hash, unsigned int agent_index, unsigned int array_index) {
    VariableHash variable_hash = variableHash(variableName);
#if !defined(SEATBELTS) || SEATBELTS
    {
        const auto cv = getVariable(variable_hash+namespace_hash);
        if (cv ==  UNKNOWN_VARIABLE) {
            DTHROW("Curve variable array with name '%s' was not found.\n", variableName);
        } else if (curve::detail::d_sizes[cv] != sizeof(T) * N) {
            DTHROW("Curve variable array with name '%s', type size mismatch %llu != %llu.\n", variableName, curve::detail::d_sizes[cv], sizeof(T) * N);
        }
    }
    if (array_index >= N) {
        DTHROW("Curve array index %u is out of bounds for variable with name '%s'.\n", array_index, variableName);
        return 0;
    }
#endif
    // Curve currently doesn't store whether a variable is an array
    // Curve stores M * sizeof(T), so this is checked instead
    return getArrayVariableByHash_ldg<T, N>(variable_hash + namespace_hash, agent_index, array_index);
}

template <typename T>
__device__ __forceinline__ void Curve::setVariableByHash(const VariableHash variable_hash, T variable, unsigned int index) {
#if !defined(SEATBELTS) || SEATBELTS
    size_t size = getVariableSize(variable_hash);
    if (size != sizeof(T)) {
        return;
    }
#endif
    size_t offset = index *sizeof(T);
    T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));
    *value_ptr = variable;
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setArrayVariableByHash(const VariableHash variable_hash, T variable, unsigned int agent_index, unsigned int array_index) {
    const size_t var_size = N * sizeof(T);
#if !defined(SEATBELTS) || SEATBELTS
    const size_t size = getVariableSize(variable_hash);
    if (size != var_size) {
        return;
    }
#endif
    const size_t offset = (agent_index * var_size) + (array_index * sizeof(T));
    T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));
    *value_ptr = variable;
}

template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setAgentVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index) {
    setVariable<T>(variableName, namespace_hash, variable, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setMessageVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index) {
    setVariable<T>(variableName, namespace_hash, variable, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setNewAgentVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index) {
    setVariable<T>(variableName, namespace_hash, variable, index);
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index) {
    VariableHash variable_hash = variableHash(variableName);
#if !defined(SEATBELTS) || SEATBELTS
    {
        const auto cv = getVariable(variable_hash+namespace_hash);
        if (cv ==  UNKNOWN_VARIABLE) {
            DTHROW("Curve variable with name '%s' was not found.\n", variableName);
        } else if (curve::detail::d_sizes[cv] != sizeof(T)) {
            DTHROW("Curve variable with name '%s', type size mismatch %llu != %llu.\n", variableName, curve::detail::d_sizes[cv], sizeof(T));
        }
    }
#endif
    setVariableByHash<T>(variable_hash+namespace_hash, variable, index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void Curve::setAgentArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index) {
    setArrayVariable<T, N>(variableName, namespace_hash, variable, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void Curve::setNewAgentArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index) {
    setArrayVariable<T, N>(variableName, namespace_hash, variable, agent_index, array_index);
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void Curve::setArrayVariable(const char(&variableName)[M], VariableHash namespace_hash, T variable, unsigned int agent_index, unsigned int array_index) {
    VariableHash variable_hash = variableHash(variableName);
#if !defined(SEATBELTS) || SEATBELTS
    {
        const auto cv = getVariable(variable_hash+namespace_hash);
        if (cv ==  UNKNOWN_VARIABLE) {
            DTHROW("Curve variable array with name '%s' was not found.\n", variableName);
        } else if (curve::detail::d_sizes[cv] != sizeof(T) * N) {
            DTHROW("Curve variable array with name '%s', size mismatch %llu != %llu.\n", variableName, curve::detail::d_sizes[cv], sizeof(T) * N);
        }
    }
    if (array_index >= N) {
        DTHROW("Curve array index %u is out of bounds for variable with name '%s'.\n", array_index, variableName);
        return;
    }
#endif
    // Curve currently doesn't store whether a variable is an array
    // Curve stores M * sizeof(T), so this is checked instead
    return setArrayVariableByHash<T, N>(variable_hash + namespace_hash, variable, agent_index, array_index);
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_H_

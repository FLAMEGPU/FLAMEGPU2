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

#include <cuda_runtime.h>

#include <cstring>
#include <typeinfo>


/** @brief    A cuRVE instance.
 *
 * cuRVE is a C library and this singleton class acts as a mechanism to ensure that any reference to the library is handled correctly.
 * For example multiple objects may which to request that curve is initialised. This class will ensure that this function call is only made once the first time that a cuRVEInstance is required.
 */
class Curve {
 public:
    static const int UNKNOWN_CURVE_VARIABLE = -1;             // !< value returned as a CurveVariable if an API function encounters an error

    typedef int                     CurveVariable;            // !< Typedef for cuRVE variable handle
    typedef unsigned int             CurveVariableHash;       // !< Typedef for cuRVE variable name string hash
    typedef unsigned int             CurveNamespaceHash;      // !< Typedef for cuRVE variable namespace string hash

    /**
     * Enumerator for GPU device error code which may be raised by CUDA kernels
     */
    enum curveDeviceError {
        CURVE_DEVICE_ERROR_NO_ERRORS,                // !< No errors raised on the device
        CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE,         // !< A function has requested an unknown variable or a variable not registered in the current namespace
        CURVE_DEVICE_ERROR_VARIABLE_DISABLED,        // !< A function has requested a variable which is disabled
        CURVE_DEVICE_ERROR_UNKNOWN_TYPE,             // !< A function has requested an unknown type or a type not registered in the current namespace
        CURVE_DEVICE_ERROR_UNKNOWN_LENGTH            // !< A function has requested an unknown vector length or the length not registered in the current namespace
    };

    /**
     * Enumerator for cuRVE host error codes which may be raised by cuRVE API function calls
     */
    enum curveHostError {
        CURVE_ERROR_NO_ERRORS,                       // !< No errors raised by host API functions
        CURVE_ERROR_UNKNOWN_VARIABLE,                // !< A host API function has requested an unknown variable or a variable not registered in the current namespace
        CURVE_ERROR_TOO_MANY_VARIABLES               // !< The maximum number of curve variables has been reached
    };

    /** @brief Main cuRVE variable hashing function for strings of length determined at runtime and not compile time
     *  Should only be used for registered variables as this will be much slower than the compile time alternative.
     *  @return a 32 bit cuRVE string variable hash.
     */
    __host__ static CurveVariableHash curveVariableRuntimeHash(const char* str);

    /** @brief Main cuRVE variable hashing function
     *  Calls recursive hashing functions
     *  @return a 32 bit cuRVE string variable hash.
     */
    template <unsigned int N>
    __device__ __host__ __forceinline__ static CurveVariableHash curveVariableHash(const char(&str)[N]);
    /** @brief Function for getting a handle (hash table index) to a cuRVE variable from a variable string hash
     *     Function performs hash collision avoidance using linear probing.
     *  @param variable_hash A cuRVE variable string hash from curveVariableHash.
     *  @return CurveVariable Handle for the cuRVE variable.
     */
    __host__ CurveVariable curveGetVariableHandle(CurveVariableHash variable_hash);
    /** @brief Function for registering a variable by a CurveVariableHash
     *     Registers a variable by insertion in a hash table. Recommend using the provided curveRegisterVariable template function.
     *  @param variable_hash A cuRVE variable string hash from curveVariableHash.
     *  @param d_ptr a pointer to the vector which holds the hashed variable of give name
     *  @return CurveVariable Handle of registered variable or UNKNOWN_CURVE_VARIABLE if an error is encountered.
     */
    __host__ CurveVariable curveRegisterVariableByHash(CurveVariableHash variable_hash, void* d_ptr, size_t size, unsigned int length);
    /** @brief Template function for registering a constant string
     *     Registers a constant string variable name by hashing and then inserting into a hash table.
     *  @param variableName A constant char array (C string) variable name.
     *  @param d_ptr a pointer to the vector which holds the variable of give name
     *  @return CurveVariable Handle of registered variable or UNKNOWN_CURVE_VARIABLE if an error is encountered.
     */ // Note: this function was never called
    template <unsigned int N, typename T>
    __host__ CurveVariable curveRegisterVariable(const char(&variableName)[N], void* d_ptr, unsigned int length);
    /** @brief Function for un-registering a variable by a CurveVariableHash
     *     Un-registers a variable by removal from a hash table. Recommend using the provided curveUnregisterVariable template function.
     *  @param variable_hash A cuRVE variable string hash from curveVariableHash.
     */
    __host__ void curveUnregisterVariableByHash(CurveVariableHash variable_hash);
    /** @brief Template function for un-registering a constant string
     *     Un-registers a constant string variable name by hashing and then removing from the hash table.
     *  @param variableName A constant char array (C string) variable name.
     */
    template <unsigned int N>
    __host__ void curveUnregisterVariable(const char(&variableName)[N]);
    /** @brief Function for disabling access to a cuRVE variable from a CurveVariableHash
     *     Disables device access to the cuRVE variable. Does not disable host access.
     *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
     */
    __host__ void curveDisableVariableByHash(CurveVariableHash variable_hash);
    /** @brief Template function for disabling access to a cuRVE variable from a constant string variable name
     *     Disables device access to the cuRVE variable. Does not disable host access.
     *  @param variableName A constant string variable name which must have been registered as a curve variable
     */
    template <unsigned int N>
    __host__ void curveDisableVariable(const char(&variableName)[N]);
    /** @brief Function for enabling access to a cuRVE variable from a CurveVariableHash
     *     Enables device access to the cuRVE variable.
     *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
     */
    __host__ void curveEnableVariableByHash(CurveVariableHash variable_hash);
    /** @brief Template function for enabling access to a cuRVE variable from a constant string variable name
     *     Enables device access to the cuRVE variable.
     *  @param variableName A constant string variable name which must have been registered as a curve variable
     */
    template <unsigned int N>
    __host__ void curveEnableVariable(const char(&variableName)[N]);
    /** @brief Function changes the current namespace from a CurveNamespaceHash
     *     Changing the namespace will affect both the host and device.
     *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
     */
    __host__ void curveSetNamespaceByHash(CurveNamespaceHash variable_hash);
    /** @brief Function changes the current namespace to the default empty namespace
     *     Changing the namespace will affect both the host and device.
     */
    __host__ void curveSetDefaultNamespace();
    /** @brief Template function changes the current namespace using a constant string namespace name
     *     Changing the namespace will affect both the host and device.
     *  @param namespaceName A constant string variable name which must have been registered as a curve variable
     */
    template <unsigned int N>
    __host__ void curveSetNamespace(const char(&namespaceName)[N]);



    __device__ __forceinline__ static CurveVariable getVariable(const CurveVariableHash variable_hash);
    /** @brief Gets the size of the cuRVE variable type given the variable hash
     * Gets the size of the cuRVE variable type given the variable hash
     *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
     *  @return A size_t which is the size of the variable or 0 otherwise
     */
    __device__ __forceinline__ static size_t curveGetVariableSize(const CurveVariableHash variable_hash);
    /** @brief Device function for getting a pointer to a variable of given name
     * Returns a generic pointer to a variable of given name at a specific offset in bytes from the start of the variable array.
     *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
     *  @param offset an offset into the variable array in bytes (offset is variable index * sizeof(variable type))
     *  @return A generic pointer to the variable value. Will be nullptr if there is an error.
     */
    __device__ __forceinline__ static void* curveGetVariablePtrByHash(const CurveVariableHash variable_hash, size_t offset);
    /** @brief Device function for getting a single typed value from a CurveVariableHash at a given index
     *     Returns a single value of specified type from a curveVariableHash using the given index position.
     * Note: No runtime type checking is done. TODO: Add runtime type checking for debug modes.
     *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
     *  @param index The index of the variable in the named variable vector.
     *  @return T A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     */
    template <typename T>
    __device__ __forceinline__ static float curveGetVariableByHash(const CurveVariableHash variable_hash, unsigned int index);
    /** @brief Template device function for getting a single typed value from a constant string variable name
     *     Returns a single value from a constant string expression using the given index position
     *  @param variableName A constant string variable name which must have been registered as a cuRVE variable.
     *  @param index The index of the variable in the named variable vector
     *  @return T A value of given typr at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static float curveGetVariable(const char(&variableName)[N], CurveVariableHash namespace_hash, unsigned int index);
    /** @brief Device function for setting a single typed value from a CurveVariableHash
     *     Sets a single value from a curveVariableHash using the given index position.
     *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
     *  @param index The index of the variable in the named variable vector.
     *  @param value The typed value to set at the given index.
     */
    template <typename T>
    __device__ __forceinline__ static void curveSetVariableByHash(const CurveVariableHash variable_hash, T variable, unsigned int index);
    /** @brief Device template function for getting a setting a single typed value from a constant string variable name
     *     Returns a single value from a constant string expression using the given index position
     *  @param variableName A constant string variable name which must have been registered as a curve variable.
     *  @param index The index of the variable in the named variable vector
     *  @param value The typed value to set at the given index.
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void curveSetVariable(const char(&variableName)[N], CurveVariableHash namespace_hash, T variable, unsigned int index);

    /* ERROR CHECKING API FUNCTIONS */

    /** @brief Device function for printing the last device error
    *     Prints the last device error using the provided source location information. The preferred method for printing is to use the curveReportLastDeviceError macro which inserts source location information.
    *  @param file A constant string filename.
    *  @param function A constant string function name.
    *  @param line A constant integer line number.
    */
    __device__ __forceinline__ static void curvePrintLastDeviceError(const char* file, const char* function, const int line);
    /** @brief Host API function for printing the last host error
     *     Prints the last host API error using the provided source location information. The preferred method for printing is to use the curveReportLastHostError macro which inserts source location information.
     *  @param file A constant string filename.
     *  @param function A constant string function name.
     *  @param line A constant integer line number.
     */
    void __host__ curvePrintLastHostError(const char* file, const char* function, const int line);

    /** @brief Host API function for printing the last host or device error
     *     Prints the last device or host API error (or both) using the provided source location information. The preferred method for printing is to use the curveReportErrors macro which inserts source location information.
     *  @param file A constant string filename.
     *  @param function A constant string function name.
     *  @param line A constant integer line number.
     */
    void __host__ curvePrintErrors(const char* file, const char* function, const int line);
    /** @brief Device API function for returning a constant string error description
     *     Returns an error description given a curveDeviceError error code.
     *  @param error_code A curveDeviceError error code.
     *  @return constant A string error description
     */
    __device__ __host__ __forceinline__ static const char*  curveGetDeviceErrorString(curveDeviceError error_code);
    /** @brief Host API function for returning a constant string error description
     *     Returns an error description given a curveHostError error code.
     *  @param error_code A curveHostError error code.
     *  @return constant A string error description
     */
    __host__ const char*  curveGetHostErrorString(curveHostError error_code);
    /** @brief Device API function for returning the last reported error code
     *  @return A curveDeviceError error code
     */
    __device__ __forceinline__ static curveDeviceError curveGetLastDeviceError();

    /** @brief Host API function for returning the last reported error code
     *  @return A curveHostError error code
     */
    __host__ curveHostError curveGetLastHostError();
    /** @brief Host API function for clearing both the device and host error codes
    */
    __host__ void curveClearErrors();

    unsigned int h_namespace;
    static const int CURVE_MAX_VARIABLES = 32;          // !< Default maximum number of cuRVE variables (must be a power of 2)
    static const int VARIABLE_DISABLED = 0;
    static const int VARIABLE_ENABLED = 1;
    static const int NAMESPACE_NONE = 0;

 private:
    CurveVariableHash h_hashes[CURVE_MAX_VARIABLES];    // Host array of the hash values of registered variables
    void* h_d_variables[CURVE_MAX_VARIABLES];           // Host array of pointer to device memory addresses for variable storage
    int    h_states[CURVE_MAX_VARIABLES];               // Host array of the states of registered variables
    size_t h_sizes[CURVE_MAX_VARIABLES];                // Host array of the sizes of registered variable types (Note: RTTI not supported in CUDA so this is the best we can do for now)
    unsigned int h_lengths[CURVE_MAX_VARIABLES];        // Host array of the length of registered variables (i.e: vector length)

 protected:
    /** @brief    Default constructor.
     *
     *  Private destructor to prevent this singleton being created more than once. Classes requiring a cuRVEInstance object should instead use the getInstance() method.
     *  This ensure that curveInit is only ever called once by the program.
     *  This will initialise the internal storage used for hash tables.
     */
     Curve();

    ~Curve() {}

 public:
    /**
     * @brief    Gets the instance.
     *
     * @return    A new instance if this is the first request for an instance otherwise an existing instance.
     */
    static Curve& getInstance() {
        static Curve c;
        return c;
    }
};


namespace curve_internal {
    extern __constant__ Curve::CurveNamespaceHash d_namespace;
    extern __constant__ Curve::CurveVariableHash d_hashes[Curve::CURVE_MAX_VARIABLES];    // Device array of the hash values of registered variables
    extern __device__ char* d_variables[Curve::CURVE_MAX_VARIABLES];                // Device array of pointer to device memory addresses for variable storage
    extern __constant__ int d_states[Curve::CURVE_MAX_VARIABLES];                    // Device array of the states of registered variables
    extern  __constant__ size_t d_sizes[Curve::CURVE_MAX_VARIABLES];                // Device array of the types of registered variables
    extern __constant__ unsigned int d_lengths[Curve::CURVE_MAX_VARIABLES];

    extern __device__ Curve::curveDeviceError d_curve_error;
    extern Curve::curveHostError h_curve_error;
}  // namespace curve_internal


/* TEMPLATE HASHING FUNCTIONS */

/** @brief Non terminal template structure has function for a constant char array
 *     Use of template meta-programming ensures the compiler can evaluate string hashes at compile time. This reduces constant string variable names to a single 32 bit value. Hashing is based on 'Quasi Compile Time String Hashing' at http:// www.altdevblogaday.com/2011/10/27/quasi-compile-time-string-hashing/
 *     Code uses compilation flags for both the host and the CUDA device.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N, unsigned int I> struct CurveStringHash {
    __device__ __host__ inline static Curve::CurveVariableHash Hash(const char (&str)[N]) {
        return (CurveStringHash<N, I-1>::Hash(str) ^ str[I-1])*16777619u;
    }
};
/** @brief Terminal template structure has function for a constant char array
 *     Function within a template structure allows partial template specialisation for terminal case.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N> struct CurveStringHash<N, 1> {
    __device__ __host__ inline static Curve::CurveVariableHash Hash(const char (&str)[N]) {
        return (2166136261u ^ str[0])*16777619u;
    }
};

/**
 * Host side template class implementation
 */
template <unsigned int N, typename T>
__host__ Curve::CurveVariable Curve::curveRegisterVariable(const char(&variableName)[N], void* d_ptr, unsigned int length) {
    CurveVariableHash variable_hash = curveVariableHash(variableName);
    size_t size = sizeof(T);
    return curveRegisterVariableByHash(variable_hash, d_ptr, size, length);  // the const func can get const and non const argument (for 3rd argument)
}
template <unsigned int N>
__host__ void Curve::curveUnregisterVariable(const char(&variableName)[N]) {
    CurveVariableHash variable_hash = curveVariableHash(variableName);
    curveUnregisterVariableByHash(variable_hash);
}
template <unsigned int N>
__host__ void Curve::curveDisableVariable(const char (&variableName)[N]) {
    CurveVariableHash variable_hash = curveVariableHash(variableName);
    curveDisableVariableByHash(variable_hash);
}
template <unsigned int N>
__host__ void Curve::curveEnableVariable(const char (&variableName)[N]) {
    CurveVariableHash variable_hash = curveVariableHash(variableName);
    curveEnableVariableByHash(variable_hash);
}
template <unsigned int N>
__host__ void Curve::curveSetNamespace(const char (&namespaceName)[N]) {
    CurveNamespaceHash namespace_hash = curveVariableHash(namespaceName);
    curveSetNamespaceByHash(namespace_hash);
}

/**
* Device side class implementation
*/
/* loop unrolling of hash collision detection */
__device__ __forceinline__ Curve::CurveVariable Curve::getVariable(const CurveVariableHash variable_hash) {
    const CurveVariableHash hash = variable_hash + curve_internal::d_namespace;
    for (unsigned int x = 0; x< CURVE_MAX_VARIABLES; x++) {
        const CurveVariable i = ((hash + x) & (CURVE_MAX_VARIABLES - 1));
        const CurveVariableHash h = curve_internal::d_hashes[i];
        if (h == hash)
            return i;
    }
    return UNKNOWN_CURVE_VARIABLE;
}


template <unsigned int N>
__device__ __host__ __forceinline__ Curve::CurveVariableHash Curve::curveVariableHash(const char(&str)[N]) {
    return CurveStringHash<N, N>::Hash(str);
}
__device__ __forceinline__ size_t Curve::curveGetVariableSize(const CurveVariableHash variable_hash) {
    CurveVariable cv;

    cv = getVariable(variable_hash);

    return curve_internal::d_sizes[cv];
}
__device__ __forceinline__ void* Curve::curveGetVariablePtrByHash(const CurveVariableHash variable_hash, size_t offset) {
    CurveVariable cv;

    cv = getVariable(variable_hash);
    // error checking
    if (cv == UNKNOWN_CURVE_VARIABLE) {
        curve_internal::d_curve_error = CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE;
        return nullptr;
    }
    if (!curve_internal::d_states[cv]) {
        curve_internal::d_curve_error = CURVE_DEVICE_ERROR_VARIABLE_DISABLED;
        return nullptr;
    }

    // check vector length
    if (offset > curve_internal::d_sizes[cv] * curve_internal::d_lengths[cv]) {  // Note : offset is basicly index * sizeof(T)
        curve_internal::d_curve_error = CURVE_DEVICE_ERROR_UNKNOWN_LENGTH;
        return nullptr;
    }

    // return a generic pointer to variable address for given offset (no bounds checking here!)
    return curve_internal::d_variables[cv] + offset;
}
template <typename T>
__device__ __forceinline__ float Curve::curveGetVariableByHash(const CurveVariableHash variable_hash, unsigned int index) {
    size_t offset = index *sizeof(T);

    // do a check on the size as otherwise the value_ptr may eb out of bounds.
    size_t size = curveGetVariableSize(variable_hash);

    // error checking
    if (size != sizeof(T)) {
        curve_internal::d_curve_error = CURVE_DEVICE_ERROR_UNKNOWN_TYPE;
        return NULL;
    } else {
        // get a pointer to the specific variable by offsetting by the provided index
        T *value_ptr = reinterpret_cast<T*>(curveGetVariablePtrByHash(variable_hash, offset));

        if (value_ptr)
            return *value_ptr;
        else
            return 0;
    }
}
template <typename T, unsigned int N>
__device__ __forceinline__ float Curve::curveGetVariable(const char (&variableName)[N], CurveVariableHash namespace_hash, unsigned int index) {
    CurveVariableHash variable_hash = curveVariableHash(variableName);

    return curveGetVariableByHash<T>(variable_hash+namespace_hash, index);
}
template <typename T>
__device__ __forceinline__ void Curve::curveSetVariableByHash(const CurveVariableHash variable_hash, T variable, unsigned int index) {
    size_t size = curveGetVariableSize(variable_hash);

    if (size != sizeof(T)) {
        curve_internal::d_curve_error = CURVE_DEVICE_ERROR_UNKNOWN_TYPE;
    } else {
        size_t offset = index *sizeof(T);
        T *value_ptr = reinterpret_cast<T*>(curveGetVariablePtrByHash(variable_hash, offset));
        *value_ptr = variable;
    }
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::curveSetVariable(const char(&variableName)[N], CurveVariableHash namespace_hash, T variable, unsigned int index) {
    CurveVariableHash variable_hash = curveVariableHash(variableName);
    curveSetVariableByHash<T>(variable_hash+namespace_hash, variable, index);
}

/* ERROR CHECKING API FUNCTIONS */
#define curveReportLastDeviceError() { Curve::curvePrintLastDeviceError(__FILE__, __FUNCTION__, __LINE__); }    // ! Prints the last reported device error using the file, function and line number of the call to this macro
#define curveReportLastHostError() { curvePrintLastHostError(__FILE__, __FUNCTION__, __LINE__); }        // ! Prints the last reported host API error using the file, function and line number of the call to this macro
#define curveReportErrors() { curvePrintErrors(__FILE__, __FUNCTION__, __LINE__); }  // ! Prints the last reported device or host API error using the file, function and line number of the call to this macro

__device__ __forceinline__ void Curve::curvePrintLastDeviceError(const char* file, const char* function, const int line) {
    if (curve_internal::d_curve_error != CURVE_DEVICE_ERROR_NO_ERRORS) {
        printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)curve_internal::d_curve_error, curveGetDeviceErrorString(curve_internal::d_curve_error));
    }
}
__device__ __host__ __forceinline__ const char* Curve::curveGetDeviceErrorString(curveDeviceError e) {
    switch (e) {
    case(CURVE_DEVICE_ERROR_NO_ERRORS):
        return "No cuRVE errors";
    case(CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE):
        return "Unknown cuRVE variable in current namespace";
    case(CURVE_DEVICE_ERROR_VARIABLE_DISABLED):
        return "cuRVE variable is disabled";
    default:
        return "Unspecified cuRVE error";
    }
}
__device__ __forceinline__ Curve::curveDeviceError Curve::curveGetLastDeviceError() {
    return curve_internal::d_curve_error;
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_H_

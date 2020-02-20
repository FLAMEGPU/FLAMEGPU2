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
#include <cstdio>
#include <typeinfo>


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
     * Enumerator for GPU device error code which may be raised by CUDA kernels
     */
    enum DeviceError {
        DEVICE_ERROR_NO_ERRORS,                // !< No errors raised on the device
        DEVICE_ERROR_UNKNOWN_VARIABLE,         // !< A function has requested an unknown variable or a variable not registered in the current namespace
        DEVICE_ERROR_VARIABLE_DISABLED,        // !< A function has requested a variable which is disabled
        DEVICE_ERROR_UNKNOWN_TYPE,             // !< A function has requested an unknown type or a type not registered in the current namespace
        DEVICE_ERROR_UNKNOWN_LENGTH            // !< A function has requested an unknown vector length or the length not registered in the current namespace
    };

    /**
     * Enumerator for cuRVE host error codes which may be raised by cuRVE API function calls
     */
    enum HostError {
        ERROR_NO_ERRORS,                       // !< No errors raised by host API functions
        ERROR_UNKNOWN_VARIABLE,                // !< A host API function has requested an unknown variable or a variable not registered in the current namespace
        ERROR_TOO_MANY_VARIABLES               // !< The maximum number of curve variables has been reached
    };

    /** @brief Main cuRVE variable hashing function for strings of length determined at runtime and not compile time
     *  Should only be used for registered variables as this will be much slower than the compile time alternative.
     *  @return a 32 bit cuRVE string variable hash.
     */
    __host__ static VariableHash variableRuntimeHash(const char* str);

    /** @brief Main cuRVE variable hashing function
     *  Calls recursive hashing functions
     *  @return a 32 bit cuRVE string variable hash.
     */
    template <unsigned int N>
    __device__ __host__ __forceinline__ static VariableHash variableHash(const char(&str)[N]);
    /** @brief Function for getting a handle (hash table index) to a cuRVE variable from a variable string hash
     *     Function performs hash collision avoidance using linear probing.
     *  @param variable_hash A cuRVE variable string hash from variableHash.
     *  @return Variable Handle for the cuRVE variable.
     */
    __host__ Variable getVariableHandle(VariableHash variable_hash);
    /** @brief Function for registering a variable by a VariableHash
     *     Registers a variable by insertion in a hash table. Recommend using the provided registerVariable template function.
     *  @param variable_hash A cuRVE variable string hash from variableHash.
     *  @param d_ptr a pointer to the vector which holds the hashed variable of give name
     *  @return Variable Handle of registered variable or UNKNOWN_VARIABLE if an error is encountered.
     */
    __host__ Variable registerVariableByHash(VariableHash variable_hash, void* d_ptr, size_t size, unsigned int length);
    /** @brief Template function for registering a constant string
     *     Registers a constant string variable name by hashing and then inserting into a hash table.
     *  @param variableName A constant char array (C string) variable name.
     *  @param d_ptr a pointer to the vector which holds the variable of give name
     *  @return Variable Handle of registered variable or UNKNOWN_VARIABLE if an error is encountered.
     */ // Note: this function was never called
    template <unsigned int N, typename T>
    __host__ Variable registerVariable(const char(&variableName)[N], void* d_ptr, unsigned int length);
    /** @brief Function for un-registering a variable by a VariableHash
     *     Un-registers a variable by removal from a hash table. Recommend using the provided unregisterVariable template function.
     *  @param variable_hash A cuRVE variable string hash from variableHash.
     */
    __host__ void unregisterVariableByHash(VariableHash variable_hash);
    /** @brief Template function for un-registering a constant string
     *     Un-registers a constant string variable name by hashing and then removing from the hash table.
     *  @param variableName A constant char array (C string) variable name.
     */
    template <unsigned int N>
    __host__ void unregisterVariable(const char(&variableName)[N]);
    /** @brief Function for disabling access to a cuRVE variable from a VariableHash
     *     Disables device access to the cuRVE variable. Does not disable host access.
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     */
    __host__ void disableVariableByHash(VariableHash variable_hash);
    /** @brief Template function for disabling access to a cuRVE variable from a constant string variable name
     *     Disables device access to the cuRVE variable. Does not disable host access.
     *  @param variableName A constant string variable name which must have been registered as a curve variable
     */
    template <unsigned int N>
    __host__ void disableVariable(const char(&variableName)[N]);
    /** @brief Function for enabling access to a cuRVE variable from a VariableHash
     *     Enables device access to the cuRVE variable.
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     */
    __host__ void enableVariableByHash(VariableHash variable_hash);
    /** @brief Template function for enabling access to a cuRVE variable from a constant string variable name
     *     Enables device access to the cuRVE variable.
     *  @param variableName A constant string variable name which must have been registered as a curve variable
     */
    template <unsigned int N>
    __host__ void enableVariable(const char(&variableName)[N]);
    /** @brief Function changes the current namespace from a NamespaceHash
     *     Changing the namespace will affect both the host and device.
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     */
    __host__ void setNamespaceByHash(NamespaceHash variable_hash);
    /** @brief Function changes the current namespace to the default empty namespace
     *     Changing the namespace will affect both the host and device.
     */
    __host__ void setDefaultNamespace();
    /** @brief Template function changes the current namespace using a constant string namespace name
     *     Changing the namespace will affect both the host and device.
     *  @param namespaceName A constant string variable name which must have been registered as a curve variable
     */
    template <unsigned int N>
    __host__ void setNamespace(const char(&namespaceName)[N]);



    __device__ __forceinline__ static Variable getVariable(const VariableHash variable_hash);
    /** @brief Gets the size of the cuRVE variable type given the variable hash
     * Gets the size of the cuRVE variable type given the variable hash
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     *  @return A size_t which is the size of the variable or 0 otherwise
     */
    __device__ __forceinline__ static size_t getVariableSize(const VariableHash variable_hash);
    /** @brief Gets the length of the cuRVE variable given the variable hash
     *  Gets the length of the cuRVE variable given the variable hash
     *  This will be 1 unless the variable is an array
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     *  @return An unsigned int which is the number of elements within the curve variable (1 unless it's an array)
     */
    __device__ __forceinline__ static unsigned int getVariableLength(const VariableHash variable_hash);
    /** @brief Device function for getting a pointer to a variable of given name
     * Returns a generic pointer to a variable of given name at a specific offset in bytes from the start of the variable array.
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     *  @param offset an offset into the variable array in bytes (offset is variable index * sizeof(variable type))
     *  @return A generic pointer to the variable value. Will be nullptr if there is an error.
     */
    __device__ __forceinline__ static void* getVariablePtrByHash(const VariableHash variable_hash, size_t offset);
    /** @brief Device function for getting a single typed value from a VariableHash at a given index
     *     Returns a single value of specified type from a variableHash using the given index position.
     * Note: No runtime type checking is done. TODO: Add runtime type checking for debug modes.
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     *  @param index The index of the variable in the named variable vector.
     *  @return T A value of given type at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     */
    template <typename T>
    __device__ __forceinline__ static float getVariableByHash(const VariableHash variable_hash, unsigned int index);
    /** @brief Template device function for getting a single typed value from a constant string variable name
     *     Returns a single value from a constant string expression using the given index position
     *  @param variableName A constant string variable name which must have been registered as a cuRVE variable.
     *  @param index The index of the variable in the named variable vector
     *  @return T A value of given typr at the given index for the variable with the provided hash. Will return 0 if an error is raised.
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static float getVariable(const char(&variableName)[N], VariableHash namespace_hash, unsigned int index);
    /** @brief Device function for setting a single typed value from a VariableHash
     *     Sets a single value from a variableHash using the given index position.
     *  @param variable_hash A cuRVE variable string hash from VariableHash.
     *  @param index The index of the variable in the named variable vector.
     *  @param value The typed value to set at the given index.
     */
    template <typename T>
    __device__ __forceinline__ static void setVariableByHash(const VariableHash variable_hash, T variable, unsigned int index);
    /** @brief Device template function for getting a setting a single typed value from a constant string variable name
     *     Returns a single value from a constant string expression using the given index position
     *  @param variableName A constant string variable name which must have been registered as a curve variable.
     *  @param index The index of the variable in the named variable vector
     *  @param value The typed value to set at the given index.
     */
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index);

    /* ERROR CHECKING API FUNCTIONS */

    /** @brief Device function for printing the last device error
    *     Prints the last device error using the provided source location information. The preferred method for printing is to use the curveReportLastDeviceError macro which inserts source location information.
    *  @param file A constant string filename.
    *  @param function A constant string function name.
    *  @param line A constant integer line number.
    */
    __device__ __forceinline__ static void printLastDeviceError(const char* file, const char* function, const int line);
    /** @brief Host API function for printing the last host error
     *     Prints the last host API error using the provided source location information. The preferred method for printing is to use the curveReportLastHostError macro which inserts source location information.
     *  @param file A constant string filename.
     *  @param function A constant string function name.
     *  @param line A constant integer line number.
     */
    void __host__ printLastHostError(const char* file, const char* function, const int line);

    /** @brief Host API function for printing the last host or device error
     *     Prints the last device or host API error (or both) using the provided source location information. The preferred method for printing is to use the curveReportErrors macro which inserts source location information.
     *  @param file A constant string filename.
     *  @param function A constant string function name.
     *  @param line A constant integer line number.
     */
    void __host__ printErrors(const char* file, const char* function, const int line);
    /** @brief Device API function for returning a constant string error description
     *     Returns an error description given a DeviceError error code.
     *  @param error_code A DeviceError error code.
     *  @return constant A string error description
     */
    __device__ __host__ __forceinline__ static const char*  getDeviceErrorString(DeviceError error_code);
    /** @brief Host API function for returning a constant string error description
     *     Returns an error description given a HostError error code.
     *  @param error_code A HostError error code.
     *  @return constant A string error description
     */
    __host__ const char*  getHostErrorString(HostError error_code);
    /** @brief Device API function for returning the last reported error code
     *  @return A DeviceError error code
     */
    __device__ __forceinline__ static DeviceError getLastDeviceError();

    /** @brief Host API function for returning the last reported error code
     *  @return A HostError error code
     */
    __host__ HostError getLastHostError();
    /** @brief Host API function for clearing both the device and host error codes
    */
    __host__ void clearErrors();

    __host__ unsigned int checkHowManyMappedItems();
    unsigned int h_namespace;
    static const int MAX_VARIABLES = 32;          // !< Default maximum number of cuRVE variables (must be a power of 2)
    static const int VARIABLE_DISABLED = 0;
    static const int VARIABLE_ENABLED = 1;
    static const int NAMESPACE_NONE = 0;
    static const VariableHash EMPTY_FLAG = 0;
    static const VariableHash DELETED_FLAG = 1;

 private:
    VariableHash h_hashes[MAX_VARIABLES];         // Host array of the hash values of registered variables
    void* h_d_variables[MAX_VARIABLES];           // Host array of pointer to device memory addresses for variable storage
    int    h_states[MAX_VARIABLES];               // Host array of the states of registered variables
    size_t h_sizes[MAX_VARIABLES];                // Host array of the sizes of registered variable types (Note: RTTI not supported in CUDA so this is the best we can do for now)
    unsigned int h_lengths[MAX_VARIABLES];        // Host array of the length of registered variables (i.e: vector length)
    bool deviceInitialised;                       // Flag indicating that curve has/hasn't been initialised yet on a device.

    /**
     * Initialises cuRVE on the currently active device.
     * 
     * @note - not yet aware if the device has been reset.
     * @todo - Need to add a device-side check for initialisation.
     */
    void initialiseDevice();

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
    extern __constant__ Curve::NamespaceHash d_namespace;
    extern __constant__ Curve::VariableHash d_hashes[Curve::MAX_VARIABLES];   // Device array of the hash values of registered variables
    extern __device__ char* d_variables[Curve::MAX_VARIABLES];                // Device array of pointer to device memory addresses for variable storage
    extern __constant__ int d_states[Curve::MAX_VARIABLES];                   // Device array of the states of registered variables
    extern  __constant__ size_t d_sizes[Curve::MAX_VARIABLES];                // Device array of the types of registered variables
    extern __constant__ unsigned int d_lengths[Curve::MAX_VARIABLES];

    extern __device__ Curve::DeviceError d_curve_error;
    extern Curve::HostError h_curve_error;
}  // namespace curve_internal


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

/**
 * Host side template class implementation
 */
template <unsigned int N, typename T>
__host__ Curve::Variable Curve::registerVariable(const char(&variableName)[N], void* d_ptr, unsigned int length) {
    VariableHash variable_hash = variableHash(variableName);
    size_t size = sizeof(T);
    return registerVariableByHash(variable_hash, d_ptr, size, length);  // the const func can get const and non const argument (for 3rd argument)
}
template <unsigned int N>
__host__ void Curve::unregisterVariable(const char(&variableName)[N]) {
    VariableHash variable_hash = variableHash(variableName);
    unregisterVariableByHash(variable_hash);
}
template <unsigned int N>
__host__ void Curve::disableVariable(const char (&variableName)[N]) {
    VariableHash variable_hash = variableHash(variableName);
    disableVariableByHash(variable_hash);
}
template <unsigned int N>
__host__ void Curve::enableVariable(const char (&variableName)[N]) {
    VariableHash variable_hash = variableHash(variableName);
    enableVariableByHash(variable_hash);
}
template <unsigned int N>
__host__ void Curve::setNamespace(const char (&namespaceName)[N]) {
    NamespaceHash namespace_hash = variableHash(namespaceName);
    setNamespaceByHash(namespace_hash);
}

/**
* Device side class implementation
*/
/* loop unrolling of hash collision detection */
__device__ __forceinline__ Curve::Variable Curve::getVariable(const VariableHash variable_hash) {
    const VariableHash hash = variable_hash + curve_internal::d_namespace;
    for (unsigned int x = 0; x< MAX_VARIABLES; x++) {
        const Variable i = ((hash + x) & (MAX_VARIABLES - 1));
        const VariableHash h = curve_internal::d_hashes[i];
        if (h == hash)
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

    return curve_internal::d_sizes[cv];
}
__device__ __forceinline__ unsigned int Curve::getVariableLength(const VariableHash variable_hash) {
    Variable cv;

    cv = getVariable(variable_hash);

    return curve_internal::d_lengths[cv];
}
__device__ __forceinline__ void* Curve::getVariablePtrByHash(const VariableHash variable_hash, size_t offset) {
    Variable cv;

    cv = getVariable(variable_hash);
    // error checking
    if (cv == UNKNOWN_VARIABLE) {
        curve_internal::d_curve_error = DEVICE_ERROR_UNKNOWN_VARIABLE;
        return nullptr;
    }
    if (!curve_internal::d_states[cv]) {
        curve_internal::d_curve_error = DEVICE_ERROR_VARIABLE_DISABLED;
        return nullptr;
    }

    // check vector length
    if (offset > curve_internal::d_sizes[cv] * curve_internal::d_lengths[cv]) {  // Note : offset is basicly index * sizeof(T)
        curve_internal::d_curve_error = DEVICE_ERROR_UNKNOWN_LENGTH;
        return nullptr;
    }
    // return a generic pointer to variable address for given offset (no bounds checking here!)
    return curve_internal::d_variables[cv] + offset;
}
template <typename T>
__device__ __forceinline__ float Curve::getVariableByHash(const VariableHash variable_hash, unsigned int index) {
    size_t offset = index *sizeof(T);

    // do a check on the size as otherwise the value_ptr may eb out of bounds.
    size_t size = getVariableSize(variable_hash);

    // error checking
    if (size != sizeof(T)) {
        curve_internal::d_curve_error = DEVICE_ERROR_UNKNOWN_TYPE;
        return NULL;
    } else {
        // get a pointer to the specific variable by offsetting by the provided index
        T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));

        if (value_ptr)
            return *value_ptr;
        else
            return 0;
    }
}
template <typename T, unsigned int N>
__device__ __forceinline__ float Curve::getVariable(const char (&variableName)[N], VariableHash namespace_hash, unsigned int index) {
    VariableHash variable_hash = variableHash(variableName);

    return getVariableByHash<T>(variable_hash+namespace_hash, index);
}
template <typename T>
__device__ __forceinline__ void Curve::setVariableByHash(const VariableHash variable_hash, T variable, unsigned int index) {
    size_t size = getVariableSize(variable_hash);

    if (size != sizeof(T)) {
        curve_internal::d_curve_error = DEVICE_ERROR_UNKNOWN_TYPE;
    } else {
        size_t offset = index *sizeof(T);
        T *value_ptr = reinterpret_cast<T*>(getVariablePtrByHash(variable_hash, offset));
        *value_ptr = variable;
    }
}
template <typename T, unsigned int N>
__device__ __forceinline__ void Curve::setVariable(const char(&variableName)[N], VariableHash namespace_hash, T variable, unsigned int index) {
    VariableHash variable_hash = variableHash(variableName);
    setVariableByHash<T>(variable_hash+namespace_hash, variable, index);
}

/* ERROR CHECKING API FUNCTIONS */
#define curveReportLastDeviceError() { Curve::curvePrintLastDeviceError(__FILE__, __FUNCTION__, __LINE__); }    // ! Prints the last reported device error using the file, function and line number of the call to this macro
#define curveReportLastHostError() { curvePrintLastHostError(__FILE__, __FUNCTION__, __LINE__); }        // ! Prints the last reported host API error using the file, function and line number of the call to this macro
#define curveReportErrors() { curvePrintErrors(__FILE__, __FUNCTION__, __LINE__); }  // ! Prints the last reported device or host API error using the file, function and line number of the call to this macro

__device__ __forceinline__ void Curve::printLastDeviceError(const char* file, const char* function, const int line) {
    if (curve_internal::d_curve_error != DEVICE_ERROR_NO_ERRORS) {
        printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)curve_internal::d_curve_error, getDeviceErrorString(curve_internal::d_curve_error));
    }
}
__device__ __host__ __forceinline__ const char* Curve::getDeviceErrorString(DeviceError e) {
    switch (e) {
    case(DEVICE_ERROR_NO_ERRORS):
        return "No cuRVE errors";
    case(DEVICE_ERROR_UNKNOWN_VARIABLE):
        return "Unknown cuRVE variable in current namespace";
    case(DEVICE_ERROR_VARIABLE_DISABLED):
        return "cuRVE variable is disabled";
    default:
        return "Unspecified cuRVE error";
    }
}
__device__ __forceinline__ Curve::DeviceError Curve::getLastDeviceError() {
    return curve_internal::d_curve_error;
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVE_H_

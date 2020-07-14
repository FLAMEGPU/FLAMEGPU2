#include <cuda_runtime.h>

#include <cstdio>
#include <cassert>

#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

namespace curve_internal {
    __constant__ Curve::NamespaceHash d_namespace;
    __constant__ Curve::VariableHash d_hashes[Curve::MAX_VARIABLES];  // Device array of the hash values of registered variables
    __device__ char* d_variables[Curve::MAX_VARIABLES];               // Device array of pointer to device memory addresses for variable storage
    __constant__ int d_states[Curve::MAX_VARIABLES];                  // Device array of the states of registered variables
    __constant__ size_t d_sizes[Curve::MAX_VARIABLES];                // Device array of the types of registered variables
    __constant__ unsigned int d_lengths[Curve::MAX_VARIABLES];        // Device array of the length of registered variables (i.e: vector length)

    __device__ Curve::DeviceError d_curve_error;
    Curve::HostError h_curve_error;
}  // namespace curve_internal

/* header implementations */
__host__ Curve::Curve() :
    deviceInitialised(false) {
    // Initialise some host variables.
    curve_internal::h_curve_error  = ERROR_NO_ERRORS;
}
__host__ void Curve::purge() {
    deviceInitialised = false;
    curve_internal::h_curve_error = ERROR_NO_ERRORS;
    initialiseDevice();
}
__host__ void Curve::initialiseDevice() {
    if (!deviceInitialised) {
        unsigned int *_d_hashes;
        char** _d_variables;
        int* _d_states;
        unsigned int* _d_lengths;
        size_t* _d_sizes;

        // namespace
        h_namespace = NAMESPACE_NONE;
        gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_namespace, &h_namespace, sizeof(unsigned int)));

        // get a host pointer to d_hashes and d_variables
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_hashes), curve_internal::d_hashes));
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_variables), curve_internal::d_variables));
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_states), curve_internal::d_states));
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_lengths), curve_internal::d_lengths));
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_sizes), curve_internal::d_sizes));

        // set values of hash table to 0 on host and device
        memset(h_hashes, 0, sizeof(unsigned int)*MAX_VARIABLES);
        memset(h_lengths, 0, sizeof(unsigned int)*MAX_VARIABLES);
        memset(h_states, 0, sizeof(int)*MAX_VARIABLES);
        memset(h_sizes, 0, sizeof(size_t)*MAX_VARIABLES);

        // initialise data to 0 on device
        gpuErrchk(cudaMemset(_d_hashes, 0, sizeof(unsigned int)*MAX_VARIABLES));
        gpuErrchk(cudaMemset(_d_variables, 0, sizeof(void*)*MAX_VARIABLES));
        gpuErrchk(cudaMemset(_d_states, VARIABLE_DISABLED, sizeof(int)*MAX_VARIABLES));
        gpuErrchk(cudaMemset(_d_lengths, 0, sizeof(unsigned int)*MAX_VARIABLES));
        gpuErrchk(cudaMemset(_d_sizes, 0, sizeof(size_t)*MAX_VARIABLES));
    }
    deviceInitialised = true;
}

__host__ Curve::VariableHash Curve::variableRuntimeHash(const char* str) {
    const size_t length = std::strlen(str) + 1;
    unsigned int hash = 2166136261u;

    for (size_t i = 0; i < length; ++i) {
        hash ^= *str++;
        hash *= 16777619u;
    }
    return hash;
}
__host__ Curve::VariableHash Curve::variableRuntimeHash(unsigned int num) {
    return variableRuntimeHash(std::to_string(num).c_str());
}

__host__ Curve::Variable Curve::getVariableHandle(VariableHash variable_hash) {
    variable_hash += h_namespace;
    unsigned int n = 0;
    unsigned int i = (variable_hash) % MAX_VARIABLES;

    while (h_hashes[i] != EMPTY_FLAG) {
        if (h_hashes[i] == variable_hash) {
            return i;
        }
        n += 1;
        if (n >= MAX_VARIABLES) {
            break;
        }
        i += 1;
        if (i >= MAX_VARIABLES) {
            i = 0;
        }
    }
    return UNKNOWN_VARIABLE;
}

__host__ Curve::Variable Curve::registerVariableByHash(VariableHash variable_hash, void * d_ptr, size_t size, unsigned int length) {
    unsigned int n = 0;
    variable_hash += h_namespace;
    assert(variable_hash != EMPTY_FLAG);
    assert(variable_hash != DELETED_FLAG);
    unsigned int i = (variable_hash) % MAX_VARIABLES;

    while (h_hashes[i] != EMPTY_FLAG && h_hashes[i] != DELETED_FLAG) {
        n += 1;
        if (n >= MAX_VARIABLES) {
            curve_internal::h_curve_error = ERROR_TOO_MANY_VARIABLES;
            return UNKNOWN_VARIABLE;
        }
        i += 1;
        if (i >= MAX_VARIABLES) {
            i = 0;
        }
    }
    h_hashes[i] = variable_hash;

    // make a host copy of the pointer
    h_d_variables[i] = d_ptr;

    // set the state to enabled
    h_states[i] = VARIABLE_ENABLED;

    // set the size of the data type
    h_sizes[i] = size;

    // set the length of variable
    h_lengths[i] = length;

    return i;
}

/**
 * TODO: Does un-registering imply that other variable with collisions will no longer be found. I.e. do you need to re-register all other variable when one is removed.
 */
__host__ void Curve::unregisterVariableByHash(VariableHash variable_hash) {
    // get the curve variable
    Variable cv = getVariableHandle(variable_hash);

    // error checking
    if (cv == UNKNOWN_VARIABLE) {
        curve_internal::h_curve_error = ERROR_UNKNOWN_VARIABLE;
        return;
    }

    // clear hash location on host and copy hash to device
    h_hashes[cv] = DELETED_FLAG;

    // set a host pointer to nullptr and copy to the device
    h_d_variables[cv] = 0;

    // return the state to disabled
    h_states[cv] = VARIABLE_DISABLED;

    // set the empty size to 0
    h_sizes[cv] = 0;

    // set the length of variable to 0
    h_lengths[cv] = 0;
}
__host__ void Curve::updateDevice() {
    // Initialise the device (if required)
    initialiseDevice();
    // Copy
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_hashes, h_hashes, sizeof(unsigned int) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_variables, h_d_variables, sizeof(void*) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_states, h_states, sizeof(int) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_sizes, h_sizes, sizeof(size_t) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_lengths, h_lengths, sizeof(unsigned int) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_namespace, &h_namespace, sizeof(unsigned int)));
}

__host__ void Curve::disableVariableByHash(VariableHash variable_hash) {
    Variable cv = getVariableHandle(variable_hash);

    // error checking
    if (cv == UNKNOWN_VARIABLE) {
        curve_internal::h_curve_error = ERROR_UNKNOWN_VARIABLE;
        return;
    }

    h_states[cv] = VARIABLE_DISABLED;
}
__host__ void Curve::enableVariableByHash(VariableHash variable_hash) {
    Variable cv = getVariableHandle(variable_hash);

    // error checking
    if (cv == UNKNOWN_VARIABLE) {
        curve_internal::h_curve_error = ERROR_UNKNOWN_VARIABLE;
        return;
    }

    h_states[cv] = VARIABLE_ENABLED;
}
__host__ void Curve::setNamespaceByHash(NamespaceHash namespace_hash) {
    h_namespace = namespace_hash;
}

__host__ void Curve::setDefaultNamespace() {
    h_namespace = NAMESPACE_NONE;
}

/* errors */
void __host__ Curve::printLastHostError(const char* file, const char* function, const int line) {
    if (curve_internal::h_curve_error != ERROR_NO_ERRORS) {
        printf("%s.%s.%d: cuRVE Host Error %d (%s)\n", file, function, line, (unsigned int)curve_internal::h_curve_error, getHostErrorString(curve_internal::h_curve_error));
    }
}

void __host__ Curve::printErrors(const char* file, const char* function, const int line) {
    // Initialise the device (if required)
    initialiseDevice();

    DeviceError d_curve_error_local;

    printLastHostError(file, function, line);

    // check device errors
    gpuErrchk(cudaMemcpyFromSymbol(&d_curve_error_local, curve_internal::d_curve_error, sizeof(DeviceError)));
    if (d_curve_error_local != DEVICE_ERROR_NO_ERRORS) {
        printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)d_curve_error_local, getDeviceErrorString(d_curve_error_local));
    }
}
__host__ const char* Curve::getHostErrorString(HostError e) {
    switch (e) {
    case(ERROR_NO_ERRORS):
        return "No cuRVE errors";
    case(ERROR_UNKNOWN_VARIABLE):
        return "Unknown cuRVE variable";
    case(ERROR_TOO_MANY_VARIABLES):
        return "Too many cuRVE variables";
    default:
        return "Unspecified cuRVE error";
    }
}
__host__ Curve::HostError Curve::getLastHostError() {
    return curve_internal::h_curve_error;
}
__host__ void Curve::clearErrors() {
    // Initialise the device (if required)
    initialiseDevice();

    DeviceError curve_error_none;

    curve_error_none = DEVICE_ERROR_NO_ERRORS;
    curve_internal::h_curve_error  = ERROR_NO_ERRORS;

    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_curve_error, &curve_error_none, sizeof(DeviceError)));
}

__host__ unsigned int Curve::checkHowManyMappedItems() {
    unsigned int rtn = 0;
    for (unsigned int i = 0; i < MAX_VARIABLES; ++i)
        if (h_hashes[i] != EMPTY_FLAG && h_hashes[i] != DELETED_FLAG)
            rtn++;
    return rtn;
}

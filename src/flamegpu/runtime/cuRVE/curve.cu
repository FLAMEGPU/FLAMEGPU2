#include <cstdio>
#include <cassert>

#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

namespace curve_internal {
    __constant__ Curve::CurveNamespaceHash d_namespace;
    __constant__ Curve::CurveVariableHash d_hashes[Curve::CURVE_MAX_VARIABLES];    // Device array of the hash values of registered variables
    __device__ char* d_variables[Curve::CURVE_MAX_VARIABLES];                // Device array of pointer to device memory addresses for variable storage
    __constant__ int d_states[Curve::CURVE_MAX_VARIABLES];                    // Device array of the states of registered variables
    __constant__ size_t d_sizes[Curve::CURVE_MAX_VARIABLES];                // Device array of the types of registered variables
    __constant__ unsigned int d_lengths[Curve::CURVE_MAX_VARIABLES];        // Device array of the length of registered variables (i.e: vector length)

    __device__ Curve::curveDeviceError d_curve_error;
    Curve::curveHostError h_curve_error;
}  // namespace curve_internal

/* header implementations */
__host__ Curve::Curve() {
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
    memset(h_hashes, 0, sizeof(unsigned int)*CURVE_MAX_VARIABLES);
    memset(h_lengths, 0, sizeof(unsigned int)*CURVE_MAX_VARIABLES);
    memset(h_states, 0, sizeof(int)*CURVE_MAX_VARIABLES);
    memset(h_sizes, 0, sizeof(size_t)*CURVE_MAX_VARIABLES);

    // initialise data to 0 on device
    gpuErrchk(cudaMemset(_d_hashes, 0, sizeof(unsigned int)*CURVE_MAX_VARIABLES));
    gpuErrchk(cudaMemset(_d_variables, 0, sizeof(void*)*CURVE_MAX_VARIABLES));
    gpuErrchk(cudaMemset(_d_states, VARIABLE_DISABLED, sizeof(int)*CURVE_MAX_VARIABLES));
    gpuErrchk(cudaMemset(_d_lengths, 0, sizeof(unsigned int)*CURVE_MAX_VARIABLES));
    gpuErrchk(cudaMemset(_d_sizes, 0, sizeof(size_t)*CURVE_MAX_VARIABLES));

    // memset the h and d types array

    curveClearErrors();
}

__host__ Curve::CurveVariableHash Curve::curveVariableRuntimeHash(const char* str) {
    const size_t length = std::strlen(str) + 1;
    unsigned int hash = 2166136261u;

    for (size_t i = 0; i < length; ++i) {
        hash ^= *str++;
        hash *= 16777619u;
    }
    return hash;
}

__host__ Curve::CurveVariable Curve::curveGetVariableHandle(CurveVariableHash variable_hash) {
    unsigned int i, n;

    variable_hash += h_namespace;
    n = 0;
    i = (variable_hash) % CURVE_MAX_VARIABLES;

    while (h_hashes[i] != 0) {
        if (h_hashes[i] == variable_hash) {
            return i;
        }
        n += 1;
        if (n >= CURVE_MAX_VARIABLES) {
            break;
        }
        i += 1;
        if (i >= CURVE_MAX_VARIABLES) {
            i = 0;
        }
    }
    return UNKNOWN_CURVE_VARIABLE;
}

__host__ Curve::CurveVariable Curve::curveRegisterVariableByHash(CurveVariableHash variable_hash, void * d_ptr, size_t size, unsigned int length) {
    unsigned int i, n;
    unsigned int *_d_hashes;
    void** _d_variables;
    int* _d_states;
    unsigned int *_d_lengths;
    size_t* _d_sizes;

    n = 0;
    variable_hash += h_namespace;
    i = (variable_hash) % CURVE_MAX_VARIABLES;

    while (h_hashes[i] != 0) {
        n += 1;
        if (n >= CURVE_MAX_VARIABLES) {
            curve_internal::h_curve_error = CURVE_ERROR_TOO_MANY_VARIABLES;
            return UNKNOWN_CURVE_VARIABLE;
        }
        i += 1;
        if (i >= CURVE_MAX_VARIABLES) {
            i = 0;
        }
    }
    h_hashes[i] = variable_hash;

    // get a host pointer to d_hashes and d_variables
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_hashes), curve_internal::d_hashes));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_variables), curve_internal::d_variables));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_states), curve_internal::d_states));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_lengths), curve_internal::d_lengths));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_sizes), curve_internal::d_sizes));

    // copy hash to device
    gpuErrchk(cudaMemcpy(&_d_hashes[i], &h_hashes[i], sizeof(unsigned int), cudaMemcpyHostToDevice));

    // make a host copy of the pointer and copy to the device
    h_d_variables[i] = d_ptr;
    gpuErrchk(cudaMemcpy(&_d_variables[i], &h_d_variables[i], sizeof(void*), cudaMemcpyHostToDevice));

    // set the state to enabled
    h_states[i] = VARIABLE_ENABLED;
    gpuErrchk(cudaMemcpy(&_d_states[i], &h_states[i], sizeof(int), cudaMemcpyHostToDevice));

    // set the size of the data type
    h_sizes[i] = size;
    gpuErrchk(cudaMemcpy(&_d_sizes[i], &h_sizes[i], sizeof(size_t), cudaMemcpyHostToDevice));

    // set the length of variable
    h_lengths[i] = length;
    gpuErrchk(cudaMemcpy(&_d_lengths[i], &h_lengths[i], sizeof(unsigned int), cudaMemcpyHostToDevice));

    return i;
}

/**
 * TODO: Does un-registering imply that other variable with collisions will no longer be found. I.e. do you need to re-register all other variable when one is removed.
 */
__host__ void Curve::curveUnregisterVariableByHash(CurveVariableHash variable_hash) {
    unsigned int *_d_hashes;
    void** _d_variables;
    int* _d_states;
    size_t* _d_sizes;
    unsigned int *_d_lengths;

    CurveVariable cv;

    // get the curve variable
    cv = curveGetVariableHandle(variable_hash);

    // error checking
    if (cv == UNKNOWN_CURVE_VARIABLE) {
        curve_internal::h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
        return;
    }

    // get a host pointer to d_hashes and d_variables
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_hashes), curve_internal::d_hashes));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_variables), curve_internal::d_variables));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_states), curve_internal::d_states));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_lengths), curve_internal::d_lengths));
    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_sizes), curve_internal::d_sizes));

    // clear hash location on host and copy hash to device
    h_hashes[cv] = 0;
    gpuErrchk(cudaMemcpy(&_d_hashes[cv], &h_hashes[cv], sizeof(unsigned int), cudaMemcpyHostToDevice));

    // set a host pointer to nullptr and copy to the device
    h_d_variables[cv] = 0;
    gpuErrchk(cudaMemcpy(&_d_variables[cv], &h_d_variables[cv], sizeof(void*), cudaMemcpyHostToDevice));

    // return the state to disabled
    h_states[cv] = VARIABLE_DISABLED;
    gpuErrchk(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));

    // set the empty size to 0
    h_sizes[cv] = 0;
    gpuErrchk(cudaMemcpy(&_d_sizes[cv], &h_sizes[cv], sizeof(size_t), cudaMemcpyHostToDevice));

    // set the length of variable to 0
    h_lengths[cv] = 0;
    gpuErrchk(cudaMemcpy(&_d_lengths[cv], &h_lengths[cv], sizeof(unsigned int), cudaMemcpyHostToDevice));
}

__host__ void Curve::curveDisableVariableByHash(CurveVariableHash variable_hash) {
    CurveVariable cv = curveGetVariableHandle(variable_hash);
    int* _d_states;

    // error checking
    if (cv == UNKNOWN_CURVE_VARIABLE) {
        curve_internal::h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
        return;
    }

    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_states), curve_internal::d_states));
    h_states[cv] = VARIABLE_DISABLED;
    gpuErrchk(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));
}
__host__ void Curve::curveEnableVariableByHash(CurveVariableHash variable_hash) {
    CurveVariable cv = curveGetVariableHandle(variable_hash);
    int* _d_states;

    // error checking
    if (cv == UNKNOWN_CURVE_VARIABLE) {
        curve_internal::h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
        return;
    }

    gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_states), curve_internal::d_states));
    h_states[cv] = VARIABLE_ENABLED;
    gpuErrchk(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));
}
__host__ void Curve::curveSetNamespaceByHash(CurveNamespaceHash namespace_hash) {
    h_namespace = namespace_hash;
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_namespace, &h_namespace, sizeof(unsigned int)));
}

__host__ void Curve::curveSetDefaultNamespace() {
    h_namespace = NAMESPACE_NONE;
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_namespace, &h_namespace, sizeof(unsigned int)));
}

/* errors */
void __host__ Curve::curvePrintLastHostError(const char* file, const char* function, const int line) {
    if (curve_internal::h_curve_error != CURVE_ERROR_NO_ERRORS) {
        printf("%s.%s.%d: cuRVE Host Error %d (%s)\n", file, function, line, (unsigned int)curve_internal::h_curve_error, curveGetHostErrorString(curve_internal::h_curve_error));
    }
}

void __host__ Curve::curvePrintErrors(const char* file, const char* function, const int line) {
    curveDeviceError d_curve_error_local;

    curvePrintLastHostError(file, function, line);

    // check device errors
    gpuErrchk(cudaMemcpyFromSymbol(&d_curve_error_local, curve_internal::d_curve_error, sizeof(curveDeviceError)));
    if (d_curve_error_local != CURVE_DEVICE_ERROR_NO_ERRORS) {
        printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)d_curve_error_local, curveGetDeviceErrorString(d_curve_error_local));
    }
}
__host__ const char* Curve::curveGetHostErrorString(curveHostError e) {
    switch (e) {
    case(CURVE_ERROR_NO_ERRORS):
        return "No cuRVE errors";
    case(CURVE_ERROR_UNKNOWN_VARIABLE):
        return "Unknown cuRVE variable";
    case(CURVE_ERROR_TOO_MANY_VARIABLES):
        return "Too many cuRVE variables";
    default:
        return "Unspecified cuRVE error";
    }
}
__host__ Curve::curveHostError Curve::curveGetLastHostError() {
    return curve_internal::h_curve_error;
}
__host__ void Curve::curveClearErrors() {
    curveDeviceError curve_error_none;

    curve_error_none = CURVE_DEVICE_ERROR_NO_ERRORS;
    curve_internal::h_curve_error  = CURVE_ERROR_NO_ERRORS;

    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_curve_error, &curve_error_none, sizeof(curveDeviceError)));
}


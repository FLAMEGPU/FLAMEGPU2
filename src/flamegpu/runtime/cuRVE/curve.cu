#include <cuda_runtime.h>

#include <cstdio>
#include <cassert>
#include <map>
#include <memory>

#include "flamegpu/runtime/cuRVE/curve.h"


#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/util/nvtx.h"

namespace curve_internal {
    __constant__ Curve::VariableHash d_hashes[Curve::MAX_VARIABLES];  // Device array of the hash values of registered variables
    __device__ char* d_variables[Curve::MAX_VARIABLES];               // Device array of pointer to device memory addresses for variable storage
    __constant__ size_t d_sizes[Curve::MAX_VARIABLES];                // Device array of the types of registered variables
    __constant__ unsigned int d_lengths[Curve::MAX_VARIABLES];        // Device array of the length of registered variables (i.e: vector length)

    __device__ Curve::DeviceError d_curve_error;
    Curve::HostError h_curve_error;
}  // namespace curve_internal

std::mutex Curve::instance_mutex;

/* header implementations */
__host__ Curve::Curve() :
    deviceInitialised(false) {
    // Initialise some host variables.
    curve_internal::h_curve_error  = ERROR_NO_ERRORS;
}
__host__ void Curve::purge() {
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
    deviceInitialised = false;
    curve_internal::h_curve_error = ERROR_NO_ERRORS;
    initialiseDevice();
}
__host__ void Curve::initialiseDevice() {
    // Don't lock mutex here, do it in the calling method
    if (!deviceInitialised) {
        unsigned int *_d_hashes;
        char** _d_variables;
        unsigned int* _d_lengths;
        size_t* _d_sizes;

        // get a host pointer to d_hashes and d_variables
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_hashes), curve_internal::d_hashes));
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_variables), curve_internal::d_variables));
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_lengths), curve_internal::d_lengths));
        gpuErrchk(cudaGetSymbolAddress(reinterpret_cast<void **>(&_d_sizes), curve_internal::d_sizes));

        // set values of hash table to 0 on host and device
        memset(h_hashes, 0, sizeof(unsigned int)*MAX_VARIABLES);
        memset(h_lengths, 0, sizeof(unsigned int)*MAX_VARIABLES);
        memset(h_sizes, 0, sizeof(size_t)*MAX_VARIABLES);

        // initialise data to 0 on device
        gpuErrchk(cudaMemset(_d_hashes, 0, sizeof(unsigned int)*MAX_VARIABLES));
        gpuErrchk(cudaMemset(_d_variables, 0, sizeof(void*)*MAX_VARIABLES));
        gpuErrchk(cudaMemset(_d_lengths, 0, sizeof(unsigned int)*MAX_VARIABLES));
        gpuErrchk(cudaMemset(_d_sizes, 0, sizeof(size_t)*MAX_VARIABLES));
    }
    deviceInitialised = true;
}

__host__ Curve::VariableHash Curve::variableRuntimeHash(const char* str) {
    // Method is static, does not require mutex
    const size_t length = std::strlen(str) + 1;
    unsigned int hash = 2166136261u;

    for (size_t i = 0; i < length; ++i) {
        hash ^= *str++;
        hash *= 16777619u;
    }
    return hash;
}
__host__ Curve::VariableHash Curve::variableRuntimeHash(unsigned int num) {
    // Method is static, does not require mutex
    return variableRuntimeHash(std::to_string(num).c_str());
}

__host__ Curve::Variable Curve::getVariableHandle(VariableHash variable_hash) {
    // Method is static, does not require mutex
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
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
    return _registerVariableByHash(variable_hash, d_ptr, size, length);
}
__host__ Curve::Variable Curve::_registerVariableByHash(VariableHash variable_hash, void * d_ptr, size_t size, unsigned int length) {
    // Do not lock mutex here, do it in the calling method
    unsigned int n = 0;
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

    // set the size of the data type
    h_sizes[i] = size;

    // set the length of variable
    h_lengths[i] = length;

    return i;
}
__host__ int Curve::size() const {
    auto lock = std::shared_lock<std::shared_timed_mutex>(mutex);
    return _size();
}
__host__ int Curve::_size() const {
    int rtn = 0;
    for (unsigned int hash : h_hashes) {
        if (hash != EMPTY_FLAG && hash != DELETED_FLAG)
            rtn++;
    }
    return rtn;
}
/**
 * TODO: Does un-registering imply that other variable with collisions will no longer be found. I.e. do you need to re-register all other variable when one is removed.
 */
__host__ void Curve::unregisterVariableByHash(VariableHash variable_hash) {
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
    _unregisterVariableByHash(variable_hash);
}
__host__ void Curve::_unregisterVariableByHash(VariableHash variable_hash) {
    // Do not lock mutex here, do it in the calling method
    // get the curve variable
    Variable cv = getVariableHandle(variable_hash);

    // error checking
    if (cv == UNKNOWN_VARIABLE) {
        THROW CurveException("Cannot unregister '%u', hash not found within curve table.", variable_hash);
    }

    // clear hash location on host and copy hash to device
    h_hashes[cv] = DELETED_FLAG;

    // set a host pointer to nullptr and copy to the device
    h_d_variables[cv] = 0;

    // set the empty size to 0
    h_sizes[cv] = 0;

    // set the length of variable to 0
    h_lengths[cv] = 0;
}
__host__ void Curve::updateDevice() {
    auto lock = std::shared_lock<std::shared_timed_mutex>(mutex);
    NVTX_RANGE("Curve::updateDevice()");
    // Initialise the device (if required)
    assert(deviceInitialised);  // No reason for this to ever fail. Purge calls init device
    // Copy
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_hashes, h_hashes, sizeof(unsigned int) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_variables, h_d_variables, sizeof(void*) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_sizes, h_sizes, sizeof(size_t) * MAX_VARIABLES));
    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_lengths, h_lengths, sizeof(unsigned int) * MAX_VARIABLES));
}

/* errors */
void __host__ Curve::printLastHostError(const char* file, const char* function, const int line) {
    // Do not lock mutex here, do it in the calling method
    if (curve_internal::h_curve_error != ERROR_NO_ERRORS) {
        printf("%s.%s.%d: cuRVE Host Error %d (%s)\n", file, function, line, (unsigned int)curve_internal::h_curve_error, getHostErrorString(curve_internal::h_curve_error));
    }
}

void __host__ Curve::printErrors(const char* file, const char* function, const int line) {
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
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
    // Do not lock mutex here, do it in the calling method
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
    auto lock = std::shared_lock<std::shared_timed_mutex>(mutex);
    return curve_internal::h_curve_error;
}
__host__ void Curve::clearErrors() {
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
    // Initialise the device (if required)
    initialiseDevice();

    DeviceError curve_error_none;

    curve_error_none = DEVICE_ERROR_NO_ERRORS;
    curve_internal::h_curve_error  = ERROR_NO_ERRORS;

    gpuErrchk(cudaMemcpyToSymbol(curve_internal::d_curve_error, &curve_error_none, sizeof(DeviceError)));
}

__host__ unsigned int Curve::checkHowManyMappedItems() {
    auto lock = std::shared_lock<std::shared_timed_mutex>(mutex);
    unsigned int rtn = 0;
    for (unsigned int i = 0; i < MAX_VARIABLES; ++i)
        if (h_hashes[i] != EMPTY_FLAG && h_hashes[i] != DELETED_FLAG)
            rtn++;
    return rtn;
}
Curve& Curve::getInstance() {
    auto lock = std::unique_lock<std::mutex>(instance_mutex);  // Mutex to protect from two threads triggering the static instantiation concurrently
    static std::map<int, std::unique_ptr<Curve>> instances = {};  // Instantiated on first use.
    int device_id = -1;
    gpuErrchk(cudaGetDevice(&device_id));
    // Can't use operator[] here, constructor is private
    const auto f = instances.find(device_id);
    if (f != instances.end())
        return *f->second;
    return *(instances.emplace(device_id, std::unique_ptr<Curve>(new Curve())).first->second);
}

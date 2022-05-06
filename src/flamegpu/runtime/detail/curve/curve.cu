#include <cuda_runtime.h>

#include <cstdio>
#include <cassert>
#include <map>
#include <memory>

#include "flamegpu/runtime/detail/curve/curve.cuh"


#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/util/nvtx.h"

namespace flamegpu {
namespace detail {
namespace curve {

std::mutex Curve::instance_mutex;

/* header implementations */
__host__ Curve::Curve()
    : deviceInitialised(false)
    , d_curve_table(nullptr) {
}
__host__ Curve::~Curve() {
    if (d_curve_table) {
        gpuErrchk(cudaFree(d_curve_table));
        d_curve_table = nullptr;
    }
}
__host__ void Curve::purge(cudaStream_t stream) {
    auto lock = std::unique_lock<std::shared_timed_mutex>(mutex);
    if (d_curve_table) {
        // gpuErrchk(cudaFree(d_curve_table));  // This fails if called after device reset
        d_curve_table = nullptr;
    }
    deviceInitialised = false;
    initialiseDevice(stream);
}
__host__ void Curve::initialiseDevice(cudaStream_t stream) {
    // Don't lock mutex here, do it in the calling method
    if (!deviceInitialised) {
        // get a host pointer to d_hashes and d_variables
        gpuErrchk(cudaMalloc(&d_curve_table, sizeof(CurveTable)));

        // set values of hash table to 0 on host and device
        memset(h_curve_table.hashes, 0, sizeof(unsigned int)*MAX_VARIABLES);
        memset(h_curve_table.lengths, 0, sizeof(unsigned int)*MAX_VARIABLES);
        memset(h_curve_table.sizes, 0, sizeof(size_t)*MAX_VARIABLES);

        // initialise data to 0 on device
        gpuErrchk(cudaMemsetAsync(d_curve_table, 0, sizeof(CurveTable), stream));
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

    while (h_curve_table.hashes[i] != EMPTY_FLAG) {
        if (h_curve_table.hashes[i] == variable_hash) {
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
    while (h_curve_table.hashes[i] != EMPTY_FLAG && h_curve_table.hashes[i] != DELETED_FLAG) {
        n += 1;
        if (n >= MAX_VARIABLES) {
            return UNKNOWN_VARIABLE;
        }
        i += 1;
        if (i >= MAX_VARIABLES) {
            i = 0;
        }
    }

    h_curve_table.hashes[i] = variable_hash;

    // make a host copy of the pointer
    h_curve_table.variables[i] = reinterpret_cast<char*>(d_ptr);

    // set the size of the data type
    h_curve_table.sizes[i] = size;

    // set the length of variable
    h_curve_table.lengths[i] = length;

    return i;
}
__host__ int Curve::size() const {
    auto lock = std::shared_lock<std::shared_timed_mutex>(mutex);
    return _size();
}
__host__ int Curve::_size() const {
    int rtn = 0;
    for (unsigned int hash : h_curve_table.hashes) {
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
        THROW exception::CurveException("Cannot unregister '%u', hash not found within curve table.", variable_hash);
    }

    // clear hash location on host and copy hash to device
    h_curve_table.hashes[cv] = DELETED_FLAG;

    // set a host pointer to nullptr and copy to the device
    h_curve_table.variables[cv] = 0;

    // set the empty size to 0
    h_curve_table.sizes[cv] = 0;

    // set the length of variable to 0
    h_curve_table.lengths[cv] = 0;
}
__host__ void Curve::updateDevice(const cudaStream_t stream) {
    auto lock = std::shared_lock<std::shared_timed_mutex>(mutex);
    NVTX_RANGE("Curve::updateDevice()");
    // Initialise the device (if required)
    assert(deviceInitialised);  // No reason for this to ever fail. Purge calls init device
    // Copy
    gpuErrchk(cudaMemcpyAsync(d_curve_table, &h_curve_table, sizeof(CurveTable), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
}
__host__ const Curve::CurveTable *Curve::getDevicePtr() const {
    return d_curve_table;
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

}  // namespace curve
}  // namespace detail
}  // namespace flamegpu

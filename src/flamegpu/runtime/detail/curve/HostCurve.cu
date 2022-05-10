#include <cuda_runtime.h>

#include <cstdio>
#include <cassert>
#include <map>
#include <memory>

#include "flamegpu/runtime/detail/curve/HostCurve.cuh"


#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/util/nvtx.h"

namespace flamegpu {
namespace detail {
namespace curve {

HostCurve::HostCurve()
    : d_curve_table(nullptr)
    , h_curve_table_ext_count{}
    , message_in_hash(Curve::variableRuntimeHash("_message_in"))
    , message_out_hash(Curve::variableRuntimeHash("_message_out"))
    , agent_out_hash(Curve::variableRuntimeHash("_agent_birth"))
    , environment_hash(Curve::variableRuntimeHash("_environment"))
    , macro_environment_hash(Curve::variableRuntimeHash("_macro_environment")) {
    //, h_curve_table_ext_type{std::type_index(typeid(void))}
    // std::fill_n(h_curve_table_ext_type, MAX_VARIABLES, typeid(void));

    // set values of hash table to 0 on host and device
    memset(h_curve_table.hashes, 0, sizeof(VariableHash)* MAX_VARIABLES);
    memset(h_curve_table.type_size, 0, sizeof(unsigned int)* MAX_VARIABLES);
    memset(h_curve_table.elements, 0, sizeof(unsigned int)* MAX_VARIABLES);
    memset(h_curve_table.count, 0, sizeof(unsigned int)* MAX_VARIABLES);
    initialiseDevice();
}
HostCurve::~HostCurve() {
    if (d_curve_table) {
        gpuErrchk(cudaFree(d_curve_table));
        d_curve_table = nullptr;
    }
}
void HostCurve::purge() {
    if (d_curve_table) {
        // gpuErrchk(cudaFree(d_curve_table));  // This fails if called after device reset
        d_curve_table = nullptr;
    }
    // set values of hash table to 0 on host and device
    memset(h_curve_table.hashes, 0, sizeof(VariableHash)* MAX_VARIABLES);
    memset(h_curve_table.type_size, 0, sizeof(unsigned int)* MAX_VARIABLES);
    memset(h_curve_table.elements, 0, sizeof(unsigned int)* MAX_VARIABLES);
    memset(h_curve_table.count, 0, sizeof(unsigned int)* MAX_VARIABLES);
    initialiseDevice();
}
void HostCurve::initialiseDevice() {
    // Don't lock mutex here, do it in the calling method
    if (!d_curve_table) {
        // get a host pointer to d_hashes and d_variables
        gpuErrchk(cudaMalloc(&d_curve_table, sizeof(CurveTable)));
    }
}

void HostCurve::registerAgentVariable(const std::string& variable_name, const std::type_index type, const size_t type_size, const unsigned int elements) {
    registerVariable(Curve::variableRuntimeHash(variable_name), type, type_size, elements);
}
void HostCurve::registerMessageInputVariable(const std::string& variable_name, const std::type_index type, const size_t type_size, const unsigned int elements) {
    registerVariable(message_in_hash + Curve::variableRuntimeHash(variable_name), type, type_size, elements);
}
void HostCurve::registerMessageOutputVariable(const std::string& variable_name, const std::type_index type, const size_t type_size, const unsigned int elements) {
    registerVariable(message_out_hash + Curve::variableRuntimeHash(variable_name), type, type_size, elements);
}
void HostCurve::registerAgentOutputVariable(const std::string& variable_name, const std::type_index type, const size_t type_size, const unsigned int elements) {
    registerVariable(agent_out_hash + Curve::variableRuntimeHash(variable_name), type, type_size, elements);
}
void HostCurve::registerSetEnvironmentProperty(const std::string& variable_name, const std::type_index type, const size_t type_size, const unsigned int elements, const ptrdiff_t offset) {
    registerVariable(environment_hash + Curve::variableRuntimeHash(variable_name), type, type_size, elements);
    setVariable(environment_hash + Curve::variableRuntimeHash(variable_name), reinterpret_cast<void *>(offset), 1);
}
void HostCurve::registerSetMacroEnvironmentProperty(const std::string& variable_name, std::type_index type, size_t type_size, unsigned int elements, void* d_ptr) {
    registerVariable(macro_environment_hash + Curve::variableRuntimeHash(variable_name), type, type_size, elements);
    setVariable(macro_environment_hash + Curve::variableRuntimeHash(variable_name), d_ptr, 1);
}
void HostCurve::registerVariable(const VariableHash variable_hash, const std::type_index type, const size_t type_size, const unsigned int elements) {
    if (variable_hash == EMPTY_FLAG) {
        THROW exception::CurveException("Unable to register variable, it's hash matches a reserved symbol!");
    }
    unsigned int i = (variable_hash) % MAX_VARIABLES;
    unsigned int n = 0;
    while (h_curve_table.hashes[i] != EMPTY_FLAG) {
        n += 1;
        if (n >= MAX_VARIABLES) {
            THROW exception::CurveException("Unable to register variable, hash table capacity exceeded!");
        }
        i += 1;
        if (i >= MAX_VARIABLES) {
            i = 0;
        }
    }

    h_curve_table.hashes[i] = variable_hash;

    // Initialise the pointer to 0
    h_curve_table.variables[i] = nullptr;

    // set the size of the data type
    h_curve_table.type_size[i] = static_cast<unsigned int>(type_size);

    // set the length of variable
    h_curve_table.elements[i] = elements;

    // set the type_index of the variable
    // h_curve_table_ext_type[i] = type; // @todo
}
void HostCurve::setAgentVariable(const std::string& variable_name, void* d_ptr, const unsigned int count) {
    setVariable(Curve::variableRuntimeHash(variable_name), d_ptr, count);
}
void HostCurve::setMessageInputVariable(const std::string& variable_name, void* d_ptr, const unsigned int count) {
    setVariable(message_in_hash + Curve::variableRuntimeHash(variable_name), d_ptr, count);
}
void HostCurve::setMessageOutputVariable(const std::string& variable_name, void* d_ptr, const unsigned int count) {
    setVariable(message_out_hash + Curve::variableRuntimeHash(variable_name), d_ptr, count);
}
void HostCurve::setAgentOutputVariable(const std::string& variable_name, void* d_ptr, const unsigned int count) {
    setVariable(agent_out_hash + Curve::variableRuntimeHash(variable_name), d_ptr, count);
}
void HostCurve::setVariable(const VariableHash variable_hash, void * d_ptr, const unsigned int count) {
    if (variable_hash == EMPTY_FLAG) {
        THROW exception::CurveException("Unable to set variable, it's hash matches a reserved symbol!");
    }
    unsigned int i = (variable_hash) % MAX_VARIABLES;
    unsigned int n = 0;
    while (h_curve_table.hashes[i] != variable_hash) {
        n += 1;
        if (n >= MAX_VARIABLES) {
            THROW exception::CurveException("Unable to set variable, not found in hash table!");
        }
        i += 1;
        if (i >= MAX_VARIABLES) {
            i = 0;
        }
    }
    // Set the pointer
    h_curve_table.variables[i] = static_cast<char*>(d_ptr);
    // Set the count
    h_curve_table.count[i] = count;
}
int HostCurve::size() const {
    // @todo Could track size as items are inserted to make this 'free'
    int rtn = 0;
    for (unsigned int hash : h_curve_table.hashes) {
        if (hash != EMPTY_FLAG)
            rtn++;
    }
    return rtn;
}
void HostCurve::updateDevice_async(const cudaStream_t stream) {
    NVTX_RANGE("HostCurve::updateDevice_async()");
    // Initialise the device (if required)
    assert(d_curve_table);  // No reason for this to ever fail. Purge calls init device
    // Copy
    gpuErrchk(cudaMemcpyAsync(d_curve_table, &h_curve_table, sizeof(CurveTable), cudaMemcpyHostToDevice, stream));
}
const CurveTable *HostCurve::getDevicePtr() const {
    return d_curve_table;
}

}  // namespace curve
}  // namespace detail
}  // namespace flamegpu

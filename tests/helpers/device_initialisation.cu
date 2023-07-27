#include "helpers/device_initialisation.h"
#include <cuda.h>
#include <stdio.h>
#include "flamegpu/flamegpu.h"

namespace flamegpu {
namespace tests {
namespace {
    // Boolean to store the result of the test, in an anonymous namespace (i.e. static)
    bool _CUDASimulationContextCreation_result = false;
}  // namespace

void runCUDASimulationContextCreationTest() {
    // Create a very simple model to enable creation of a CUDASimulation
    ModelDescription m("model");
    m.newAgent("agent");
    CUDASimulation c(m);
    c.CUDAConfig().device_id = 0;
    c.SimulationConfig().steps = 1;
    // Use the CUDA driver api to check there is no current context (i.e. it is null), ignoring any cuda errors reported
    CUresult cuErr = CUDA_SUCCESS;
    CUcontext ctxBefore = NULL;
    CUcontext ctxAfter = NULL;
    cuErr = cuCtxGetCurrent(&ctxBefore);
    if (cuErr != CUDA_SUCCESS && cuErr != CUDA_ERROR_NOT_INITIALIZED) {
        const char *error_str;
        cuGetErrorString(cuErr, &error_str);
        fprintf(stderr, "CUDA Driver API error occurred during cuCtxGetCurrent at %s(%d): %s.\n", __FILE__, __LINE__, error_str);
        return;
    }
    // Apply the config, which should establish a cuda context
    c.applyConfig();
    // Use the CUDA driver API to ensure there is now a non-null CUDA context established.
    cuErr = cuCtxGetCurrent(&ctxAfter);
    if (cuErr != CUDA_SUCCESS) {
        const char *error_str;
        cuGetErrorString(cuErr, &error_str);
        fprintf(stderr, "CUDA Driver API error occurred during cuCtxGetCurrent at %s(%d): %s.\n", __FILE__, __LINE__, error_str);
        return;
    }
    // the result is truthy if the before context was null, and the after context is non null
    _CUDASimulationContextCreation_result = ctxBefore == NULL && ctxAfter != NULL;
    // Run the simulation.
    c.simulate();
}

/**
 * Return the value stored in the anonymous namespace.
 * @note - there is no way to know if the test has not yet been ran, instead it reports false.
 */
bool getCUDASimulationContextCreationTestResult() {
    return _CUDASimulationContextCreation_result;
}

}  // namespace tests
}  // namespace flamegpu

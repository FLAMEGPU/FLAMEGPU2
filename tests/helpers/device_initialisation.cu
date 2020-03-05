#include "helpers/device_initialisation.h"
#include <stdio.h>
#include <chrono>
#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace {
    // Boolean to store the result of the test, in an anonymous namespace (i.e. static)
    bool _CUDAAgentModelContextCreationTime_result = false;
}
// Set a threshold value, which is large enough to account for context creation
// Experimentally cudaFree(0); takes ~2us (nsys) without context creation,
// while cudaFree(0) including context creation takes ~> 150ms in my linux titan v system.
// This test is a little fluffy.
const double CONTEXT_CREATION_ATLEAST_SECONDS = 0.100;  // atleast 100ms?

/* Test that CUDAAgentModel::applyConfig_derived() is invoked prior to any cuda call which will invoke the CUDA
Alternative is to use the driver API, call CuCtxGetCurrent(CuContext* pctx) immediatebly before applyConfig, and if pctx is the nullptr then the context had not yet been initialised?
@note - This needs to be called first, and only once.
*/
void timeCUDAAgentModelContextCreationTest() {
    printf("timeCUDAAgentModelContextCreationTest\n");
    // Create a very simple model to enable creation of a CudaAgentModel
    ModelDescription m("model");
    m.newAgent("agent");
    CUDAAgentModel c(m);
    c.CUDAConfig().device_id = 0;
    c.SimulationConfig().steps = 1;
    // Time how long applyconfig takes, which should invoke cudaFree as the first cuda command, initialising the context.
    auto t0 = std::chrono::high_resolution_clock::now();
    c.applyConfig();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
    // The test fails if applyconfig was too fast.
    _CUDAAgentModelContextCreationTime_result = time_span.count() >= CONTEXT_CREATION_ATLEAST_SECONDS;
    // Run the simulation.
    c.simulate();
}

/**
 * Return the value stored in the anonymous namespace.
 * @note - there is no way to know if the test has not yet been ran, instead it reports false.
 */
bool getCUDAAgentModelContextCreationTestResult() {
    return _CUDAAgentModelContextCreationTime_result;
}

#ifndef INCLUDE_FLAMEGPU_FLAMEGPU_H_
#define INCLUDE_FLAMEGPU_FLAMEGPU_H_

#ifdef FLAMEGPU_USE_GLM
#ifdef __CUDACC__
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#else
#pragma diag_suppress = esa_on_defaulted_function_ignored
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#endif  // __CUDACC__
#include <glm/glm.hpp>
#endif

/**
 * @namespace flamegpu
 * The flamegpu namespace containing all namespace'd elements of the flamegpu api.
 *
 * The inner detail namespace and it's members are implementation details not considered part of the public facing API and may change at any time.
 */

// include all host API classes (top level header from each module)
#include "flamegpu/version.h"
#include "flamegpu/runtime/HostAPI.h"
#include "flamegpu/runtime/agent/HostAgentAPI.cuh"
#include "flamegpu/runtime/DeviceAPI.cuh"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/DependencyGraph.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubAgentDescription.h"
#include "flamegpu/model/SubEnvironmentDescription.h"
#include "flamegpu/model/EnvironmentDirectedGraphDescription.cuh"
#include "flamegpu/simulation/AgentVector.h"
#include "flamegpu/runtime/agent/AgentInstance.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/runtime/messaging.h"
#include "flamegpu/runtime/AgentFunction_shim.cuh"
#include "flamegpu/runtime/AgentFunctionCondition_shim.cuh"
#include "flamegpu/simulation/CUDAEnsemble.h"
#include "flamegpu/simulation/RunPlanVector.h"
#include "flamegpu/simulation/LoggingConfig.h"
#include "flamegpu/simulation/AgentLoggingConfig.h"
#include "flamegpu/simulation/LogFrame.h"
#include "flamegpu/util/cleanup.h"
#include "flamegpu/io/Telemetry.h"

// This include has no impact if FLAMEGPU_VISUALISATION is not defined
#include "flamegpu/visualiser/visualiser_api.h"

#endif  // INCLUDE_FLAMEGPU_FLAMEGPU_H_

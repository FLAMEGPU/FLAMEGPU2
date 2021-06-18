#ifndef INCLUDE_FLAMEGPU_FLAMEGPU_H_
#define INCLUDE_FLAMEGPU_FLAMEGPU_H_

// include all host API classes (top level header from each module)
#include "flamegpu/version.h"
#include "flamegpu/runtime/HostAPI.h"
#include "flamegpu/runtime/HostAgentAPI.h"
#include "flamegpu/runtime/DeviceAPI.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/DependencyGraph.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubAgentDescription.h"
#include "flamegpu/model/SubEnvironmentDescription.h"
#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/pop/AgentInstance.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/runtime/messaging.h"
#include "flamegpu/runtime/AgentFunction_shim.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
#include "flamegpu/gpu/CUDAEnsemble.h"
#include "flamegpu/sim/RunPlanVector.h"
#include "flamegpu/sim/LoggingConfig.h"
#include "flamegpu/sim/AgentLoggingConfig.h"
#include "flamegpu/sim/LogFrame.h"

// This include has no impact if VISUALISATION is not defined
#include "flamegpu/visualiser/visualiser_api.h"

#endif  // INCLUDE_FLAMEGPU_FLAMEGPU_H_

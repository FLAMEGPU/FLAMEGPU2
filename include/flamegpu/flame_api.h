/*
* flame_api.h
*
*  Created on: 19 Feb 2014
*      Author: paul
*/

#ifndef INCLUDE_FLAMEGPU_FLAME_API_H_
#define INCLUDE_FLAMEGPU_FLAME_API_H_

// include all host API classes (top level header from each module)
#include "flamegpu/runtime/flamegpu_api.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
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
#include "flamegpu/visualiser/ModelVis.h"
#include "flamegpu/gpu/CUDAEnsemble.h"
#include "flamegpu/sim/RunPlanVec.h"
#include "flamegpu/sim/LoggingConfig.h"
#include "flamegpu/sim/AgentLoggingConfig.h"
#include "flamegpu/sim/LogFrame.h"

#endif  // INCLUDE_FLAMEGPU_FLAME_API_H_

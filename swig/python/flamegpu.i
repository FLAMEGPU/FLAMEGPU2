%begin %{
// define SWIG_PYTHON_INTERPRETER_NO_DEBUG on windows debug builds as pythonXX_d is not packaged unless built from source
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

%module pyflamegpu
%{
/* Includes the header in the wrapper code */
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

//#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
//#include "flamegpu/runtime/flamegpu_host_agent_api.h"

#include "flamegpu/runtime/messaging.h"
#include "flamegpu/runtime/AgentFunction_shim.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
%}


/* Parse the header file to generate wrappers */
%include "flamegpu/model/ModelDescription.h"
%include "flamegpu/model/AgentDescription.h"
%include "flamegpu/model/AgentFunctionDescription.h"
%include "flamegpu/model/EnvironmentDescription.h"
%include "flamegpu/model/LayerDescription.h"
%include "flamegpu/pop/AgentPopulation.h"
%include "flamegpu/pop/AgentInstance.h"
%include "flamegpu/gpu/CUDAAgentModel.h"

//%include "flamegpu/runtime/flamegpu_device_api.h"
//%include "flamegpu/runtime/flamegpu_host_api.h"
//#include "flamegpu/runtime/flamegpu_host_agent_api.h" // issues with cub and swig

%include "flamegpu/runtime/messaging.h"
%include "flamegpu/runtime/AgentFunction_shim.h"
%include "flamegpu/runtime/AgentFunctionCondition_shim.h"

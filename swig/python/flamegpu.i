%begin %{
// define SWIG_PYTHON_INTERPRETER_NO_DEBUG on windows debug builds as pythonXX_d is not packaged unless built from source
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

// supress known warnings
#pragma SWIG nowarn=325,302,401

// string support
%include <std_string.i>

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

/* Disable non RTC function and function condtion set methods */
%ignore AgentDescription::newFunction;
%ignore AgentFunctionDescription::getFunctionPtr;
%ignore AgentFunctionDescription::setFunctionCondition;
%ignore AgentFunctionDescription::getConditionPtr;

/* Parse the header file to generate wrappers */
%include "flamegpu/model/ModelDescription.h"
%include "flamegpu/model/AgentDescription.h"

/* Instanciate template functions */
%template(newFloatVariable) AgentDescription::newVariable<float>;
%template(newFloatVariable) AgentDescription::newVariable<float, 1>;




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

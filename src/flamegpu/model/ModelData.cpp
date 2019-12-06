#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

const std::string ModelData::DEFAULT_STATE = "default";


ModelData::ModelData(const std::string &model_name)
    : environment(new EnvironmentDescription())
    , name(model_name) { }

AgentData::AgentData(const std::string &agent_name)
    : description(new AgentDescription())
    , name(agent_name) { }

MessageData::MessageData(const std::string &message_name)
    : description(new MessageDescription())
    , name(message_name) { }

AgentFunctionData::AgentFunctionData(const std::string &function_name, AgentFunctionWrapper *agent_function)
    : func(agent_function)
    , description(new AgentFunctionDescription())
    , name(function_name) { }


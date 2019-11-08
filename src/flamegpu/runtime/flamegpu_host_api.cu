#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/runtime/flamegpu_host_agent_api.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

FLAMEGPU_HOST_AGENT_API FLAMEGPU_HOST_API::agent(const std::string &agent_name)
{
    return FLAMEGPU_HOST_AGENT_API(*this, agentModel.getCUDAAgent(agent_name));
}

FLAMEGPU_HOST_AGENT_API FLAMEGPU_HOST_API::agent(const std::string &agent_name, const std::string &stateName) 
{
    return FLAMEGPU_HOST_AGENT_API(*this, agentModel.getCUDAAgent(agent_name), stateName);
}
#ifndef INCLUDE_FLAMEGPU_SIM_AGENTINTERFACE_H_
#define INCLUDE_FLAMEGPU_SIM_AGENTINTERFACE_H_

#include "flamegpu/model/ModelData.h"

struct AgentData;
/**
 * Base-class (interface) for classes like CUDAAgent, which provide access to agent data
 */
class AgentInterface {
public:
    virtual ~AgentInterface() = default;
    virtual const AgentData &getAgentDescription() const = 0;
    virtual void *getStateVariablePtr(const std::string &state_name, const std::string &variable_name) = 0;
    virtual ModelData::size_type getStateSize(const std::string &state_name) const = 0;
};

#endif  // INCLUDE_FLAMEGPU_SIM_AGENTINTERFACE_H_
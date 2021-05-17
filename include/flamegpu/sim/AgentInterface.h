#ifndef INCLUDE_FLAMEGPU_SIM_AGENTINTERFACE_H_
#define INCLUDE_FLAMEGPU_SIM_AGENTINTERFACE_H_

#include <string>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/defines.h"

namespace flamegpu {

/**
 * Base-class (interface) for classes like CUDAAgent, which provide access to agent data
 */
class AgentInterface {
 public:
    virtual ~AgentInterface() = default;
    virtual const AgentData &getAgentDescription() const = 0;
    virtual void *getStateVariablePtr(const std::string &state_name, const std::string &variable_name) = 0;
    virtual ModelData::size_type getStateSize(const std::string &state_name) const = 0;
    /**
     * Returns the next free agent id, and increments the ID tracker by the specified count
     * @param count Number that will be added to the return value on next call to this function
     * @return An ID that can be assigned to an agent that wil be stored within this Agent collection
     */
    virtual id_t nextID(unsigned int count) = 0;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_AGENTINTERFACE_H_

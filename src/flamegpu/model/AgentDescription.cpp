#include "flamegpu/model/AgentDescription.h"

#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {

CAgentDescription::CAgentDescription(std::shared_ptr<AgentData> data)
    : agent(std::move(data)) { }
CAgentDescription::CAgentDescription(std::shared_ptr<const AgentData> data)
    : agent(std::move(std::const_pointer_cast<AgentData>(data))) { }

bool CAgentDescription::operator==(const CAgentDescription& rhs) const {
    return *this->agent == *rhs.agent;  // Compare content is functionally the same
}
bool CAgentDescription::operator!=(const CAgentDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
std::string CAgentDescription::getName() const {
    return agent->name;
}

flamegpu::size_type CAgentDescription::getStatesCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<flamegpu::size_type>(agent->states.size());
}
std::string CAgentDescription::getInitialState() const {
    return agent->initial_state;
}
const std::type_index& CAgentDescription::getVariableType(const std::string& variable_name) const {
    auto f = agent->variables.find(variable_name);
    if (f != agent->variables.end()) {
        return f->second.type;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableType().",
        agent->name.c_str(), variable_name.c_str());
}
size_t CAgentDescription::getVariableSize(const std::string& variable_name) const {
    auto f = agent->variables.find(variable_name);
    if (f != agent->variables.end()) {
        return f->second.type_size;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableSize().",
        agent->name.c_str(), variable_name.c_str());
}

flamegpu::size_type CAgentDescription::getVariableLength(const std::string& variable_name) const {
    auto f = agent->variables.find(variable_name);
    if (f != agent->variables.end()) {
        return f->second.elements;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableLength().",
        agent->name.c_str(), variable_name.c_str());
}

flamegpu::size_type CAgentDescription::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<flamegpu::size_type>(agent->variables.size());
}
CAgentFunctionDescription CAgentDescription::getFunction(const std::string& function_name) const {
    auto f = agent->functions.find(function_name);
    if (f != agent->functions.end()) {
        return CAgentFunctionDescription(f->second);
    }
    THROW exception::InvalidAgentFunc("Agent ('%s') does not contain function '%s', "
        "in AgentDescription::getFunction().",
        agent->name.c_str(), function_name.c_str());
}
flamegpu::size_type CAgentDescription::getFunctionsCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<flamegpu::size_type>(agent->functions.size());
}

flamegpu::size_type CAgentDescription::getAgentOutputsCount() const {
    return agent->agent_outputs;
}

const std::set<std::string>& CAgentDescription::getStates() const {
    return agent->states;
}

bool CAgentDescription::hasState(const std::string& state_name) const {
    return agent->states.find(state_name) != agent->states.end();
}
bool CAgentDescription::hasVariable(const std::string& variable_name) const {
    return agent->variables.find(variable_name) != agent->variables.end();
}
bool CAgentDescription::hasFunction(const std::string& function_name) const {
    return agent->functions.find(function_name) != agent->functions.end();
}
bool CAgentDescription::isOutputOnDevice() const {
    return agent->isOutputOnDevice();
}

/**
 * Constructors
 */
AgentDescription::AgentDescription(std::shared_ptr<AgentData> data)
    : CAgentDescription(std::move(data)) { }

/**
 * Accessors
 */
void AgentDescription::newState(const std::string &state_name) {
    // If state doesn't already exist
    if (agent->states.find(state_name) == agent->states.end()) {
        // If default state  has not been added
        if (!agent->keepDefaultState) {
            // Special case, where default state has been replaced
            if (agent->states.size() == 1 && (*agent->states.begin()) == ModelData::DEFAULT_STATE) {
                agent->states.clear();
                agent->initial_state = state_name;
                // Update initial/end state on all functions
                // As prev has been removed
                for (auto &f : agent->functions) {
                    f.second->initial_state = state_name;
                    f.second->end_state = state_name;
                }
            }
        }
        agent->states.insert(state_name);
        return;
    } else if (state_name == ModelData::DEFAULT_STATE) {
        agent->keepDefaultState = true;
        agent->states.insert(state_name);  // Re add incase it was dropped
    } else {
        THROW exception::InvalidStateName("Agent ('%s') already contains state '%s', "
            "in AgentDescription::newState().",
            agent->name.c_str(), state_name.c_str());
    }
}
void AgentDescription::setInitialState(const std::string &init_state) {
    if (agent->states.find(init_state) != agent->states.end()) {
        this->agent->initial_state = init_state;
        return;
    }
    THROW exception::InvalidStateName("Agent ('%s') does not contain state '%s', "
        "in AgentDescription::setInitialState().",
        agent->name.c_str(), init_state.c_str());
}

AgentFunctionDescription AgentDescription::Function(const std::string &function_name) {
    auto f = agent->functions.find(function_name);
    if (f != agent->functions.end()) {
        return AgentFunctionDescription(f->second);
    }
    THROW exception::InvalidAgentFunc("Agent ('%s') does not contain function '%s', "
        "in AgentDescription::Function().",
        agent->name.c_str(), function_name.c_str());
}

void AgentDescription::setSortPeriod(const unsigned int sortPeriod) {
    agent->sortPeriod = sortPeriod;
}



}  // namespace flamegpu

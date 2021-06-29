#include "flamegpu/model/AgentDescription.h"

#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {

/**
 * Constructors
 */
AgentDescription::AgentDescription(std::shared_ptr<const ModelData> _model, AgentData *const data)
    : model(_model)
    , agent(data) { }


bool AgentDescription::operator==(const AgentDescription& rhs) const {
    return *this->agent == *rhs.agent;  // Compare content is functionally the same
}
bool AgentDescription::operator!=(const AgentDescription& rhs) const {
    return !(*this == rhs);
}


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

AgentFunctionDescription &AgentDescription::Function(const std::string &function_name) {
    auto f = agent->functions.find(function_name);
    if (f != agent->functions.end()) {
        return *f->second->description;
    }
    THROW exception::InvalidAgentFunc("Agent ('%s') does not contain function '%s', "
        "in AgentDescription::Function().",
        agent->name.c_str(), function_name.c_str());
}

/**
 * Const Accessors
 */
std::string AgentDescription::getName() const {
    return agent->name;
}

ModelData::size_type AgentDescription::getStatesCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<ModelData::size_type>(agent->states.size());
}
std::string AgentDescription::getInitialState() const {
    return agent->initial_state;
}
const std::type_index &AgentDescription::getVariableType(const std::string &variable_name) const {
    auto f = agent->variables.find(variable_name);
    if (f != agent->variables.end()) {
        return f->second.type;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableType().",
        agent->name.c_str(), variable_name.c_str());
}
size_t AgentDescription::getVariableSize(const std::string &variable_name) const {
    auto f = agent->variables.find(variable_name);
    if (f != agent->variables.end()) {
        return f->second.type_size;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableSize().",
        agent->name.c_str(), variable_name.c_str());
}
ModelData::size_type AgentDescription::getVariableLength(const std::string &variable_name) const {
    auto f = agent->variables.find(variable_name);
    if (f != agent->variables.end()) {
        return f->second.elements;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableLength().",
        agent->name.c_str(), variable_name.c_str());
}
ModelData::size_type AgentDescription::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<ModelData::size_type>(agent->variables.size());
}
const AgentFunctionDescription& AgentDescription::getFunction(const std::string &function_name) const {
    auto f = agent->functions.find(function_name);
    if (f != agent->functions.end()) {
        return *f->second->description;
    }
    THROW exception::InvalidAgentFunc("Agent ('%s') does not contain function '%s', "
        "in AgentDescription::getFunction().",
        agent->name.c_str(), function_name.c_str());
}
ModelData::size_type AgentDescription::getFunctionsCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<ModelData::size_type>(agent->functions.size());
}

ModelData::size_type AgentDescription::getAgentOutputsCount() const {
    return agent->agent_outputs;
}

const std::set<std::string> &AgentDescription::getStates() const {
    return agent->states;
}

bool AgentDescription::hasState(const std::string &state_name) const {
    return agent->states.find(state_name) != agent->states.end();
}
bool AgentDescription::hasVariable(const std::string &variable_name) const {
    return agent->variables.find(variable_name) != agent->variables.end();
}
bool AgentDescription::hasFunction(const std::string &function_name) const {
    return agent->functions.find(function_name) != agent->functions.end();
}
bool AgentDescription::isOutputOnDevice() const {
    return agent->isOutputOnDevice();
}

}  // namespace flamegpu

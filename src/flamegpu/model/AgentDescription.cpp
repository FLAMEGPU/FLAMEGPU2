#include "flamegpu/model/AgentDescription.h"

#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/exception/FGPUException.h"

/**
 * Constructors
 */
AgentDescription::AgentDescription(ModelDescription * const _model, const std::string &agent_name)
    : name(agent_name), model(_model) { }
// Copy Construct
AgentDescription::AgentDescription(const AgentDescription &other_agent)
    : model(other_agent.model) {
    // TODO
}
// Move Construct
AgentDescription::AgentDescription(AgentDescription &&other_agent)
    : model(other_agent.model) {
    // TODO
}
// Copy Assign
AgentDescription& AgentDescription::operator=(const AgentDescription &other_agent) {
    // TODO
    return *this;
}
// Move Assign
AgentDescription& AgentDescription::operator=(AgentDescription &&other_agent) {
    // TODO
    return *this;
}

AgentDescription AgentDescription::clone(const std::string &cloned_agent_name) const {
    // TODO
    return AgentDescription(model, cloned_agent_name);
}


/**
 * Accessors
 */
void AgentDescription::newState(const std::string &state_name) {
    if (states.find(state_name) == states.end()) {
        states.insert(state_name);
        // Special case, where default state has been replaced
        if (states.size() == 1)
            this->initial_state = state_name;
        return;
    }
    THROW InvalidStateName("Agent ('%s') already contains state '%s', "
        "in AgentDescription::newState().",
        name.c_str(), state_name.c_str());
}
void AgentDescription::setInitialState(const std::string &init_state) {
    if (states.find(init_state) != states.end()) {
        this->initial_state = init_state;
        return;
    }
    THROW InvalidStateName("Agent ('%s') does not contain state '%s', "
        "in AgentDescription::setInitialState().",
        name.c_str(), init_state.c_str());
}

AgentFunctionDescription &AgentDescription::Function(const std::string &function_name) {
    auto f = functions.find(function_name);
    if (f != functions.end()) {
        return *f->second;
    }
    THROW InvalidAgentFunc("Agent ('%s') does not contain function '%s', "
        "in AgentDescription::Function().",
        name.c_str(), function_name.c_str());
}
AgentFunctionDescription &AgentDescription::cloneFunction(const AgentFunctionDescription &function) {
    // TODO
    return *((*functions.begin()).second);
}

/**
 * Const Accessors
 */
std::string AgentDescription::getName() const {
    return name;
}
    
std::type_index AgentDescription::getVariableType(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return f->second.type;
    }
    THROW InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableType().",
        name.c_str(), variable_name.c_str());
}
size_t AgentDescription::getVariableSize(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return f->second.type_size;
    }
    THROW InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableSize().",
        name.c_str(), variable_name.c_str());
}
AgentDescription::size_type AgentDescription::getVariableLength(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return f->second.elements;
    }
    THROW InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableLength().",
        name.c_str(), variable_name.c_str());
}
AgentDescription::size_type AgentDescription::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<size_type>(variables.size());
}
const AgentFunctionDescription& AgentDescription::getFunction(const std::string &function_name) const {
    auto f = functions.find(function_name);
    if (f != functions.end()) {
        return *f->second;
    }
    THROW InvalidAgentFunc("Agent ('%s') does not contain function '%s', "
        "in AgentDescription::getFunction().",
        name.c_str(), function_name.c_str());
}

const std::set<std::string> &AgentDescription::getStates() const {
    return states;
}
const AgentDescription::VariableMap &AgentDescription::getVariables() const {
    return variables;
}
const AgentDescription::FunctionMap& AgentDescription::getFunctions() const {
    return functions;
}

bool AgentDescription::hasState(const std::string &state_name) const {
    return states.find(state_name) != states.end();
}
bool AgentDescription::hasVariable(const std::string &variable_name) const {
    return variables.find(variable_name) != variables.end();
}
bool AgentDescription::hasFunction(const std::string &function_name) const {
    return functions.find(function_name) != functions.end();
}
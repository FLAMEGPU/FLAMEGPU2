#include "flamegpu/model/AgentDescription.h"

#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/exception/FGPUException.h"

/**
 * Constructors
 */
AgentDescription::AgentDescription(const std::string &agent_name) 
    : name(agent_name) { }
// Copy Construct
AgentDescription::AgentDescription(const AgentDescription &other_agent) {
    // TODO
}
// Move Construct
AgentDescription::AgentDescription(AgentDescription &&other_agent) {
    // TODO
}
// Copy Assign
AgentDescription& AgentDescription::operator=(const AgentDescription &other_agent) {
    // TODO
}
// Move Assign
AgentDescription& AgentDescription::operator=(AgentDescription &&other_agent) {
    // TODO
}

AgentDescription AgentDescription::clone(const std::string &cloned_agent_name) const {
    // TODO
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

template <typename T, AgentDescription::size_type N>
void AgentDescription::newVariable(const std::string &variable_name) {
    if (variables.find(variable_name) == variables.end()) {
        variables.emplace(variable_name, std::make_tuple(typeid(T), sizeof(T), N));
        return;
    }
    THROW InvalidAgentVar("Agent ('%s') already contains variable '%s', "
        "in AgentDescription::newVariable().",
        name.c_str(), variable_name.c_str());
}

AgentFunctionDescription &AgentDescription::newFunction(const std::string &function_name) {
    if (functions.find(function_name) == functions.end()) {
        auto rtn = std::make_shared<AgentFunctionDescription>(function_name, );  // TODO
        functions.emplace(function_name, rtn);
        return rtn;
    }
    THROW InvalidAgentFunc("Agent ('%s') already contains variable '%s', "
        "in AgentDescription::newFunction().",
        name.c_str(), function_name.c_str());
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
        return std::get<0>(f->second);
    }
    THROW InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableType().",
        name.c_str(), variable_name.c_str());
}
size_t AgentDescription::getVariableSize(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return std::get<1>(f->second);
    }
    THROW InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableSize().",
        name.c_str(), variable_name.c_str());
}
size_t AgentDescription::getVariableLength(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return std::get<2>(f->second);
    }
    THROW InvalidAgentVar("Agent ('%s') does not contain variable '%s', "
        "in AgentDescription::getVariableLength().",
        name.c_str(), variable_name.c_str());
}
AgentDescription::size_type AgentDescription::getVariablesCount() const {
    return variables.size();
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
    // TODO
}
const AgentDescription::FunctionMap& AgentDescription::getFunctions() const {
    // TODO
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
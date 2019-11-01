/**
 * @file AgentDescription.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <flamegpu/model/AgentDescription.h>

#include <flamegpu/exception/FGPUException.h>

AgentDescription::AgentDescription(std::string name) : default_state(new AgentStateDescription("default")), states(), functions(), memory(), sizes() {

    stateless = true;
    this->name = name;
    addState(*default_state, true);
}

AgentDescription::~AgentDescription() {


}

void AgentDescription::setName(std::string _name) {

}

const std::string AgentDescription::getName() const {

    return name;
}

void AgentDescription::addState(const AgentStateDescription& state, bool is_initial_state) {


    // check if this is a stateless system
    if (stateless) {

        stateless = false;
    }

    states.insert(StateMap::value_type(state.getName(), state));
    if (is_initial_state)
        setInitialState(state.getName());
}

void AgentDescription::setInitialState(const std::string _initial_state) {

    this->initial_state = _initial_state;
}

void AgentDescription::addAgentFunction(AgentFunctionDescription& function) {

    functions.insert(FunctionMap::value_type(function.getName(), function));

    // TODO: Set the parent of the function
    function.setParent(*this);
}

MemoryMap& AgentDescription::getMemoryMap() {

    return memory;
}

const MemoryMap& AgentDescription::getMemoryMap() const {

    return memory;
}

const StateMap& AgentDescription::getStateMap() const {

    return states;
}

const FunctionMap& AgentDescription::getFunctionMap() const {

    return functions;
}

size_t AgentDescription::getMemorySize() const {

    size_t size = 0;
    for (TypeSizeMap::const_iterator it = sizes.begin(); it != sizes.end(); it++) {

        size += it->second;
    }
    return size;
}

unsigned int AgentDescription::getNumberAgentVariables() const {

    return static_cast<unsigned int>(memory.size());
}

size_t AgentDescription::getAgentVariableSize(const std::string variable_name) const {

    // get the variable name type
    MemoryMap::const_iterator mm = memory.find(variable_name);
    if (mm == memory.end())
        throw InvalidAgentVar("Invalid agent memory variable");
    const std::type_info *t = &(mm->second);
    // get the type size
    TypeSizeMap::const_iterator tsm = sizes.find(t);
    if (tsm == sizes.end())
        throw InvalidMapEntry("Missing entry in type sizes map");
    return tsm->second;
}

bool AgentDescription::requiresAgentCreation() const {


    // needs to search entire model for any functions with an agent output for this agent
    for (FunctionMap::const_iterator it= functions.begin(); it != functions.end(); it++) {

        // if (*it->second()->)
    }

    return false;
}

const std::type_info& AgentDescription::getVariableType(const std::string variable_name) const {

    MemoryMap::const_iterator iter;
    iter = memory.find(variable_name);

    if (iter == memory.end())
        throw InvalidAgentVar("Invalid agent memory variable");

    return iter->second;

}

bool AgentDescription::hasAgentFunction(const std::string function_name) const {

    FunctionMap::const_iterator f;
    f = functions.find(function_name);
    return (f != functions.end());
}

/*
StateMemoryMap AgentDescription::getEmptyStateMemoryMap() const {

    // needs to do some deep copying
    return sm_map;    // returns a copy of the sm memorymap
}
*/

void AgentDescription::initEmptyStateMemoryMap(StateMemoryMap& map) const {

    for (const StateMemoryMapPair& sm_p : sm_map){
        map.insert(StateMemoryMap::value_type(sm_p.first, std::unique_ptr<GenericMemoryVector>(sm_p.second->clone())));
    }
}

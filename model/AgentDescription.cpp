/**
 * @file AgentDescription.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "AgentDescription.h"

AgentDescription::AgentDescription(std::string name) : states(), functions(), memory(), sizes(), default_state(new AgentStateDescription("default"))
{
    stateless = true;
    this->name = name;
    addState(*default_state, true);
}

AgentDescription::~AgentDescription()
{

}


void AgentDescription::setName(std::string name)
{
}

const std::string AgentDescription::getName() const
{
    return name;
}

void AgentDescription::addState(const AgentStateDescription& state, bool initial_state)
{

    //check if this is a stateless system
    if (stateless)
    {
        stateless = false;
    }

    states.insert(StateMap::value_type(state.getName(), state));
    if (initial_state)
        setInitialState(state.getName());
}

void AgentDescription::setInitialState(const std::string initial_state)
{
    this->initial_state = initial_state;
}

void AgentDescription::addAgentFunction(const AgentFunctionDescription& function)
{

    functions.insert(FunctionMap::value_type(function.getName(), function));
}

MemoryMap& AgentDescription::getMemoryMap()   // Moz: why two getMemoryMap, one is const
{
    return memory;
}


const MemoryMap& AgentDescription::getMemoryMap() const
{
    return memory;
}


const StateMap& AgentDescription::getStateMap() const
{
    return states;
}

unsigned int AgentDescription::getMemorySize() const
{
    unsigned int size = 0;
    for (TypeSizeMap::const_iterator it = sizes.begin(); it != sizes.end(); it++)
    {
        size += it->second;
    }
    return size;
}

boost::any AgentDescription::getDefaultValue(const std::string variable_name) const
{
    //need to do a check to make sure that the varibale name exists
    //if it does then the following is safe
    DefaultValueMap::const_iterator dm = defaults.find(variable_name);
    if (dm == defaults.end())
        //throw std::runtime_error("Invalid agent memory variable");
        throw InvalidAgentVar();

    return dm->second;
}

unsigned int AgentDescription::getNumberAgentVariables() const
{
    return memory.size();
}

const unsigned int AgentDescription::getAgentVariableSize(const std::string variable_name) const
{
    //get the variable name type
    MemoryMap::const_iterator mm = memory.find(variable_name);
    if (mm == memory.end())
        // throw std::runtime_error("Invalid agent memory variable");
        throw InvalidAgentVar();
    const std::type_info *t = &(mm->second);
    //get the type size
    TypeSizeMap::const_iterator tsm = sizes.find(t);
    if (tsm == sizes.end())
        //throw std::runtime_error("Missing entry in type sizes map. Something went bad.");
        throw InvalidMapEntry();
    return tsm->second;
}

bool AgentDescription::requiresAgentCreation() const
{

    //needs to search entire model for any functions with an agent output for this agent
    for (FunctionMap::const_iterator it= functions.begin(); it != functions.end(); it++)
    {
        //if (*it->second()->)
    }

    return false;
}

const std::type_info& AgentDescription::getVariableType(const std::string variable_name) const
{
    MemoryMap::const_iterator iter;
    iter = memory.find(variable_name);

    if (iter == memory.end())
        // throw std::runtime_error("Invalid agent memory variable");
        throw InvalidAgentVar();

    return iter->second;

}

bool AgentDescription::hasAgentFunction(const std::string function_name) const
{
    FunctionMap::const_iterator f;
    f = functions.find(function_name);
    return (f != functions.end());
}

/*
StateMemoryMap AgentDescription::getEmptyStateMemoryMap() const
{
	//needs to do some deep copying
	return sm_map;	//returns a copy of the sm memorymap
}
*/

void AgentDescription::initEmptyStateMemoryMap(StateMemoryMap& map) const
{
	for (const StateMemoryMapPair& sm_p : sm_map){
		map.insert(StateMemoryMap::value_type(sm_p.first, std::unique_ptr<GenericAgentMemoryVector>(sm_p.second->clone())));
	}
}
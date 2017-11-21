/**
 * @file AgentDescription.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
 */

#ifndef AGENTDESCRIPTION_H_
#define AGENTDESCRIPTION_H_

#include <string>
#include <map>
#include <typeinfo>

//include generic memory vector
#include "../pop/MemoryVector.h"

//include class dependencies
#include "AgentStateDescription.h"
#include "AgentFunctionDescription.h"


//State map is a mapping between a state name (i.e. default) and a state description object reference
/*! */
typedef std::map<const std::string, const AgentStateDescription&> StateMap;

/*! */
typedef std::pair<const std::string, const AgentStateDescription&> StateMapPair;

/*! */
typedef std::map<const std::string, const AgentFunctionDescription&> FunctionMap;

/*! */
typedef std::map<const std::string, const std::type_info&> MemoryMap;

/*! */
typedef std::pair<const std::string, const std::type_info&> MemoryMapPair;

/*! Create a map with std::type_info for keys (indexes) and std::size_t values*/
typedef std::map<const std::type_info*, std::size_t> TypeSizeMap;	//not something that the user every sees. This is an interval map only for tracking the size of data types.



class AgentDescription
{
public:

    /**
    *
    */
    AgentDescription(std::string name);

    virtual ~AgentDescription();

    void setName(std::string name);

    const std::string getName() const;

    /*
     * All agent models start initially by being defined as stateless. A stateless model has a single state called "default".
     * If the addState function is called then the stateless flag should be set to false and the default state removed.
     * Care should be taken if default is removed as this may invalidate some agent functions already added to the model that use the default state. Not clear yet if model validation should be done at the end or during model building.
     * Inclined to think that validation should be done at the end and then the model set to read only.
     */
    void addState(const AgentStateDescription& state, bool initial_state=false);

    void setInitialState(const std::string initial_state);

    void addAgentFunction(AgentFunctionDescription &function);

    /*
     *
     */
	template <typename T> void addAgentVariable(const std::string variable_name);


    MemoryMap& getMemoryMap(); //TODO should be shared pointer

    const MemoryMap& getMemoryMap() const;

    const StateMap& getStateMap() const;

    const FunctionMap& getFunctionMap() const;

    const size_t getAgentVariableSize(const std::string variable_name) const;

	size_t getMemorySize() const;

    unsigned int getNumberAgentVariables() const;

    bool requiresAgentCreation() const;

    const std::type_info& getVariableType(const std::string variable_name) const;

    bool hasAgentFunction(const std::string function_name) const;

	//StateMemoryMap getEmptyStateMemoryMap() const;

	void initEmptyStateMemoryMap(StateMemoryMap&) const;

private:
    std::string name;
    bool stateless;												//system does not use states (i.e. only has a default state)
    std::string initial_state;
    std::unique_ptr<AgentStateDescription> default_state;

    StateMap states;
    FunctionMap functions;
    MemoryMap memory;
    TypeSizeMap sizes;
	StateMemoryMap sm_map;									//used to hold a default empty vector
};

template <typename T> void AgentDescription::addAgentVariable(const std::string variable_name)
{
	memory.insert(MemoryMap::value_type(variable_name, typeid(T)));
	sizes.insert(TypeSizeMap::value_type(&typeid(T), (unsigned int)sizeof(T)));
	sm_map.insert(StateMemoryMap::value_type(variable_name, std::unique_ptr<GenericMemoryVector>(new MemoryVector<T>())));
}


#endif /* AGENTDESCRIPTION_H_ */

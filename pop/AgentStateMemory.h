/*
 * AgentMemory.h
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#ifndef AGENTMEMORY_H_
#define AGENTMEMORY_H_


#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <memory>
#include <boost/any.hpp>
#include <typeinfo>


#include "../model/AgentDescription.h"

//now define in agentstatememory
//typedef std::map<const std::string, GenericAgentMemoryVector> StateMemoryMap;
//typedef std::pair<const std::string, GenericAgentMemoryVector> StateMemoryMapPair;

class AgentStateMemory  // agent_list
{
public:
    AgentStateMemory(const AgentDescription &description, const std::string agent_state) ;
    virtual ~AgentStateMemory() {}

    unsigned int getSize() const;

    //void init(const AgentDescription &agent_description);
    unsigned int creatNewInstance();

	GenericAgentMemoryVector& getMemoryVector(const std::string variable_name);

	const GenericAgentMemoryVector& getReadOnlyMemoryVector(const std::string variable_name) const;

    //todo:templated get vector function with boost any cast

    const std::type_info& getVariableType(std::string variable_name); //const

    bool isSameDescription(const AgentDescription& description) const;

protected:
    const AgentDescription &agent_description;
    const std::string agent_state;
    StateMemoryMap state_memory;
    unsigned int size;
};

#endif /* AGENTMEMORY_H_ */

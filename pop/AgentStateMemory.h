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
#include <typeinfo>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/any.hpp>
#include <typeinfo>


#include "../model/AgentDescription.h"

typedef boost::ptr_map<std::string, std::vector<boost::any> > StateMemoryMap;

class AgentStateMemory {
public:
	AgentStateMemory(AgentDescription &description, std::string agent_state) ;
	virtual ~AgentStateMemory() {}

	unsigned int getSize();
	void incrementSize();

	std::vector<boost::any>& getMemoryVector(const std::string variable_name);

	const std::vector<boost::any>& getReadOnlyMemoryVector(const std::string variable_name) const;

	//todo:templated get vector function with boost any cast

	const std::type_info& getVariableType(std::string variable_name);

protected:
	AgentDescription &agent_description;
	std::string agent_name;
	std::string agent_state;
	StateMemoryMap state_memory;
	unsigned int size;
};

#endif /* AGENTMEMORY_H_ */

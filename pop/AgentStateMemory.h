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

typedef std::map<const std::string, std::unique_ptr<std::vector<boost::any>>> StateMemoryMap;

class AgentStateMemory {
public:
	AgentStateMemory(const AgentDescription &description, const std::string agent_state) ;
	virtual ~AgentStateMemory() {}

	unsigned int getSize() const;

	void incrementSize();

	std::vector<boost::any>& getMemoryVector(const std::string variable_name);

	const std::vector<boost::any>& getReadOnlyMemoryVector(const std::string variable_name) const;

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

/*
 * AgentMemory.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include "AgentStateMemory.h"
#include <iostream>

AgentStateMemory::AgentStateMemory(const AgentDescription& description, const std::string state): agent_description(description), state_memory(), agent_state(state), size(0) {
	MemoryMap::const_iterator iter;
	const MemoryMap &m = description.getMemoryMap();

	for (iter = m.begin(); iter != m.end(); iter++){
		const std::string variable_name = iter->first;
		const std::type_info& type = iter->second;

		state_memory.insert(StateMemoryMap::value_type(variable_name, std::unique_ptr<std::vector<boost::any>> (new std::vector<boost::any>())));

    }
    //init(description);
}

/*
void AgentStateMemory::init(const AgentDescription &description){

	MemoryMap::const_iterator iter;
	const MemoryMap &m = description.getMemoryMap();

	for (iter = m.begin(); iter != m.end(); iter++){
		const std::string variable_name = iter->first;

		std::vector<boost::any>& v = getMemoryVector(variable_name);
		std::cout << "after get \n" ;
		std::vector<boost::any>::iterator it = v.begin() + getSize();
        std::cout << "after iter \n" ;
		//do the insert
		v.insert(it,0);
		std::cout << "after insert\n" ;

		  std::cout << "myvector contains:";
  for ()
    std::cout << ' ' << *it;
  std::cout << '\n';

		}
		std::cout << "init done \n" ;

}
*/

unsigned int AgentStateMemory::getSize() const{
	return size;
}


unsigned int AgentStateMemory::creatNewInstance(){

    //loop through the memory maps
    MemoryMap::const_iterator iter;
	const MemoryMap &m = agent_description.getMemoryMap();

	for (iter = m.begin(); iter != m.end(); iter++){
		const std::string variable_name = iter->first;

        //add zero values at current size
		std::vector<boost::any>& v = getMemoryVector(variable_name);
		std::vector<boost::any>::iterator it = v.begin() + size;
		//do the insert
		//get the type of the variable
		//const std::type_info& v_type = state_memory.getVariableType(variable_name);
		//creat a varibale of that type and add it to the vector
		//auto zero = 0;

		//get the default value for this varibale name from the model description and add a copy of this.
        boost::any temp =  agent_description.getDefaultValue(variable_name);

		v.insert(it,temp);


    }

    //return current size
    //increment size
    return size++;


}



std::vector<boost::any>& AgentStateMemory::getMemoryVector(const std::string variable_name) {
	StateMemoryMap::iterator iter;
	iter = state_memory.find(variable_name);

	if (iter == state_memory.end())
		throw std::runtime_error("Invalid agent memory variable");

	return *iter->second;
}

const std::vector<boost::any>& AgentStateMemory::getReadOnlyMemoryVector(const std::string variable_name) const {
	StateMemoryMap::const_iterator iter;
	iter = state_memory.find(variable_name);

	if (iter == state_memory.end())
		throw std::runtime_error("Invalid agent memory variable");

	return *iter->second;
}

const std::type_info& AgentStateMemory::getVariableType(std::string variable_name) {
		return agent_description.getVariableType(variable_name);
}

bool AgentStateMemory::isSameDescription(const AgentDescription& description) const{
	return (&description == &agent_description);
}

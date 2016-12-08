/*
 * AgentInstance.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include "AgentInstance.h"

AgentInstance::AgentInstance(AgentStateMemory& state_memory): agent_state_memory(state_memory), index(state_memory.creatNewInstance()) {

    //when you add a new instance it should add zeros to all the memory vectors at the new index
	//agent_state_memory.incrementSize();

}

AgentInstance::~AgentInstance() {}

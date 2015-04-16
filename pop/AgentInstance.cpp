/*
 * AgentInstance.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include "AgentInstance.h"

AgentInstance::AgentInstance(AgentStateMemory& state_memory): agent_state_memory(state_memory) {
	index = agent_state_memory.getSize();
	agent_state_memory.incrementSize();

}


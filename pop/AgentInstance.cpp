/**
 * @file AgentInstance.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "AgentInstance.h"

AgentInstance::AgentInstance(AgentStateMemory& state_memory, unsigned int i) : agent_state_memory(state_memory), index(i)
{

}

AgentInstance::~AgentInstance() {}

/**
 * @file AgentInstance.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <flamegpu/pop/AgentInstance.h>

AgentInstance::AgentInstance(AgentStateMemory& state_memory, unsigned int i) : index(i), agent_state_memory(state_memory) {
}

AgentInstance::~AgentInstance() {
}


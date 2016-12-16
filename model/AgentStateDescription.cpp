 /**
 * @file AgentStateDescription.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */
#include "AgentStateDescription.h"


AgentStateDescription::AgentStateDescription(const std::string state_name) : name(state_name){

}

AgentStateDescription::~AgentStateDescription() {

}

const std::string AgentStateDescription::getName() const {
	return name;
}

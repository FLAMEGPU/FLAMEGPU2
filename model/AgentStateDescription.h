/*
 * AgentStateDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef AGENTSTATEDESCRIPTION_H_
#define AGENTSTATEDESCRIPTION_H_

#include <string>

class AgentStateDescription {
public:
	AgentStateDescription(std::string name) { this->name = name; }
	virtual ~AgentStateDescription() {}

	std::string getName();

	void setName(std::string name);

private:

	std::string name;
};

#endif /* AGENTSTATEDESCRIPTION_H_ */

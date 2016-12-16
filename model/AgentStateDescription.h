 /**
 * @file AgentStateDescription.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef AGENTSTATEDESCRIPTION_H_
#define AGENTSTATEDESCRIPTION_H_

#include <string>

class AgentStateDescription {
public:
	AgentStateDescription(const std::string name);
	virtual ~AgentStateDescription();

	const std::string getName() const;

private:

	const std::string name;
};

#endif /* AGENTSTATEDESCRIPTION_H_ */

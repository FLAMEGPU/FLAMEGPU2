
 /**
 * @file  AgentStateDescription.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
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

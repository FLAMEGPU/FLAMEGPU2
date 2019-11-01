
 /**
 * @file  AgentStateDescription.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
 */

#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTSTATEDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTSTATEDESCRIPTION_H_

#include <string>

class AgentStateDescription {
 public:
    explicit AgentStateDescription(const std::string name);
    virtual ~AgentStateDescription();

    const std::string getName() const;

 private:
    const std::string name;
};

#endif // INCLUDE_FLAMEGPU_MODEL_AGENTSTATEDESCRIPTION_H_

 /**
 * @file  AgentFunctionInput.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
 */

#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONINPUT_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONINPUT_H_

#include <string>

class AgentFunctionInput {
 public:
    explicit AgentFunctionInput(const std::string input_message_name);
    virtual ~AgentFunctionInput();

    const std::string getMessageName() const;

 private:
    const std::string message_name;

// input type (single or optional)
};

#endif // INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONINPUT_H_

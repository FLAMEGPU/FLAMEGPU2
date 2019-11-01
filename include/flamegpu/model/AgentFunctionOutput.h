 /**
 * @file AgentFunctionOutput.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONOUTPUT_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONOUTPUT_H_

#include <string>

typedef enum {
    SINGLE_MESSAGE,
    OPTIONAL_MESSAGE
} FunctionOutputType;

class AgentFunctionOutput {
 public:
    explicit AgentFunctionOutput(const std::string output_message_name);

    virtual ~AgentFunctionOutput();

    const std::string getMessageName() const;

    void setFunctionOutputType(FunctionOutputType type);

    FunctionOutputType getFunctionOutoutType();

 private:
    const std::string message_name;
    FunctionOutputType type;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONOUTPUT_H_

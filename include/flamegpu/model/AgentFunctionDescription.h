 /**
 * @file AgentFunctionDescription.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_

#include <string>
#include <map>
#include <memory>

#include "flamegpu/model/AgentFunctionInput.h"
#include "flamegpu/model/AgentFunctionOutput.h"

// include the agent function syntax form the runtime api
#include "flamegpu/runtime/flame_functions_api.h"

class AgentDescription;

typedef std::map<std::string, const AgentFunctionInput&> InputsMap;
typedef std::map<std::string, const AgentFunctionOutput&> OutputsMap;
// typedef std::map<const std::string, FLAMEGPU_AGENT_FUNCTION> FunctionPointerMap;  // @question: This is not required as each agent function will have only a single pointer

class AgentFunctionDescription {
 public:
    explicit AgentFunctionDescription(const std::string function_name);

    virtual ~AgentFunctionDescription();

    const std::string getEndState() const;

    void setEndState(const std::string _end_state);

    const std::string getInitialState() const;

    void setInitialState(const std::string _initial_state);

    const std::string getName() const;

    void addInput(const AgentFunctionInput &input);

    void addOutput(const AgentFunctionOutput &output);

    const InputsMap & getInput();
    const OutputsMap & getOutput();

    const std::string getInputMessageName() const;
    const std::string getOutputMessageName() const;

    // TODO(Paul)
    // sets the function pointer by adding to the FunctionPointerMap
    void setFunction(FLAMEGPU_AGENT_FUNCTION_POINTER *p_func);

    FLAMEGPU_AGENT_FUNCTION_POINTER* getFunction() const;

    // TODO(Paul): add agent output

    void setParent(AgentDescription& agent);
    const AgentDescription& getParent() const;

    void setInputChild(AgentFunctionInput& input);
    const AgentFunctionInput& getInputChild() const;

    void setOutputChild(AgentFunctionOutput& output);
    const AgentFunctionOutput& getOutputChild() const;

    bool hasInputMessage();
    bool hasOutputMessage();

 public:
    const std::string function_name;
    std::string initial_state;
    std::string end_state;
    InputsMap inputs;
    OutputsMap outputs;
    FLAMEGPU_AGENT_FUNCTION_POINTER *func = nullptr;
    // TODO(Paul): Paul idea for parent objects
    AgentDescription* parent = 0;
    AgentFunctionInput* inputChild = 0;
    AgentFunctionOutput* outputChild = 0;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_

 /**
 * @file AgentFunctionDescription.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef AGENTFUNCTIONDESCRIPTION_H_
#define AGENTFUNCTIONDESCRIPTION_H_

#include <string>
#include <map>
#include <memory>

#include "AgentFunctionInput.h"
#include "AgentFunctionOutput.h"

class AgentDescription;

/*! FLAMEGPU function return type */
enum FLAME_GPU_AGENT_STATUS { ALIVE, DEAD };

//! FLAME GPU Agent Function pointer
typedef FLAME_GPU_AGENT_STATUS(*FLAMEGPU_AGENT_FUNCTION)(void);




typedef std::map<std::string, const AgentFunctionInput&> InputsMap;
typedef std::map<std::string, const AgentFunctionOutput&> OutputsMap;
//typedef std::map<const std::string, FLAMEGPU_AGENT_FUNCTION> FunctionPointerMap; //@question: This is not required as each agent function will have only a single pointer


class AgentFunctionDescription {
public:

	AgentFunctionDescription(const std::string function_name);

	virtual ~AgentFunctionDescription();

	const std::string getEndState() const;

	void setEndState(const std::string end_state);

	const std::string getIntialState() const;

	void setIntialState(const std::string intial_state);

	const std::string getName() const;

	void addInput(const AgentFunctionInput &input);

	void addOutput(const AgentFunctionOutput &output);

	//TODO
	//sets the function pointer by adding to the FunctionPointerMap
	void setFunction(FLAMEGPU_AGENT_FUNCTION p_func);

    FLAMEGPU_AGENT_FUNCTION getFunction() const;

	//todo: add agent output


public:

	const std::string function_name;
	std::string initial_state;
	std::string end_state;
	InputsMap inputs;
	OutputsMap outputs;
	FLAMEGPU_AGENT_FUNCTION func = NULL;
};

#endif /* AGENTFUNCTIONDESCRIPTION_H_ */

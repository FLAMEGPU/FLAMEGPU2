/*
 * flamegpu2.h
 *
 *  Created on: 19 Feb 2014
 *      Author: paul
 */

#ifndef FLAMEGPU2_H_
#define FLAMEGPU2_H_

/*
createAgent("agentname");
createAgentState("agentname", "statename", default);
createAgentVariable("agentname", "type", "variablename");


createMessage("messagename");
createMessageVariable("messagename");
setMessagePattern("messagename", MESSAGE_PATTERN);


createAgentFunction("agentname", "functioname", "start_state", "end_state", device_func_pointer);


loadPopulationData("filename.xml");
setAgentVariableData("agentname", "variable", vector<type> data);
getAgentVariableData("agentname", "variable");
setPopulationBounds("agentname", maximum_size);


executeAgentFunction("agentname", "agentfunc");
queAgentFunction("agentname", "agentfunction");
executeQueuedFunctions();
*/


#endif /* FLAMEGPU2_H_ */

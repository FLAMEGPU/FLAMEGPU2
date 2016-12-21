 /**
 * @file CUDAAgent.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef CUDAAGENT_H_
#define CUDAAGENT_H_

#include <memory>
#include <map>

//include sub classes
#include "CUDAAgentStateList.h"

//forward declare classes from other modules
class AgentDescription;
class AgentPopulation;

typedef std::map<const std::string, std::unique_ptr<CUDAAgentStateList>> CUDAStateMap;	//map of state name to CUDAAgentStateList which allocates memory on the device
typedef std::pair<const std::string, std::unique_ptr<CUDAAgentStateList>> CUDAStateMapPair;

/** \breif CUDAAgent class is used as a container for storing the GPU data of all variables in all states
 * The CUDAAgent contains a hash index which maps a variable name to a unique index. Each CUDAAgentStateList will use this hash index to map variable names to unique pointers in GPU memory space. This is required so that at runtime a variable name can be related to a unique array of data on the device. It works like a traditional hashmap however the same hashing is used for all states that an agent can be in (as agents have the same variables regardless of state).
 */
class CUDAAgent
{
public:
	CUDAAgent(const AgentDescription& description);
	virtual ~CUDAAgent(void);

	/*
	* Returns the hash list size required to store pointers to all agent variables. Current 2X the number of agent variables to minimise hash collisions.
	*/
	unsigned int getHashListSize() const;

	/*
	* Host function to get the hash index given a variable name. The hash list will have been completed by initialisation.
	*/
	int getHashIndex(const char * variable_name) const;

	const AgentDescription& getAgentDescription() const;

	/* Should be set initial population data and should only be called once as it does all the GPU device memory allocation */
	void setInitialPopulationData(const AgentPopulation& population);

	/* Can be used to override the current population data without reallocating */
	void setPopulationData(const AgentPopulation& population);

	unsigned int getMaximumListSize() const;

protected:

	void zeroAllStateVariableData();



private:
	const AgentDescription& agent_description;
	CUDAStateMap state_map;

	unsigned int* h_hashes; //host hash index table //USE SHARED POINTER??
	unsigned int* d_hashes; //device hash index table (used by runtime)

	unsigned int max_list_size; //The maximum length of the agent variable arrays based on the maximum population size passed to setPopulationData
};

#endif

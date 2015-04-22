/*
 * CUDAAgentList.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef CUDAAGENTLIST_H_
#define CUDAAGENTLIST_H_

#include <memory>
#include <vector>

#include "../model/AgentDescription.h"
#include "../pop/AgentPopulation.h"
#include "CUDAAgent.h"

struct AgentList{
	//void **d_memory;	//device array of pointers to device variable arrays
	void **h_d_memory;  //host array of pointers to device variable arrays
};

class CUDAAgentStateList {
public:
	CUDAAgentStateList(CUDAAgent& cuda_agent);
	virtual ~CUDAAgentStateList();

	void setAgentData(const AgentStateMemory &state_memory);

	void getAgentData(AgentStateMemory &state_memory);

protected:

	void allocateDeviceAgentList(AgentList** agent_list);
	
	void releaseDeviceAgentList(AgentList* agent_list);


private:
	AgentList *d_list;
	AgentList *d_swap_list;
	AgentList *d_new_list;

	unsigned int current_list_size; //???
	unsigned int max_list_size;

	CUDAAgent& agent;
};

#endif /* CUDAAGENTLIST_H_ */

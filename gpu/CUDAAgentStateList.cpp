/*
 * CUDAAgentStateList.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAAgentStateList.h"
#include "CUDAErrorChecking.h"



CUDAAgentStateList::CUDAAgentStateList(CUDAAgent& cuda_agent) : agent(cuda_agent){

	//allocate state lists
	allocateDeviceAgentList(&d_list);
	allocateDeviceAgentList(&d_swap_list);
	if (agent.getAgentDescription().requiresAgentCreation()) // Moz:  how about in 'CUDAAgentModel::simulate'?
		allocateDeviceAgentList(&d_new_list);
	else
		d_new_list = 0;

}

CUDAAgentStateList::~CUDAAgentStateList(){
	//cleanup
	releaseDeviceAgentList(d_list);
	releaseDeviceAgentList(d_swap_list);
	if (d_new_list != 0)
		releaseDeviceAgentList(d_new_list);
}

void CUDAAgentStateList::allocateDeviceAgentList(AgentList** agent_list){

	const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

	//allocate host vector to hold device pointers
	*agent_list = new AgentList();
	(*agent_list)->h_d_memory = (void**)malloc(sizeof(void*)*agent.getHashListSize());
	memset((*agent_list)->h_d_memory, 0, sizeof(void*)*agent.getHashListSize());

	//for each variable allocate a device array and register in the hash list
	unsigned int i = 0;
	for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++){
		gpuErrchk( cudaMalloc( (void**) &((*agent_list)->h_d_memory[i]), agent.getAgentDescription().getAgentVariableSize(it->first) * agent.getMaximumListSize()));
		i++;
	}



}

void CUDAAgentStateList::releaseDeviceAgentList(AgentList* agent_list){
	const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();
	//for each variable allocate a device array
	int i=0;
	for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++){
		gpuErrchk( cudaFree( &agent_list->h_d_memory[i] )); //todo suspect.....
		i++;
	}

	free(agent_list->h_d_memory);
	delete agent_list;
}


void CUDAAgentStateList::setAgentData(const AgentStateMemory &state_memory){

	//check that we are refering to the same agent description
	if (!state_memory.isSameDescription(agent.getAgentDescription())){
		throw std::runtime_error("CUDA Agent uses different agent description."); 
	}

	//copy raw agent data to device pointers
	const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

	int i=0;
	for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++){
		
		//gpuErrchk( cudaMemcpy( d_Circles_default, h_Circles_default, xmachine_Circle_SoA_size, cudaMemcpyHostToDevice));

		//gpuErrchk( cudaMalloc( (void**) &((*agent_list)->h_d_memory[i]), agent_description.getAgentVariableSize(it->first) * max_list_size));
		i++;
	}

	
}


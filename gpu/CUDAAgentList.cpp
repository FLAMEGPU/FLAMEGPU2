/*
 * CUDAAgentList.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAAgentList.h"

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
}

CUDAAgentList::CUDAAgentList(std::shared_ptr<const AgentDescription> agent_description) : agent_description_(agent_description){

	//allocate some device memory
	max_list_size = 0; //TODO

	unsigned int state_list_size = agent_description_->getMemorySize()*max_list_size;

	//allocate state lists
	allocateDeviceAgentList(&d_list);
	allocateDeviceAgentList(&d_swap_list);
	if (agent_description_->requiresAgentCreation())
		allocateDeviceAgentList(&d_new_list);
	else
		d_new_list = 0;

	//allocate hash list
	gpuErrchk( cudaMalloc( (void**) &d_hashes, sizeof(int)*max_list_size));

}

void CUDAAgentList::allocateDeviceAgentList(AgentList** agent_list){
	
	*agent_list = new AgentList();

	const MemoryMap &mem = agent_description_->getMemoryMap();

	//allocate host array of device pointers to agent functions
	(*agent_list)->h_d_memory = (void**)malloc(sizeof(void*)*agent_description_->getNumberAgentVariables());

	//for each variable allocate a device array
	int i=0;
	for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++){
		gpuErrchk( cudaMalloc( (void**) &((*agent_list)->h_d_memory[i]), agent_description_->getAgentVariableSize(it->first) * max_list_size));
		i++;
	}

}

void CUDAAgentList::setAgentData(const AgentStateMemory &state_memory){

	//iterate the 

	
}

CUDAAgentList::~CUDAAgentList() {
	// TODO Auto-generated destructor stub
}


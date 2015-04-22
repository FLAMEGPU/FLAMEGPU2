
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAAgent.h"
#include "CUDAAgentStateList.h"
#include "RuntimeHashing.h"
#include "CUDAErrorChecking.h"

CUDAAgent::CUDAAgent(const AgentDescription& description) : agent_description(description), state_map()
{
	const StateMap& sm = agent_description.getStateMap();
	StateMap::const_iterator it;

	//create map of device state lists
	for(it = sm.begin(); it != sm.end(); it++){
		state_map.insert(CUDAStateMap::value_type(it->first, std::unique_ptr<CUDAAgentStateList>( new CUDAAgentStateList(*this))));
	}

	
	//allocate hash list
	h_hashes = (unsigned int*) malloc(sizeof(unsigned int)*getHashListSize());
	gpuErrchk( cudaMalloc( (void**) &d_hashes, sizeof(unsigned int)*getHashListSize()));

	//init has list
	const MemoryMap &mem = agent_description.getMemoryMap();
	for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++){

		//save the variable hash in the host hash list
		unsigned int hash = VariableHash(it->first.c_str());
		unsigned int n = 0;
		unsigned int i = (hash) % agent_description.getNumberAgentVariables();

		while (h_hashes[i] != 0)
		{
			n += 1;
			if (n >= agent_description.getNumberAgentVariables())
			{
				throw std::runtime_error("Hash list full. This should never happen.");  
			}
			i += 1;
			if (i >= agent_description.getNumberAgentVariables())
			{
				i = 0;
			}
		}
		h_hashes[i] = hash;
	}

	//copy hash list to device
	gpuErrchk( cudaMemcpy( d_hashes, h_hashes, sizeof(unsigned int) * getHashListSize(), cudaMemcpyHostToDevice));

}


CUDAAgent::~CUDAAgent(void)
{
	free (h_hashes);
	gpuErrchk( cudaFree(d_hashes));
}


unsigned int CUDAAgent::getHashListSize(){
	return 2*agent_description.getNumberAgentVariables();
}

const AgentDescription& CUDAAgent::getAgentDescription() const {
	return agent_description;
}
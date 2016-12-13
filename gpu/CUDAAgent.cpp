#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAAgent.h"
#include "CUDAAgentStateList.h"
#include "RuntimeHashing.h"
#include "CUDAErrorChecking.h"

CUDAAgent::CUDAAgent(const AgentDescription& description) : agent_description(description), state_map(), max_list_size(0)
{

    //allocate hash list
    h_hashes = (unsigned int*) malloc(sizeof(unsigned int)*getHashListSize());
    memset(h_hashes, EMPTY_HASH_VALUE, sizeof(unsigned int)*getHashListSize());
    gpuErrchk( cudaMalloc( (void**) &d_hashes, sizeof(unsigned int)*getHashListSize()));

    //init has list
    const MemoryMap &mem = agent_description.getMemoryMap();
    for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++)
    {

        //save the variable hash in the host hash list
        unsigned int hash = VariableHash(it->first.c_str());
        unsigned int n = 0;
		unsigned int i = (hash) % getHashListSize();	// agent_description.getNumberAgentVariables();

        while (h_hashes[i] != EMPTY_HASH_VALUE)
        {
            n += 1;
            if (n >= getHashListSize())
            {
                //throw std::runtime_error("Hash list full. This should never happen.");
                throw InvalidHashList();
            }
            i += 1;
            if (i >= getHashListSize())
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


unsigned int CUDAAgent::getHashListSize() const
{
    return 2*agent_description.getNumberAgentVariables();
}

int CUDAAgent::getHashIndex(const char * variable_name) const
{
	//function resolves hash collisions
	unsigned int hash = VariableHash(variable_name);
	unsigned int n = 0;
	unsigned int i = (hash) % agent_description.getNumberAgentVariables();

	while (h_hashes[i] != EMPTY_HASH_VALUE)
	{
		if (h_hashes[i] == hash){
			return i; //return index if hash is found
		}
		n += 1;
		if (n >= getHashListSize())
		{
			//throw std::runtime_error("Hash list full. This should never happen.");
			throw InvalidHashList();
		}
		i += 1;
		if (i >= getHashListSize())
		{
			i = 0;
		}
	}
	
	throw std::runtime_error("This should be an unknown variable error. Should never occur");
	return -1; //return invalid index
}

const AgentDescription& CUDAAgent::getAgentDescription() const
{
    return agent_description;
}

void CUDAAgent::setPopulationData(const AgentPopulation& population)
{
    max_list_size = population.getMaximumPopulationSize();
    const StateMap& sm = agent_description.getStateMap();
    StateMap::const_iterator it;

    //create map of device state lists
    for(it = sm.begin(); it != sm.end(); it++)
    {
        state_map.insert(CUDAStateMap::value_type(it->first, std::unique_ptr<CUDAAgentStateList>( new CUDAAgentStateList(*this))));
    }

}

unsigned int CUDAAgent::getMaximumListSize() const
{
    return max_list_size;
}

 /**
 * @file CUDAAgentStateList.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAAgentStateList.h"

#include "CUDAAgent.h"
#include "CUDAErrorChecking.h"
#include "../pop/AgentStateMemory.h"
#include "../model/AgentDescription.h"
#include "../pop/AgentPopulation.h"


/**
* CUDAAgentStateList class
* @brief populates CUDA agent map, CUDA message map
*/
CUDAAgentStateList::CUDAAgentStateList(CUDAAgent& cuda_agent) : agent(cuda_agent)
{

    //allocate state lists
    allocateDeviceAgentList(&d_list);
    allocateDeviceAgentList(&d_swap_list);
    if (agent.getAgentDescription().requiresAgentCreation())
        allocateDeviceAgentList(&d_new_list);
	else
	{
		//set new list hash map pointers to zero
		d_new_list.d_d_memory = 0;
		d_new_list.h_d_memory = 0;
	}

}

/**
 * A destructor.
 * @brief Destroys the CUDAAgentStateList object
 */
CUDAAgentStateList::~CUDAAgentStateList()
{
	//if cleanupAllocatedData function has not been called throw an error.
	if (d_list.d_d_memory != 0){
		throw std::exception("Error cleaning up CUDAStateList data. cleanupAllocatedData function must be called!");
	}
}

void CUDAAgentStateList::cleanupAllocatedData()
{
	//clean up
	releaseDeviceAgentList(&d_list);
	d_list.d_d_memory = 0;
	d_list.h_d_memory = 0;
	releaseDeviceAgentList(&d_swap_list);
	d_swap_list.d_d_memory = 0;
	d_swap_list.h_d_memory = 0;
	if (agent.getAgentDescription().requiresAgentCreation()){
		releaseDeviceAgentList(&d_new_list);
		d_new_list.d_d_memory = 0;
		d_new_list.h_d_memory = 0;
	}

}

/**
* @brief Allocates Device agent list
* @param variable of type CUDAAgentMemoryHashMap struct type
* @return none
*/
void CUDAAgentStateList::allocateDeviceAgentList(CUDAAgentMemoryHashMap* memory_map)
{
	//we use the agents memory map to iterate the agent variables and do allocation within our GPU hash map
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

    //allocate host vector (the map) to hold device pointers
	memory_map->h_d_memory = (void**)malloc(sizeof(void*)*agent.getHashListSize());
	//set all map values to zero
	memset(memory_map->h_d_memory, 0, sizeof(void*)*agent.getHashListSize());

    //for each variable allocate a device array and register in the hash map
	for (const MemoryMapPair& mm : mem)
    {

		//get the hash index of the variable so we know what position to allocate in the map
		int hash_index = agent.getHashIndex(mm.first.c_str());

		//get the variable size from agent description
		size_t var_size = agent.getAgentDescription().getAgentVariableSize(mm.first);

		//do the device allocation at the correct index and store the pointer in the host hash map
		gpuErrchk(cudaMalloc((void**)&(memory_map->h_d_memory[hash_index]), var_size * agent.getMaximumListSize()));
    }

	//allocate device vector (the map) to hold device pointers (which have already been allocated)
	gpuErrchk(cudaMalloc((void**)&(memory_map->d_d_memory), sizeof(void*)*agent.getHashListSize()));

	//copy the host array of map pointers to the device array of map pointers
	gpuErrchk(cudaMemcpy(memory_map->d_d_memory, memory_map->h_d_memory, sizeof(void*)*agent.getHashListSize(), cudaMemcpyHostToDevice));



}

/**
* @brief Frees
* @param variable of type CUDAAgentMemoryHashMap struct type
* @return none
*/
void CUDAAgentStateList::releaseDeviceAgentList(CUDAAgentMemoryHashMap* memory_map)
{
	//we use the agents memory map to iterate the agent variables and do deallocation within our GPU hash map
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

	//for each device pointer in the map we need to free these
	for (const MemoryMapPair& mm : mem)
    {
		//get the hash index of the variable so we know what position to allocate
		int hash_index = agent.getHashIndex(mm.first.c_str());

		//free the memory on the device
		gpuErrchk(cudaFree(memory_map->h_d_memory[hash_index]));
    }

	//free the device memory map
	gpuErrchk(cudaFree(memory_map->d_d_memory));

	//free the host memory map
	free(memory_map->h_d_memory);
}

/**
* @brief
* @param variable of type CUDAAgentMemoryHashMap struct type
* @return none
*/
void CUDAAgentStateList::zeroDeviceAgentList(CUDAAgentMemoryHashMap* memory_map)
{
	//we use the agents memory map to iterate the agent variables and do deallocation within our GPU hash map
	const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

	//for each device pointer in the map we need to free these
	for (const MemoryMapPair& mm : mem)
	{
		//get the hash index of the variable so we know what position to allocate
		int hash_index = agent.getHashIndex(mm.first.c_str());

		//get the variable size from agent description
		size_t var_size = agent.getAgentDescription().getAgentVariableSize(mm.first);

		//set the memory to zero
		gpuErrchk(cudaMemset(memory_map->h_d_memory[hash_index], 0, var_size*agent.getMaximumListSize()));
	}
}

/**
* @brief
* @param AgenstStateMemory object
* @return none
* @todo
*/
void CUDAAgentStateList::setAgentData(const AgentStateMemory &state_memory)
{

    //check that we are using the same agent description
    if (!state_memory.isSameDescription(agent.getAgentDescription()))
    {
        //throw std::runtime_error("CUDA Agent uses different agent description.");
        throw InvalidCudaAgentDesc();
    }


    //copy raw agent data to device pointers
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();
	for (const MemoryMapPair& m : mem){
		//get the hash index of the variable so we know what position to allocate
		int hash_index = agent.getHashIndex(m.first.c_str());

		//get the variable size from agent description
		size_t var_size = agent.getAgentDescription().getAgentVariableSize(m.first);

		//get the vector
		const GenericMemoryVector &m_vec = state_memory.getReadOnlyMemoryVector(m.first);

		//get pointer to vector data
		const void * v_data = m_vec.getReadOnlyDataPtr();

		//set the current list size
		current_list_size = state_memory.getStateListSize();

		//TODO: copy the boost any data to GPU
		gpuErrchk(cudaMemcpy(d_list.h_d_memory[hash_index], v_data, var_size*current_list_size, cudaMemcpyHostToDevice));
	}

}

void CUDAAgentStateList::getAgentData(AgentStateMemory &state_memory)
{

	//check that we are using the same agent description
	if (!state_memory.isSameDescription(agent.getAgentDescription()))
	{
		//throw std::runtime_error("CUDA Agent uses different agent description.");
		throw InvalidCudaAgentDesc();
	}


	//copy raw agent data to device pointers
	const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();
	for (const MemoryMapPair& m : mem){
		//get the hash index of the variable so we know what position to allocate
		int hash_index = agent.getHashIndex(m.first.c_str());

		//get the variable size from agent description
		size_t var_size = agent.getAgentDescription().getAgentVariableSize(m.first);

		//get the vector
		GenericMemoryVector &m_vec = state_memory.getMemoryVector(m.first);

		//get pointer to vector data
		void * v_data = m_vec.getDataPtr();

		//check  the current list size
		if (current_list_size > state_memory.getPopulationCapacity())
			throw std::exception("Current GPU state list size exceed the state memory available!");
		

		//copy the GPU data to host
		gpuErrchk(cudaMemcpy(v_data, d_list.h_d_memory[hash_index], var_size*current_list_size, cudaMemcpyDeviceToHost));

		//set the new state list size
		state_memory.overrideStateListSize(current_list_size);
	}

}

/**
* @brief
* @param none
* @return none
*/
void CUDAAgentStateList::zeroAgentData(){
	zeroDeviceAgentList(&d_list);
	zeroDeviceAgentList(&d_swap_list);
	if (agent.getAgentDescription().requiresAgentCreation())
		zeroDeviceAgentList(&d_new_list);
}


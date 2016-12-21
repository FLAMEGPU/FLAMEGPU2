 /**
 * @file CUDAAgentList.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef CUDAAGENTLIST_H_
#define CUDAAGENTLIST_H_

#include <memory>
#include <vector>

class CUDAAgent;
class AgentStateMemory;

#define UNIFIED_GPU_MEMORY

/**
 * Stores a map of pointers to device memory locations. Must use the CUDAAgent hash table functions to access the correct index.
 */
struct CUDAAgentMemoryHashMap{
	void **d_d_memory;	//device array of pointers to device variable arrays
	void **h_d_memory;  //host array of pointers to device variable arrays
};

class CUDAAgentStateList {
public:
	CUDAAgentStateList(CUDAAgent& cuda_agent);
	virtual ~CUDAAgentStateList();

	void setAgentData(const AgentStateMemory &state_memory);

	void getAgentData(AgentStateMemory &state_memory);

	void zeroAgentData();

protected:

	/*
	 * The purpose of this function is to allocate on the device a block of memory for each variable. These vectors are stored within a hash list using the cuRVE technique so that the location of the vectors can be quickly determined at runtime by FLAME GPU functions.
	 */
	void allocateDeviceAgentList(CUDAAgentMemoryHashMap* agent_list);

	void releaseDeviceAgentList(CUDAAgentMemoryHashMap* agent_list);

	void zeroDeviceAgentList(CUDAAgentMemoryHashMap* agent_list);


private:
	CUDAAgentMemoryHashMap d_list;
	CUDAAgentMemoryHashMap d_swap_list;
	CUDAAgentMemoryHashMap d_new_list;

	unsigned int current_list_size; //???

	CUDAAgent& agent;
};

#endif /* CUDAAGENTLIST_H_ */

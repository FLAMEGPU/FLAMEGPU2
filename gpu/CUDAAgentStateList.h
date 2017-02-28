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

#include <string>
#include <memory>
#include <vector>
#include <map>

class CUDAAgent;
class AgentStateMemory;

//#define UNIFIED_GPU_MEMORY

typedef std::map <std::string, void*> CUDAMemoryMap;		//map of pointers to gpu memory for each variable name
typedef std::pair <std::string, void*> CUDAMemoryMapPair;

class CUDAAgentStateList {
public:
	CUDAAgentStateList(CUDAAgent& cuda_agent);
	virtual ~CUDAAgentStateList();

	//cant be done in destructor as it requires access to the parent CUDAAgent object
	void cleanupAllocatedData();

	void setAgentData(const AgentStateMemory &state_memory);

	void getAgentData(AgentStateMemory &state_memory);

	void* getAgentListVariablePointer(std::string variable_name);

	void zeroAgentData();

protected:

	/*
	 * The purpose of this function is to allocate on the device a block of memory for each variable. These vectors are stored within a hash list using the cuRVE technique so that the location of the vectors can be quickly determined at runtime by FLAME GPU functions.
	 */
	void allocateDeviceAgentList(CUDAMemoryMap &agent_list);

	void releaseDeviceAgentList(CUDAMemoryMap &agent_list);

	void zeroDeviceAgentList(CUDAMemoryMap &agent_list);


private:
	CUDAMemoryMap d_list;
	CUDAMemoryMap d_swap_list;
	CUDAMemoryMap d_new_list;

	unsigned int current_list_size; //???

	CUDAAgent& agent;
};

#endif /* CUDAAGENTLIST_H_ */

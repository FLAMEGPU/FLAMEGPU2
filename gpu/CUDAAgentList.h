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

struct AgentList{
	//void **d_memory;	//device array of pointers to device variable arrays
	void **h_d_memory;  //host array of pointers to device variable arrays
};


class CUDAAgentList {
public:
	CUDAAgentList(std::shared_ptr<const AgentDescription> agent_description);
	virtual ~CUDAAgentList();

	void setAgentData(const AgentStateMemory &state_memory); //cannot retain ownership of ref

	void getAgentData(AgentStateMemory &state_memory); //cannot retair ownership of ref

protected:

	void allocateDeviceAgentList(AgentList** agent_list);
	
	AgentList* freeDeviceAgentList();


	/*
	unsigned int getListSize();
	void reallocteList(unsigned int size);
	*/

private:
	int* d_hashes; //same for each list of same max_size
	AgentList *d_list;
	AgentList *d_swap_list;
	AgentList *d_new_list;
	unsigned int current_list_size;
	unsigned int max_list_size;

	std::map<std::string, AgentList> AgentListStatesMap;

	std::shared_ptr<const AgentDescription> agent_description_; //shared pointer to const description (read only)
};

#endif /* CUDAAGENTLIST_H_ */

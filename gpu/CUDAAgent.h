#ifndef CUDAAGENT_H_
#define CUDAAGENT_H_

#include <memory>
#include <map>

#include "../model/AgentDescription.h"
#include "../pop/AgentPopulation.h"

class CUDAAgentStateList; //forward declaration to avoid cirular references

typedef std::map<const std::string, std::unique_ptr<CUDAAgentStateList>> CUDAStateMap;


class CUDAAgent
{
public:
	CUDAAgent(const AgentDescription& description);
	virtual ~CUDAAgent(void);

	unsigned int getHashListSize();

	const AgentDescription& getAgentDescription() const;

	void setPopulationData(const AgentPopulation& population);

	unsigned int getMaximumListSize() const;


private:
	const AgentDescription& agent_description;
	CUDAStateMap state_map;

	unsigned int* h_hashes; //same for each list of same max_size
	unsigned int* d_hashes; //same for each list of same max_size

	unsigned int max_list_size;
};

#endif

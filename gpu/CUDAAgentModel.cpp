/*
 * CUDAAgentModel.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "CUDAAgentModel.h"

CUDAAgentModel::CUDAAgentModel(const ModelDescription& description) : model_description(description), agent_map() {
	
	//populate the CUDA agent map
	const AgentMap &am = model_description.getAgentMap();
	AgentMap::const_iterator it;

	//create new cuda agent and add to the map
	for(it = am.begin(); it != am.end(); it++){
		agent_map.insert(CUDAAgentMap::value_type(it->first, std::unique_ptr<CUDAAgent>(new CUDAAgent(it->second))));
	}

	//same for messages

	//same for functions

}

CUDAAgentModel::~CUDAAgentModel() {
	//unique pointers cleanup by automatically
}

void CUDAAgentModel::setPopulationData(AgentPopulation& population, bool overwite_exiting){
	CUDAAgentMap::iterator it;
	it = agent_map.find(population.getAgentName());

	if (it == agent_map.end()){
		throw std::runtime_error("CUDA agent not found. This should not happen.");  
	}

	//create agent state lists
	it->second->setPopulationData(population);
}

void CUDAAgentModel::simulate(const Simulation& sim){
	//check any CUDAAgents with population size == 0
	//if they have executable functions then these can be ignored
	//if they have agent creations then buffer space must be allocated for them
}


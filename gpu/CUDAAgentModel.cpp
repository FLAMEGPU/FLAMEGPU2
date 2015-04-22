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

}

CUDAAgentModel::~CUDAAgentModel() {
	//unique pointers cleanup by automatically
}


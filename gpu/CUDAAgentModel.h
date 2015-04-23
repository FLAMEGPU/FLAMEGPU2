/*
 * CUDAAgentModel.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef CUDAAGENTMODEL_H_
#define CUDAAGENTMODEL_H_

#include <memory>
#include <map>

#include "../model/ModelDescription.h"
#include "../pop/AgentPopulation.h"
#include "../sim/Simulation.h"

#include "CUDAAgent.h"

typedef std::map<const std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap;

class CUDAAgentModel {
public:
	CUDAAgentModel(const ModelDescription& description);
	virtual ~CUDAAgentModel();

	void setPopulationData(AgentPopulation& population, bool overwite_exiting = true);


	void simulate(const Simulation& sim);
private:
	const ModelDescription& model_description;
	CUDAAgentMap agent_map;

};

#endif /* CUDAAGENTMODEL_H_ */

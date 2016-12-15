/*
 * CUDAAgentModel.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 *  Last modified : 28 Nov 2016
 */

#ifndef CUDAAGENTMODEL_H_
#define CUDAAGENTMODEL_H_

#include <memory>
#include <map>

#include "../model/ModelDescription.h"
#include "../pop/AgentPopulation.h"
#include "../sim/Simulation.h"


#include "CUDAAgent.h"

typedef std::map<const std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap; //map of a number of CUDA agents by name. The CUDA agents are responsible for allocating and managing all the device memory
//typedef std::map<const std::string, std::unique_ptr<CUDAMessage>> CUDAMessageMap; /*Moz*/
//typedef std::map<const std::string, std::unique_ptr<CUDAAgentFunction>> CUDAFunctionMap; /*Moz*/

class CUDAAgentModel {
public:
	CUDAAgentModel(const ModelDescription& description);
	virtual ~CUDAAgentModel();

	void setInitialPopulationData(AgentPopulation& population);

	void setPopulationData(AgentPopulation& population);


	void simulate(const Simulation& sim);
private:
	const ModelDescription& model_description;
	CUDAAgentMap agent_map;
//	CUDAMessageMap message_map; /*Moz*/
//	CUDAFunctionMap function_map; /*Moz*/

};

#endif /* CUDAAGENTMODEL_H_ */

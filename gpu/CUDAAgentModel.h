/**
* @file CUDAAgentModel.h
* @authors Paul
* @date 5 Mar 2014
* @brief
*
* @see
* @warning
*/

#ifndef CUDAAGENTMODEL_H_
#define CUDAAGENTMODEL_H_

#include <memory>
#include <map>
#include <cuda_runtime.h>


//include sub classes
#include "CUDAAgent.h"
#include "../runtime/cuRVE/cuRVEInstance.h"

//forward declare classes from other modules
class ModelDescription;
class Simulation;

typedef std::map<const std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap; //map of a number of CUDA agents by name. The CUDA agents are responsible for allocating and managing all the device memory
//typedef std::map<const std::string, std::unique_ptr<CUDAMessage>> CUDAMessageMap; /*Moz*/
//typedef std::map<const std::string, std::unique_ptr<CUDAAgentFunction>> CUDAFunctionMap; /*Moz*/

class CUDAAgentModel
{
public:
    CUDAAgentModel(const ModelDescription& description);
    virtual ~CUDAAgentModel();

    void setInitialPopulationData(AgentPopulation& population);

    void setPopulationData(AgentPopulation& population);

    void getPopulationData(AgentPopulation& population);

    void init(void);

    void addSimulation(const Simulation& sim);

	//TODO: Is this needed? Probably not as it is the same as simulate. Do however require a SimulateN() for simulate a number of iterations.
    void step(const Simulation& sim);

    void simulate(const Simulation& sim);

	const CUDAAgent& getCUDAAgent(std::string agent_name) const;

private:
    const ModelDescription& model_description;
    CUDAAgentMap agent_map;
	cuRVEInstance &curve;

//	CUDAMessageMap message_map; /*Moz*/
//	CUDAFunctionMap function_map; /*Moz*/

};

#endif /* CUDAAGENTMODEL_H_ */

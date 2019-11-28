/**
 * @file CUDAAgentModel.h
 * @author
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
 */

#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_

#include <cuda_runtime.h>

#include <memory>
#include <map>
#include<string>

// include sub classes
#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/runtime/cuRVE/cuRVEInstance.h"  // @todo move to externals
#include "flamegpu/runtime/flamegpu_host_api.h"

// forward declare classes from other modules
class ModelDescription;
class Simulation;
class RandomManager;

typedef std::map<const std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap;  // map of a number of CUDA agents by name. The CUDA agents are responsible for allocating and managing all the device memory
typedef std::map<const std::string, std::unique_ptr<CUDAMessage>> CUDAMessageMap;
// typedef std::map<const std::string, std::unique_ptr<CUDAAgentFunction>> CUDAFunctionMap; /*Moz*/

class CUDAAgentModel {
 public:
    explicit CUDAAgentModel(const ModelDescription& description);
    virtual ~CUDAAgentModel();

    void setInitialPopulationData(AgentPopulation& population);

    void setPopulationData(AgentPopulation& population);

    void getPopulationData(AgentPopulation& population);

    void init(void);

    // TODO: Is this needed? Probably not as it is the same as simulate. Do however require a SimulateN() for simulate a number of iterations.
    /**
     * @return Returns False if an exit condition has requested exit
     */
    bool step(const Simulation& sim);

    void simulate(const Simulation& sim);

    const CUDAAgent& getCUDAAgent(std::string agent_name) const;
    const CUDAMessage& getCUDAMessage(std::string message_name) const;

 private:
    const ModelDescription& model_description;
    CUDAAgentMap agent_map;
    Curve &curve;

    CUDAMessageMap message_map;
    /**
     * One instance of host api is used for entire CUDA model
     */
    FLAMEGPU_HOST_API host_api;
    /**
     * Resizes device random array during step()
     */
    RandomManager &rng;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_

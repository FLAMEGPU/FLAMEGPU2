#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_

#include "flamegpu/sim/Simulation.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"

class CUDAAgentModel : public Simulation {
    typedef std::map<std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap;  // map of a number of CUDA agents by name. The CUDA agents are responsible for allocating and managing all the device memory
    typedef std::map<std::string, std::unique_ptr<CUDAMessage>> CUDAMessageMap;
 public:
    CUDAAgentModel(const ModelDescription& model);
    ~CUDAAgentModel();
    bool step() override;
    void simulate() override;
    void setInitialPopulationData(AgentPopulation& population) override;
    void setPopulationData(AgentPopulation& population) override;
    void getPopulationData(AgentPopulation& population) override;
    const CUDAAgent& getCUDAAgent(std::string agent_name) const;
    AgentInterface &getAgent(const std::string &name) override;
    const CUDAMessage& getCUDAMessage(std::string message_name) const;
 protected:
    bool checkArgs_derived(int argc, const char** argv) override;
    void printHelp_derived() override;
    void _initialise() override;
 private:
    unsigned int device_id;
    CUDAAgentMap agent_map;
    Curve &curve;

    CUDAMessageMap message_map;
    /**
    * Resizes device random array during step()
    */
    RandomManager &rng;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_

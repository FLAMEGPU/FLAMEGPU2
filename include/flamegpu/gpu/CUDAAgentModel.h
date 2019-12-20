#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_

#include <map>
#include <memory>
#include <string>

#include "flamegpu/sim/Simulation.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"

class CUDAAgentModel : public Simulation {
    typedef std::map<std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap;  // map of a number of CUDA agents by name. The CUDA agents are responsible for allocating and managing all the device memory
    typedef std::map<std::string, std::unique_ptr<CUDAMessage>> CUDAMessageMap;

 public:
    struct Config {
         int device_id = 0;
    };
    explicit CUDAAgentModel(const ModelDescription& model);
    virtual ~CUDAAgentModel();  // Virtual required by gmock
    bool step() override;
    void simulate() override;
    void setPopulationData(AgentPopulation& population) override;
    void getPopulationData(AgentPopulation& population) override;
    CUDAAgent& getCUDAAgent(const std::string &agent_name) const;
    AgentInterface &getAgent(const std::string &name) override;
    CUDAMessage& getCUDAMessage(const std::string &message_name) const;
    Config &CUDAConfig();
    const Config &getCUDAConfig() const;

 protected:
    void applyConfig_derived() override;
    bool checkArgs_derived(int argc, const char** argv, int &i) override;
    void printHelp_derived() override;
    void resetDerivedConfig() override;

 private:
    CUDAAgentMap agent_map;
    Curve &curve;
    Config config;
    CUDAMessageMap message_map;
    /**
    * Resizes device random array during step()
    */
    RandomManager &rng;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_

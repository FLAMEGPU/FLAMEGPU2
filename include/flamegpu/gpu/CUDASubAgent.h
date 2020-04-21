#ifndef INCLUDE_FLAMEGPU_GPU_CUDASUBAGENT_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASUBAGENT_H_

#include <memory>

#include "flamegpu/gpu/CUDAAgent.h"

struct AgentData;
struct SubAgentData;

/**
 * CUDAAgent for Agents within SubModels
 * This is a largely rewritten version, whereby the CUDASubAgent matches the size of 'master_agent' at all times
 * Some or all CUDAAgentStateLists may depend on state lists within master_agent
 */
class CUDASubAgent : public CUDAAgent {
 public:
    /**
     * Constructs a new CUDASubAgent
     * @param description Agent description from the model description hierarchy
     * @param master_agent The CUDAAgent which this Agent is bound to and inherits from
     * @param mapping Mapping information, which lists the state and variable mappings
     */
    CUDASubAgent(const AgentData &description, const CUDAAgentModel& cuda_model, const std::unique_ptr<CUDAAgent> &master_agent, const std::shared_ptr<SubAgentData> &mapping);

    /**
     * Instead returns max size from master_agent, as we copy master_agent size throughout
     */
    unsigned int getMaximumListSize() const override;

 private:
    const std::unique_ptr<CUDAAgent> &master_agent;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASUBAGENT_H_

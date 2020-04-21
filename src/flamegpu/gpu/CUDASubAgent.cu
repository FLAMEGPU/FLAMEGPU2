#include "flamegpu/gpu/CUDASubAgent.h"

#include "flamegpu/gpu/CUDASubAgentStateList.h"
#include "flamegpu/model/SubModelData.h"

CUDASubAgent::CUDASubAgent(const AgentData &description, const CUDAAgentModel& cuda_model, const std::unique_ptr<CUDAAgent> &_master_agent, const std::shared_ptr<SubAgentData> &mapping)
    : CUDAAgent(description, cuda_model)
    , master_agent(_master_agent) {
    // Replace all mapped states in the statemap
    for (const auto &s_pair : mapping->states) {
        // Replace regular CUDAAgentStatelist with CUDASubAgentStateList
        state_map.at(s_pair.first) = std::make_shared<CUDASubAgentStateList>(*this, master_agent->getAgentStateList(s_pair.second), mapping);
    }
}

unsigned int CUDASubAgent::getMaximumListSize() const {
    return master_agent->getMaximumListSize();
}

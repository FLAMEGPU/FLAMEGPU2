#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/runtime/flamegpu_host_agent_api.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

FLAMEGPU_HOST_API::FLAMEGPU_HOST_API(CUDAAgentModel &_agentModel)
    :agentModel(_agentModel),
    d_cub_temp(nullptr),
    d_cub_temp_size(0),
    d_output_space(nullptr),
    d_output_space_size(0) {
}
FLAMEGPU_HOST_API::~FLAMEGPU_HOST_API() {
    if (d_cub_temp) {
        cudaFree(d_cub_temp);
        d_cub_temp_size = 0;
    }
    if (d_output_space_size) {
        cudaFree(d_output_space);
        d_output_space_size = 0;
    }
}

// FLAMEGPU_HOST_AGENT_API FLAMEGPU_HOST_API::agent(const std::string &agent_name) {
//     return FLAMEGPU_HOST_AGENT_API(*this, agentModel.getCUDAAgent(agent_name));
// }

FLAMEGPU_HOST_AGENT_API FLAMEGPU_HOST_API::agent(const std::string &agent_name, const std::string &stateName) {
    return FLAMEGPU_HOST_AGENT_API(*this, agentModel.getCUDAAgent(agent_name), stateName);
}

bool FLAMEGPU_HOST_API::tempStorageRequiresResize(const CUB_Config &cc, const unsigned int &items) {
    auto lao = cub_largestAllocatedOp.find(cc);
    if (lao != cub_largestAllocatedOp.end()) {
        if (lao->second < items)
            return false;
    }
    return true;
}
void FLAMEGPU_HOST_API::resizeTempStorage(const CUB_Config &cc, const unsigned int &items, const size_t &newSize) {
    if (newSize > d_cub_temp_size) {
        if (d_cub_temp) {
            cudaFree(d_cub_temp);
        }
        cudaMalloc(&d_cub_temp, newSize);
        d_cub_temp_size = newSize;
    }
    assert(tempStorageRequiresResize(cc, items));
    cub_largestAllocatedOp[cc] = items;
}


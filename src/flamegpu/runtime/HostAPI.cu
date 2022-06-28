#include "flamegpu/runtime/HostAPI.h"

#include <map>
#include "flamegpu/runtime/agent/HostAgentAPI.cuh"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/simulation/Simulation.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {

HostAPI::HostAPI(CUDASimulation &_agentModel,
    detail::RandomManager& rng,
    detail::CUDAScatter &_scatter,
    const AgentOffsetMap &_agentOffsets,
    AgentDataMap &_agentData,
    const std::shared_ptr<detail::EnvironmentManager>& env,
    detail::CUDAMacroEnvironment &macro_env,
    CUDADirectedGraphMap &directed_graph_map,
    const unsigned int _streamId,
    cudaStream_t _stream)
    : random(rng)
    , environment(_agentModel.getInstanceID(), env, macro_env, directed_graph_map, _scatter, _streamId, _stream)
    , agentModel(_agentModel)
    , d_output_space(nullptr)
    , d_output_space_size(0)
    , agentOffsets(_agentOffsets)
    , agentData(_agentData)
    , scatter(_scatter)
    , streamId(_streamId)
    , stream(_stream) { }

HostAPI::~HostAPI() {
    // @todo - cuda is not allowed in destructor
    if (d_output_space_size) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_output_space));
        d_output_space_size = 0;
    }
}

HostAgentAPI HostAPI::agent(const std::string &agent_name, const std::string &state_name) {
    auto agt = agentData.find(agent_name);
    if (agt == agentData.end()) {
        THROW exception::InvalidAgent("Agent '%s' was not found in model description hierarchy.\n", agent_name.c_str());
    }
    auto state = agt->second.find(state_name);
    if (state == agt->second.end()) {
        THROW exception::InvalidAgentState("Agent '%s' in model description hierarchy does not contain state '%s'.\n", agent_name.c_str(), state_name.c_str());
    }
    return HostAgentAPI(*this, agentModel.getCUDAAgent(agent_name), state_name, agentOffsets.at(agent_name), state->second);
}

/**
 * Access the current stepCount
 * Sepearate implementation to avoid dependency loop with cuda agent model.
 * @return the current step count, 0 indexed unsigned.
 */
unsigned int HostAPI::getStepCounter() const {
    return agentModel.getStepCounter();
}

}  // namespace flamegpu

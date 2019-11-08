#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_

#include <string>
#include <cub/cub.cuh>

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/flamegpu_host_api.h"

class FLAMEGPU_HOST_AGENT_API {
 public:
    FLAMEGPU_HOST_AGENT_API(FLAMEGPU_HOST_API &_api, const CUDAAgent &_agent, const std::string &_stateName="default")
        :api(_api),
        agent(_agent),
        hasState(true),
        stateName(_stateName) {
         
    }
    /*FLAMEGPU_HOST_AGENT_API(const CUDAAgent &_agent)
        :agent(_agent),
        hasState(false),
        stateName("") {

    }*/
    /**
     * Wraps cub::DeviceReduce::Sum()
     */
    template<typename T>
    T sum(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Min()
     */
    template<typename T>
    T min(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Max()
     */
    template<typename T>
    T max(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Reduce()
     */
    // template<typename T>
    // T reduce(const std::string &variable, ? reductionOperator) const;

 private:
    FLAMEGPU_HOST_API &api;
    const CUDAAgent &agent;
    bool hasState;
    const std::string stateName;
};



template<typename T>
T FLAMEGPU_HOST_AGENT_API::sum(const std::string &variable) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(T) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of sum()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    //Resize cub storage
    //TODO: Move inside host_api
    size_t tempByte = 0;
    cub::DeviceReduce::Sum(nullptr, tempByte, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
    if(tempByte>api.d_cub_temp_size) {
        if(api.d_cub_temp) {
            cudaFree(api.d_cub_temp);
        }
        cudaMalloc(&api.d_cub_temp, tempByte);
        api.d_cub_temp_size = tempByte;
    }
    //Resize output storage
    //TODO: Move inside host_api
    if (sizeof(T)>api.d_output_space_size) {
        if (api.d_output_space_size) {
            cudaFree(api.d_output_space);
        }
        cudaMalloc(&api.d_output_space, sizeof(T));
        api.d_output_space_size = tempByte;
    }
    cub::DeviceReduce::Sum(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
    T rtn;
    cudaMemcpy(&rtn, api.d_output_space, sizeof(T), cudaMemcpyDeviceToHost);
    return rtn;
}
template<typename T>
T FLAMEGPU_HOST_AGENT_API::min(const std::string &variable) const {
    return T();
}
template<typename T>
T FLAMEGPU_HOST_AGENT_API::max(const std::string &variable) const {
    return T();
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_

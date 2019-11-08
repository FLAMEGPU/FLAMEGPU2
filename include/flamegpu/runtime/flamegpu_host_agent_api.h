#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_

#include <cub/cub.cuh>
#include <algorithm>
#include <string>


#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/flamegpu_host_api.h"

class FLAMEGPU_HOST_AGENT_API {
 public:
    FLAMEGPU_HOST_AGENT_API(FLAMEGPU_HOST_API &_api, const CUDAAgent &_agent, const std::string &_stateName = "default")
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
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::SUM, typeid(T).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Sum(nullptr, tempByte, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<T>();
    cub::DeviceReduce::Sum(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
    T rtn;
    cudaMemcpy(&rtn, api.d_output_space, sizeof(T), cudaMemcpyDeviceToHost);
    return rtn;
}
template<typename T>
T FLAMEGPU_HOST_AGENT_API::min(const std::string &variable) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(T) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of sum()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::MIN, typeid(T).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Min(nullptr, tempByte, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<T>();
    cub::DeviceReduce::Min(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
    T rtn;
    cudaMemcpy(&rtn, api.d_output_space, sizeof(T), cudaMemcpyDeviceToHost);
    return rtn;
}
template<typename T>
T FLAMEGPU_HOST_AGENT_API::max(const std::string &variable) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(T) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of sum()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::MAX, typeid(T).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Max(nullptr, tempByte, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<T>();
    cub::DeviceReduce::Max(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<T*>(var_ptr), reinterpret_cast<T*>(api.d_output_space), static_cast<int>(agentCount));
    T rtn;
    cudaMemcpy(&rtn, api.d_output_space, sizeof(T), cudaMemcpyDeviceToHost);
    return rtn;
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_

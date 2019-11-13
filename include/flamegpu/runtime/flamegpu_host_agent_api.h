#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#ifdef _MSC_VER
#pragma warning(push, 3)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif
#include <algorithm>
#include <string>
#include <vector>


#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

#define FLAMEGPU_CUSTOM_REDUCTION(funcName, a, b) \
struct funcName ## _impl { \
    template <typename T> \
    __device__ __forceinline__ T operator()(const T &, const T &) const;\
}; \
funcName ## _impl funcName; \
template <typename T> \
__device__ __forceinline__ T funcName ## _impl::operator()(const T & a, const T & b) const

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
     * @param variable The agent variable to perform the sum reduction across
     */
    template<typename InT>
    InT sum(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Sum()
     * @param variable The agent variable to perform the sum reduction across
     * @note The template arg, 'OutT' can be used if the sum is expected to exceed the representation of the type being summed
     */
    template<typename InT, typename OutT>
    OutT sum(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Min()
     * @param variable The agent variable to perform the min reduction across
     */
    template<typename InT>
    InT min(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Max()
     * @param variable The agent variable to perform the max reduction across
     */
    template<typename InT>
    InT max(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Reduce(), to perform a reduction with a custom operator
     * @param variable The agent variable to perform the reduction across
     * @param reductionOperator The custom reduction function
     * @param init Initial value of the reduction
     */
    template<typename InT, typename reductionOperatorT>
    InT reduce(const std::string &variable, reductionOperatorT reductionOperator, const InT&init) const;
    /**
     * Wraps cub::DeviceHistogram::HistogramEven()
     * @param variable The agent variable to perform the reduction across
     * @param histogramBins The number of bins the histogram should have
     * @param lowerBound The lower sample value boundary of lowest bin
     * @param upperBound The upper sample value boundary of upper bin
     */
    template<typename InT>
    std::vector<unsigned int> histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const;
    template<typename InT, typename OutT>
    std::vector<OutT> histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const;

 private:
    FLAMEGPU_HOST_API &api;
    const CUDAAgent &agent;
    bool hasState;
    const std::string stateName;
};

//
// Implementation
//

template<typename InT>
InT FLAMEGPU_HOST_AGENT_API::sum(const std::string &variable) const {
    return sum<InT, InT>(variable);
}
template<typename InT, typename OutT>
OutT FLAMEGPU_HOST_AGENT_API::sum(const std::string &variable) const {
    static_assert(sizeof(InT) <= sizeof(OutT), "Template arg OutT should not be of a smaller size than InT");
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(InT) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of sum()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::SUM, typeid(OutT).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Sum(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), static_cast<int>(agentCount));
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<OutT>();
    cub::DeviceReduce::Sum(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), static_cast<int>(agentCount));
    gpuErrchkLaunch();
    OutT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(OutT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT>
InT FLAMEGPU_HOST_AGENT_API::min(const std::string &variable) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(InT) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of min()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::MIN, typeid(InT).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Min(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount));
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<InT>();
    cub::DeviceReduce::Min(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount));
    gpuErrchkLaunch();
    InT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT>
InT FLAMEGPU_HOST_AGENT_API::max(const std::string &variable) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(InT) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of max()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::MAX, typeid(InT).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Max(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount));
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<InT>();
    cub::DeviceReduce::Max(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount));
    gpuErrchkLaunch();
    InT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT, typename reductionOperatorT>
InT FLAMEGPU_HOST_AGENT_API::reduce(const std::string &variable, reductionOperatorT reductionOperator, const InT &init) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(InT) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of reduce()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::CUSTOM_REDUCE, typeid(InT).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Reduce(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount), reductionOperator, init);
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<InT>();
    cub::DeviceReduce::Reduce(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount), reductionOperator, init);
    gpuErrchkLaunch();
    InT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT>
std::vector<unsigned int> FLAMEGPU_HOST_AGENT_API::histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const {
    return histogramEven<InT, unsigned int>(variable, histogramBins, lowerBound, upperBound);
}
template<typename InT, typename OutT>
std::vector<OutT> FLAMEGPU_HOST_AGENT_API::histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const {
    assert(lowerBound < upperBound);
    const auto &agentDesc = agent.getAgentDescription();
    if (typeid(InT) != agentDesc.getVariableType(variable))
        throw InvalidVarType("variable type does not match type of histogramEven()");
    const auto &stateAgent = agent.getAgentStateList(stateName);
    void *var_ptr = stateAgent->getAgentListVariablePointer(variable);
    const auto agentCount = stateAgent->getCUDAStateListSize();
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::HISTOGRAM_EVEN, histogramBins * sizeof(OutT) };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceHistogram::HistogramEven(nullptr, tempByte,
            reinterpret_cast<InT*>(var_ptr), reinterpret_cast<int*>(api.d_output_space), histogramBins + 1, lowerBound, upperBound, static_cast<int>(agentCount));
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<OutT>(histogramBins);
    cub::DeviceHistogram::HistogramEven(api.d_cub_temp, api.d_cub_temp_size,
        reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), histogramBins + 1, lowerBound, upperBound, static_cast<int>(agentCount));
    gpuErrchkLaunch();
    std::vector<OutT> rtn(histogramBins);
    gpuErrchk(cudaMemcpy(rtn.data(), api.d_output_space, histogramBins * sizeof(OutT), cudaMemcpyDeviceToHost));
    return rtn;
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_

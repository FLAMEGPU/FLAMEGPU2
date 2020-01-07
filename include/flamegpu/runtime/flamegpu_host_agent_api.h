#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#ifdef _MSC_VER
#pragma warning(push, 3)
#include <cub/cub.cuh>
#pragma warning(pop)
#pragma warning(push, 2)
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#endif

#include <algorithm>
#include <string>
#include <vector>
#include <functional>


#include "flamegpu/sim/AgentInterface.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

#define FLAMEGPU_CUSTOM_REDUCTION(funcName, a, b)\
struct funcName ## _impl {\
 public:\
    template <typename OutT>\
    struct binary_function : public std::binary_function<OutT, OutT, OutT> {\
        __device__ __forceinline__ OutT operator()(const OutT &a, const OutT &b) const;\
    };\
};\
funcName ## _impl funcName;\
template <typename OutT>\
__device__ __forceinline__ OutT funcName ## _impl::binary_function<OutT>::operator()(const OutT & a, const OutT & b) const

#define FLAMEGPU_CUSTOM_TRANSFORM(funcName, a)\
struct funcName ## _impl {\
 public:\
    template<typename InT, typename OutT>\
    struct unary_function : public std::unary_function<InT, OutT> {\
        __host__ __device__ OutT operator()(const InT &a) const;\
    };\
};\
funcName ## _impl funcName;\
template<typename InT, typename OutT>\
__device__ __forceinline__ OutT funcName ## _impl::unary_function<InT, OutT>::operator()(const InT &a) const

class FLAMEGPU_HOST_AGENT_API {
 public:
    FLAMEGPU_HOST_AGENT_API(FLAMEGPU_HOST_API &_api, AgentInterface &_agent, const std::string &_stateName = "default")
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
    /*
     * Returns the number of agents in this state
     */
    unsigned int count();
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
     * Wraps thrust::count(), to count the number of occurences of the provided value
     * @param variable The agent variable to perform the count reduction across
     * @param value The value to count occurences of
     */
    template<typename InT>
    unsigned int count(const std::string &variable, const InT &value);
    /**
     * Wraps cub::DeviceHistogram::HistogramEven()
     * @param variable The agent variable to perform the reduction across
     * @param histogramBins The number of bins the histogram should have
     * @param lowerBound The (inclusive) lower sample value boundary of lowest bin
     * @param upperBound The (exclusive) upper sample value boundary of upper bin
     * @note 2nd template arg can be used if calculation requires higher bit type to avoid overflow
     */
    template<typename InT>
    std::vector<unsigned int> histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const;
    template<typename InT, typename OutT>
    std::vector<OutT> histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const;
    /**
     * Wraps cub::DeviceReduce::Reduce(), to perform a reduction with a custom operator
     * @param variable The agent variable to perform the reduction across
     * @param reductionOperator The custom reduction function
     * @param init Initial value of the reduction
     */
    template<typename InT, typename reductionOperatorT>
    InT reduce(const std::string &variable, reductionOperatorT reductionOperator, const InT &init) const;
    /**
     * Wraps thrust::transformReduce(), to perform a custom transform on values before performing a custom reduction
     * @param variable The agent variable to perform the reduction across
     * @param transformOperator The custom unary transform function
     * @param reductionOperator The custom binary reduction function
     * @param init Initial value of the reduction
     */
    // template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
    // OutT transformReduce(const std::string &variable, std::unary_function<OutT, OutT> transformOperator, std::binary_function<OutT, OutT, OutT>, const OutT &init) const;
    template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
    OutT transformReduce(const std::string &variable, transformOperatorT transformOperator, reductionOperatorT reductionOperator, const OutT &init) const;

 private:
    FLAMEGPU_HOST_API &api;
    AgentInterface &agent;
    bool hasState;
    const std::string stateName;
};

inline unsigned FLAMEGPU_HOST_AGENT_API::count() {
    return agent.getStateSize(stateName);
}

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
    if (std::type_index(typeid(InT)) != agentDesc.description->getVariableType(variable)) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::sum(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
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
    if (std::type_index(typeid(InT)) != agentDesc.description->getVariableType(variable)) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::min(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
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
    if (std::type_index(typeid(InT)) != agentDesc.description->getVariableType(variable)) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::max(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
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
template<typename InT>
unsigned int FLAMEGPU_HOST_AGENT_API::count(const std::string &variable, const InT &value) {
    const auto &agentDesc = agent.getAgentDescription();
    if (std::type_index(typeid(InT)) != agentDesc.description->getVariableType(variable)) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::count(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Cast return from ptrdiff_t (int64_t) to (uint32_t)
    unsigned int rtn = static_cast<unsigned int>(thrust::count(thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr)), thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr) + agentCount), value));
    gpuErrchkLaunch();
    return rtn;
}
template<typename InT>
std::vector<unsigned int> FLAMEGPU_HOST_AGENT_API::histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const {
    return histogramEven<InT, unsigned int>(variable, histogramBins, lowerBound, upperBound);
}
template<typename InT, typename OutT>
std::vector<OutT> FLAMEGPU_HOST_AGENT_API::histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const {
    if (lowerBound >= upperBound) {
        THROW InvalidArgument("lowerBound (%s) must be lower than < upperBound (%s) in FLAMEGPU_HOST_AGENT_API::histogramEven().",
            std::to_string(lowerBound).c_str(), std::to_string(upperBound).c_str());
    }
    const auto &agentDesc = agent.getAgentDescription();
    if (std::type_index(typeid(InT)) != agentDesc.description->getVariableType(variable)) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::histogramEven(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
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
template<typename InT, typename reductionOperatorT>
InT FLAMEGPU_HOST_AGENT_API::reduce(const std::string &variable, reductionOperatorT /*reductionOperator*/, const InT &init) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (std::type_index(typeid(InT)) != agentDesc.description->getVariableType(variable)) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::reduce(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Check if we need to resize cub storage
    FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::CUSTOM_REDUCE, typeid(InT).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        cub::DeviceReduce::Reduce(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space),
            static_cast<int>(agentCount), typename reductionOperatorT::template binary_function<InT>(), init);
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<InT>();
    cub::DeviceReduce::Reduce(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space),
        static_cast<int>(agentCount), typename reductionOperatorT::template binary_function<InT>(), init);
    gpuErrchkLaunch();
    InT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
OutT FLAMEGPU_HOST_AGENT_API::transformReduce(const std::string &variable, transformOperatorT /*transformOperator*/, reductionOperatorT /*reductionOperator*/, const OutT &init) const {
    const auto &agentDesc = agent.getAgentDescription();
    if (std::type_index(typeid(InT)) != agentDesc.description->getVariableType(variable)) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::transformReduce(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // auto a = is_1_0<InT, OutT>();
    // auto b = my_sum<OutT>();
    // <thrust::device_ptr<InT>, std::unary_function<InT, OutT>, OutT, std::binary_function<OutT, OutT, OutT>>
    OutT rtn = thrust::transform_reduce(thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr)), thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr) + agentCount),
        typename transformOperatorT::template unary_function<InT, OutT>(), init, typename reductionOperatorT::template binary_function<OutT>());
    gpuErrchkLaunch();
    return rtn;
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_

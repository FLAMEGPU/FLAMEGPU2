#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_
#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#endif

#include <algorithm>
#include <string>
#include <vector>
#include <functional>


#include "flamegpu/sim/AgentInterface.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/gpu/CUDAAgent.h"

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

class HostAgentInstance {
 public:
    HostAgentInstance(FLAMEGPU_HOST_API &_api, AgentInterface &_agent, const std::string &_stateName = "default")
        : api(_api),
        agent(_agent),
        hasState(true),
        stateName(_stateName) {
    }
    /*HostAgentInstance(const CUDAAgent &_agent)
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
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT>
    InT sum(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Sum()
     * @param variable The agent variable to perform the sum reduction across
     * @tparam OutT The template arg, 'OutT' can be used if the sum is expected to exceed the representation of the type being summed
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT, typename OutT>
    OutT sum(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Min()
     * @param variable The agent variable to perform the lowerBound reduction across
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT>
    InT min(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Max()
     * @param variable The agent variable to perform the upperBound reduction across
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT>
    InT max(const std::string &variable) const;
    /**
     * Wraps thrust::count(), to count the number of occurences of the provided value
     * @param variable The agent variable to perform the count reduction across
     * @param value The value to count occurences of
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
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
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
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
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT, typename reductionOperatorT>
    InT reduce(const std::string &variable, reductionOperatorT reductionOperator, const InT &init) const;
    /**
     * Wraps thrust::transformReduce(), to perform a custom transform on values before performing a custom reduction
     * @param variable The agent variable to perform the reduction across
     * @param transformOperator The custom unary transform function
     * @param reductionOperator The custom binary reduction function
     * @param init Initial value of the reduction
     * @tParam InT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
    OutT transformReduce(const std::string &variable, transformOperatorT transformOperator, reductionOperatorT reductionOperator, const OutT &init) const;
    // template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
    // OutT transformReduce(const std::string &variable, std::unary_function<OutT, OutT> transformOperator, std::binary_function<OutT, OutT, OutT>, const OutT &init) const;
    enum Order {Asc, Desc};
    /**
     * Sorts agents according to the named variable
     * @param variable The agent variable to sort the agents according to
     * @param order Whether the agents should be sorted in ascending or descending order of the variable
     * @param beginBit Advanced Option, see note
     * @param endBit Advanced Option, see note
     * @tParam VarT The type of the variable as specified in the model description hierarchy
     * @throws UnsupportedVarType Array variables are not supported
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note An optional bit subrange [begin_bit, end_bit) of differentiating variable bits can be specified. This can reduce overall sorting overhead and yield a corresponding performance improvement.
     * @note The sort provides no guarantee of stability
     */
    template<typename VarT>
    void sort(const std::string &variable, Order order, int beginBit = 0, int endBit = sizeof(VarT)*8);
    /**
     * Sort agents according to two variables e.g. [1:c, 3:b, 1:b, 1:a] -> [1:a, 1:b, 1:c, 3:b]
     * @param variable1 This variable will be the main direction that agents are sorted
     * @param order1 The order that variable 1 should be sorted according to
     * @param variable2 Agents with equal variable1's, will be sorted according this this variable
     * @param order2 The order that variable 2 should be sorted according to
     * @throws UnsupportedVarType Array variables are not supported
     * @tParam Var1T The type of variable1 as specified in the model description hierarchy
     * @tParam Var2T The type of variable2 as specified in the model description hierarchy
     * @throws InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename Var1T, typename Var2T>
    void sort(const std::string &variable1, Order order1, const std::string &variable2, Order order2);

 private:
    /**
     * Fills the provided device buffer with consecutive integers
     * @param d_buffer Device pointer to buffer to be filled
     * @param length Length of the buffer (how many unsigned ints can it hold)
     */
    static void fillTIDArray(unsigned int *d_buffer, const unsigned int &length, const cudaStream_t &stream);
    /**
     * Sorts a buffer by the positions array, used for multi variable agent sorts
     * @param dest Device pointer to buffer for sorted data to be placed
     * @param src Device pointer to buffer to be sorted
     * @param typeLen sizeof the type stored in the buffer (e.g. sizeof(int))
     * @param length Length of the buffer (how many items it can it hold)
     */
    static void sortBuffer(void *dest, void*src, unsigned int *position, const size_t &typeLen, const unsigned int &length, const cudaStream_t &stream);
    FLAMEGPU_HOST_API &api;
    AgentInterface &agent;
    bool hasState;
    const std::string stateName;
};

inline unsigned HostAgentInstance::count() {
    return agent.getStateSize(stateName);
}

//
// Implementation
//

template<typename InT>
InT HostAgentInstance::sum(const std::string &variable) const {
    return sum<InT, InT>(variable);
}
template<typename InT, typename OutT>
OutT HostAgentInstance::sum(const std::string &variable) const {
    static_assert(sizeof(InT) <= sizeof(OutT), "Template arg OutT should not be of a smaller size than InT");
    const auto &agentDesc = agent.getAgentDescription();

    std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::sum() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW InvalidVarType("Wrong variable type passed to HostAgentInstance::sum(). "
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
        gpuErrchk(cub::DeviceReduce::Sum(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), static_cast<int>(agentCount)));
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<OutT>();
    gpuErrchk(cub::DeviceReduce::Sum(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), static_cast<int>(agentCount)));
    gpuErrchkLaunch();
    OutT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(OutT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT>
InT HostAgentInstance::min(const std::string &variable) const {
    const auto &agentDesc = agent.getAgentDescription();
    const std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::lowerBound() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW InvalidVarType("Wrong variable type passed to HostAgentInstance::lowerBound(). "
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
        gpuErrchk(cub::DeviceReduce::Min(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount)));
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<InT>();
    gpuErrchk(cub::DeviceReduce::Min(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount)));
    gpuErrchkLaunch();
    InT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT>
InT HostAgentInstance::max(const std::string &variable) const {
    const auto &agentDesc = agent.getAgentDescription();
    const std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::upperBound() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW InvalidVarType("Wrong variable type passed to FLAMEGPU_HOST_AGENT_API::upperBound(). "
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
        gpuErrchk(cub::DeviceReduce::Max(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount)));
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<InT>();
    gpuErrchk(cub::DeviceReduce::Max(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount)));
    gpuErrchkLaunch();
    InT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT>
unsigned int HostAgentInstance::count(const std::string &variable, const InT &value) {
    const auto &agentDesc = agent.getAgentDescription();
    const std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::count() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
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
std::vector<unsigned int> HostAgentInstance::histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const {
    return histogramEven<InT, unsigned int>(variable, histogramBins, lowerBound, upperBound);
}
template<typename InT, typename OutT>
std::vector<OutT> HostAgentInstance::histogramEven(const std::string &variable, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) const {
    if (lowerBound >= upperBound) {
        THROW InvalidArgument("lowerBound (%s) must be lower than < upperBound (%s) in HostAgentInstance::histogramEven().",
            std::to_string(lowerBound).c_str(), std::to_string(upperBound).c_str());
    }
    const auto &agentDesc = agent.getAgentDescription();
    const std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::histogramEven() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW InvalidVarType("Wrong variable type passed to HostAgentInstance::histogramEven(). "
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
        gpuErrchk(cub::DeviceHistogram::HistogramEven(nullptr, tempByte,
            reinterpret_cast<InT*>(var_ptr), reinterpret_cast<int*>(api.d_output_space), histogramBins + 1, lowerBound, upperBound, static_cast<int>(agentCount)));
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<OutT>(histogramBins);
    gpuErrchk(cub::DeviceHistogram::HistogramEven(api.d_cub_temp, api.d_cub_temp_size,
        reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), histogramBins + 1, lowerBound, upperBound, static_cast<int>(agentCount)));
    gpuErrchkLaunch();
    std::vector<OutT> rtn(histogramBins);
    gpuErrchk(cudaMemcpy(rtn.data(), api.d_output_space, histogramBins * sizeof(OutT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT, typename reductionOperatorT>
InT HostAgentInstance::reduce(const std::string &variable, reductionOperatorT /*reductionOperator*/, const InT &init) const {
    const auto &agentDesc = agent.getAgentDescription();
    const std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::reduce() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW InvalidVarType("Wrong variable type passed to HostAgentInstance::reduce(). "
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
        gpuErrchk(cub::DeviceReduce::Reduce(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space),
            static_cast<int>(agentCount), typename reductionOperatorT::template binary_function<InT>(), init));
        gpuErrchkLaunch();
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // Resize output storage
    api.resizeOutputSpace<InT>();
    gpuErrchk(cub::DeviceReduce::Reduce(api.d_cub_temp, api.d_cub_temp_size, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space),
        static_cast<int>(agentCount), typename reductionOperatorT::template binary_function<InT>(), init));
    gpuErrchkLaunch();
    InT rtn;
    gpuErrchk(cudaMemcpy(&rtn, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost));
    return rtn;
}
template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
OutT HostAgentInstance::transformReduce(const std::string &variable, transformOperatorT /*transformOperator*/, reductionOperatorT /*reductionOperator*/, const OutT &init) const {
    const auto &agentDesc = agent.getAgentDescription();
    const std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::transformReduce() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW InvalidVarType("Wrong variable type passed to HostAgentInstance::transformReduce(). "
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


template<typename VarT>
void HostAgentInstance::sort(const std::string &variable, Order order, int beginBit, int endBit) {
    const unsigned int streamId = 0;
    auto &scatter = api.agentModel.singletons->scatter;
    auto &scan = scatter.Scan();
    // Check variable is valid
    const auto &agentDesc = agent.getAgentDescription();
    const std::type_index typ = agentDesc.description->getVariableType(variable);  // This will throw name exception
    if (agentDesc.variables.at(variable).elements != 1) {
        THROW UnsupportedVarType("HostAgentInstance::sort() does not support agent array variables.");
    }
    if (std::type_index(typeid(VarT)) != typ) {
        THROW InvalidVarType("Wrong variable type passed to HostAgentInstance::sort(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.variables.at(variable).type.name(), typeid(VarT).name());
    }
    // We will use scan_flag agent_death/message_output here so resize
    const unsigned int agentCount = agent.getStateSize(stateName);
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const size_t total_variable_buffer_size = sizeof(VarT) * agentCount;
    const unsigned int fake_num_agent = static_cast<unsigned int>(total_variable_buffer_size/sizeof(unsigned int)) +1;
    scan.resize(fake_num_agent, CUDAScanCompaction::AGENT_DEATH, streamId);
    scan.resize(agentCount, CUDAScanCompaction::MESSAGE_OUTPUT, streamId);
    VarT *keys_in = reinterpret_cast<VarT *>(scan.Config(CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.scan_flag);
    VarT *keys_out = reinterpret_cast<VarT *>(scan.Config(CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.position);
    unsigned int *vals_in = scan.Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.scan_flag;
    unsigned int *vals_out = scan.Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.position;
    // Create array of TID (use scanflag_death.position)
    fillTIDArray(vals_in, agentCount, 0);  // @todo - use a non default stream
    // Create array of agent values (use scanflag_death.scan_flag)
    gpuErrchk(cudaMemcpy(keys_in, var_ptr, total_variable_buffer_size, cudaMemcpyDeviceToDevice));
    // Check if we need to resize cub storage
    const FLAMEGPU_HOST_API::CUB_Config cc = { FLAMEGPU_HOST_API::SORT, typeid(VarT).hash_code() };
    if (api.tempStorageRequiresResize(cc, agentCount)) {
        // Resize cub storage
        size_t tempByte = 0;
        if (order == Asc) {
            gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, tempByte, keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit));
        } else {
            gpuErrchk(cub::DeviceRadixSort::SortPairsDescending(nullptr, tempByte, keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit));
        }
        api.resizeTempStorage(cc, agentCount, tempByte);
    }
    // pair sort
    if (order == Asc) {
        gpuErrchk(cub::DeviceRadixSort::SortPairs(api.d_cub_temp, api.d_cub_temp_size, keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit));
    } else {
        gpuErrchk(cub::DeviceRadixSort::SortPairsDescending(api.d_cub_temp, api.d_cub_temp_size, keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit));
    }
    // Scatter all agent variables
    api.agentModel.agent_map.at(agentDesc.name)->scatterSort(stateName, scatter, streamId, 0);  // @todo use a per simulation stream?
}


template<typename Var1T, typename Var2T>
void HostAgentInstance::sort(const std::string &variable1, Order order1, const std::string &variable2, Order order2) {
    const unsigned int streamId = 0;
    auto &scatter = api.agentModel.singletons->scatter;
    auto &scan = scatter.Scan();
    const auto &agentDesc = agent.getAgentDescription();
    {  // Check variable 1 is valid
        const std::type_index typ = agentDesc.description->getVariableType(variable1);  // This will throw name exception
        if (agentDesc.variables.at(variable1).elements != 1) {
            THROW UnsupportedVarType("HostAgentInstance::sort() does not support agent array variables.");
        }
        if (std::type_index(typeid(Var1T)) != typ) {
            THROW InvalidVarType("Wrong type for variable '%s' passed to HostAgentInstance::sort(). "
                "This call expects '%s', but '%s' was requested.",
                variable1.c_str(), agentDesc.variables.at(variable1).type.name(), typeid(Var1T).name());
        }
    }
    {  // Check variable 2 is valid
        const std::type_index typ = agentDesc.description->getVariableType(variable2);  // This will throw name exception
        if (agentDesc.variables.at(variable2).elements != 1) {
            THROW UnsupportedVarType("HostAgentInstance::sort() does not support agent array variables.");
        }
        if (std::type_index(typeid(Var2T)) != typ) {
            THROW InvalidVarType("Wrong type for variable '%s' passed to HostAgentInstance::sort(). "
                "This call expects '%s', but '%s' was requested.",
                variable2.c_str(), agentDesc.variables.at(variable2).type.name(), typeid(Var2T).name());
        }
    }
    const unsigned int agentCount = agent.getStateSize(stateName);
    // Fill array with var1 keys
    {
        // Resize
        const size_t total_variable_buffer_size = sizeof(Var1T) * agentCount;
        const unsigned int fake_num_agent = static_cast<unsigned int>(total_variable_buffer_size/sizeof(unsigned int)) +1;
        scan.resize(fake_num_agent, CUDAScanCompaction::AGENT_DEATH, streamId);
        // Fill
        void *keys1b = scan.Config(CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.position;
        void *var_ptr = agent.getStateVariablePtr(stateName, variable1);
        gpuErrchk(cudaMemcpy(keys1b, var_ptr, total_variable_buffer_size, cudaMemcpyDeviceToDevice));
    }
    // Fill array with var2 keys
    {
        // Resize
        const size_t total_variable_buffer_size = sizeof(Var2T) * agentCount;
        const unsigned int fake_num_agent = static_cast<unsigned int>(total_variable_buffer_size/sizeof(unsigned int)) +1;
        scan.resize(std::max(agentCount, fake_num_agent), CUDAScanCompaction::MESSAGE_OUTPUT, streamId);
        // Fill
        void *keys2 = scan.Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.scan_flag;
        void *var_ptr = agent.getStateVariablePtr(stateName, variable2);
        gpuErrchk(cudaMemcpy(keys2, var_ptr, total_variable_buffer_size, cudaMemcpyDeviceToDevice));
    }
    // Define our buffers (here, after resize)
    Var1T *keys1 = reinterpret_cast<Var1T *>(scan.Config(CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.scan_flag);
    Var1T *keys1b = reinterpret_cast<Var1T *>(scan.Config(CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.position);
    Var2T *keys2 = reinterpret_cast<Var2T *>(scan.Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.scan_flag);
    unsigned int *vals = scan.Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.position;
    // Init value array
    fillTIDArray(vals, agentCount, 0);  // @todo - use a non default stream
    // Process variable 2 first
    {
        // pair sort values
        if (order2 == Asc) {
            thrust::stable_sort_by_key(thrust::device_ptr<Var2T>(keys2), thrust::device_ptr<Var2T>(keys2 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::less<Var2T>());
        } else {
            thrust::stable_sort_by_key(thrust::device_ptr<Var2T>(keys2), thrust::device_ptr<Var2T>(keys2 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::greater<Var2T>());
        }
        gpuErrchkLaunch();
        // sort keys1 based on this order
        sortBuffer(keys1, keys1b, vals, sizeof(Var1T), agentCount, 0);  // @todo use a non default stream
    }
    // Process variable 1 second
    {
        // pair sort
        if (order1 == Asc) {
            thrust::stable_sort_by_key(thrust::device_ptr<Var1T>(keys1), thrust::device_ptr<Var1T>(keys1 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::less<Var1T>());
        } else {
            thrust::stable_sort_by_key(thrust::device_ptr<Var1T>(keys1), thrust::device_ptr<Var1T>(keys1 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::greater<Var1T>());
        }
        gpuErrchkLaunch();
    }
    // Scatter all agent variables
    api.agentModel.agent_map.at(agentDesc.name)->scatterSort(stateName, scatter, streamId, 0);  // @todo - use simulation specific stream.
}
#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_AGENT_API_H_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENT_HOSTAGENTAPI_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENT_HOSTAGENTAPI_CUH_
#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#endif  // _MSC_VER
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 1719
#else
#pragma diag_suppress 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_default 1719
#else
#pragma diag_default 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <utility>

#include "flamegpu/simulation/detail/AgentInterface.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/HostAPI.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/simulation/detail/CUDAAgent.h"
#include "flamegpu/runtime/agent/DeviceAgentVector.h"
#include "flamegpu/runtime/agent/DeviceAgentVector_impl.h"
#include "flamegpu/simulation/AgentLoggingConfig_Reductions.cuh"
#include "flamegpu/simulation/AgentLoggingConfig_SumReturn.h"
#include "flamegpu/detail/type_decode.h"

namespace flamegpu {

/**
 * Macro for defining custom reduction functions with the correct inputs.
 *
 * (a, b)->(c) 
 *
 * These functions must be valid CUDA code, and have no access to the FLAMEGPU DeviceAPI.
 *
 * Saves users from manually defining custom reductions, e.g.:
 * @code{.cpp}
 * // User Implemented custom reduction
 * struct SomeCustomReduction_impl {
 *  public:
 *     template <typename OutT>
 *     struct binary_function {
 *         __device__ __forceinline__ OutT operator()(const OutT &a, const OutT &b) const {
 *              // reduce something
                return a + b;
 *         }
 *     };
 * };
 * SomeCustomReduction_impl SomeCustomReduction;
 * @endcode
 */
#define FLAMEGPU_CUSTOM_REDUCTION(funcName, a, b)\
struct funcName ## _impl {\
 public:\
    template <typename OutT>\
    struct binary_function {\
        __host__ __device__ __forceinline__ OutT operator()(const OutT &a, const OutT &b) const;\
    };\
};\
funcName ## _impl funcName;\
template <typename OutT>\
__host__ __device__ __forceinline__ OutT funcName ## _impl::binary_function<OutT>::operator()(const OutT & a, const OutT & b) const

 /**
  * Macro for defining custom transform functions with the correct inputs.
  *
  * (a)->(b)
  *
  * These functions must be valid CUDA code, and have no access to the FLAMEGPU DeviceAPI.
  *
  * Saves users from manually defining custom transformations, e.g.:
  * @code{.cpp}
  * // User Implemented custom transform
  * struct SomeCustomTransform_impl {
  *  public:
  *     template<typename InT, typename OutT>
  *     struct unary_function {
  *         __device__ __forceinline__ OutT operator()(const InT &a) const {
  *              // transform something
                 return a * a;
  *         }
  *     };
  * };
  * SomeCustomTransform_impl SomeCustomTransform;
  * @endcode
  */
#define FLAMEGPU_CUSTOM_TRANSFORM(funcName, a)\
struct funcName ## _impl {\
 public:\
    template<typename InT, typename OutT>\
    struct unary_function {\
        __host__ __device__ OutT operator()(const InT &a) const;\
    };\
};\
funcName ## _impl funcName;\
template<typename InT, typename OutT>\
__device__ __forceinline__ OutT funcName ## _impl::unary_function<InT, OutT>::operator()(const InT &a) const

/**
 * Collection of HostAPI functions related to agents
 *
 * Mostly provides access to reductions over agent variables
 */
class HostAgentAPI {
    /**
     * Access to async sort method/s by spatialSortAgent_async()
     */
    friend class CUDASimulation;

 public:
   /**
    * Construct a new HostAgentAPI instance for a specified agent type and state
    *
    * @param _api Parent HostAPI instance
    * @param _agent Agent object holding the agent data
    * @param _stateName Name of the agent state to be represented
    * @param _agentOffsets Layout of memory within the Host Agent Birth data structure (_newAgentData)
    * @param _newAgentData Structure containing agents birthed via Host Agent Birth
    */
    HostAgentAPI(HostAPI &_api, detail::AgentInterface &_agent, const std::string &_stateName, const VarOffsetStruct &_agentOffsets, HostAPI::AgentDataBuffer&_newAgentData)
        : api(_api)
        , agent(_agent)
        , stateName(_stateName)
        , agentOffsets(_agentOffsets)
        , newAgentData(_newAgentData) { }
    /**
     * Copy constructor
     * Not actually sure this is required
     */
    HostAgentAPI(const HostAgentAPI& other)
        : api(other.api)
        , agent(other.agent)
        , stateName(other.stateName)
        , agentOffsets(other.agentOffsets)
        , newAgentData(other.newAgentData)
    { }
    /**
     * Creates a new agent in the current agent and returns an object for configuring it's member variables
     * 
     * This mode of agent creation is more efficient than manipulating the vector returned by getPopulationData(),
     * as it batches agent creation to a single scatter kernel if possible (e.g. no data dependencies).
     */
    HostNewAgentAPI newAgent();
    /*
     * Returns the number of agents in this state
     */
    unsigned int count();
    /**
     * Wraps cub::DeviceReduce::Sum()
     * @param variable The agent variable to perform the sum reduction across
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT>
    InT sum(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Sum()
     * @param variable The agent variable to perform the sum reduction across
     * @tparam OutT The template arg, 'OutT' can be used if the sum is expected to exceed the representation of the type being summed
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT, typename OutT>
    OutT sum(const std::string &variable) const;
    /**
     * Returns the mean and standard deviation of the specified variable in the agent population
     * The return value is a pair, where the first item holds the mean and the second item the standard deviation.
     * @param variable The agent variable to perform the sum reduction across
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note If you only require the mean, it is more efficient to use sum()/count()
     */
    // Suppress GCC >= 10.1 diagnostic due to ABI change in C++17 mode on aarch64 for parameter passing of std::pair<double, double>
#if defined(__GNUC__) && (( __GNUC__ == 10 && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 1) || __GNUC__ > 10)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpsabi"
#endif
    template<typename InT>
    std::pair<double, double> meanStandardDeviation(const std::string& variable) const;
    /**
     * Wraps cub::DeviceReduce::Min()
     * @param variable The agent variable to perform the lowerBound reduction across
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
#if defined(__GNUC__) && (( __GNUC__ == 10 && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 1) || __GNUC__ > 10)
    #pragma GCC diagnostic pop
#endif
    template<typename InT>
    InT min(const std::string &variable) const;
    /**
     * Wraps cub::DeviceReduce::Max()
     * @param variable The agent variable to perform the upperBound reduction across
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT>
    InT max(const std::string &variable) const;
    /**
     * Wraps thrust::count(), to count the number of occurences of the provided value
     * @param variable The agent variable to perform the count reduction across
     * @param value The value to count occurences of
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT>
    unsigned int count(const std::string &variable, InT value) const;
    /**
     * Wraps cub::DeviceHistogram::HistogramEven()
     * @param variable The agent variable to perform the reduction across
     * @param histogramBins The number of bins the histogram should have
     * @param lowerBound The (inclusive) lower sample value boundary of lowest bin
     * @param upperBound The (exclusive) upper sample value boundary of upper bin
     * @note 2nd template arg can be used if calculation requires higher bit type to avoid overflow
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT>
    std::vector<unsigned int> histogramEven(const std::string &variable, unsigned int histogramBins, InT lowerBound, InT upperBound) const;
    template<typename InT, typename OutT>
    std::vector<OutT> histogramEven(const std::string &variable, unsigned int histogramBins, InT lowerBound, InT upperBound) const;
    /**
     * Wraps cub::DeviceReduce::Reduce(), to perform a reduction with a custom operator
     * @param variable The agent variable to perform the reduction across
     * @param reductionOperator The custom reduction function
     * @param init Initial value of the reduction
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT, typename reductionOperatorT>
    InT reduce(const std::string &variable, reductionOperatorT reductionOperator, InT init) const;
    /**
     * Wraps thrust::transformReduce(), to perform a custom transform on values before performing a custom reduction
     * @param variable The agent variable to perform the reduction across
     * @param transformOperator The custom unary transform function
     * @param reductionOperator The custom binary reduction function
     * @param init Initial value of the reduction
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
    OutT transformReduce(const std::string &variable, transformOperatorT transformOperator, reductionOperatorT reductionOperator, OutT init) const;
    /**
     * Sort ordering
     * Ascending or Descending
     */
    enum Order {Asc, Desc};
    /**
     * Sorts agents according to the named variable
     * @param variable The agent variable to sort the agents according to
     * @param order Whether the agents should be sorted in ascending or descending order of the variable
     * @param beginBit Advanced Option, see note
     * @param endBit Advanced Option, see note
     * @tparam VarT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
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
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @tparam Var1T The type of variable1 as specified in the model description hierarchy
     * @tparam Var2T The type of variable2 as specified in the model description hierarchy
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     */
    template<typename Var1T, typename Var2T>
    void sort(const std::string &variable1, Order order1, const std::string &variable2, Order order2);
    /**
     * Downloads the current agent state from device into an AgentVector which is returned
     *
     * This function is considered expensive, as it triggers a high number of host-device memory transfers.
     * It should be used as a last resort
     */
    DeviceAgentVector getPopulationData();

 private:
    /**
     * Fills the provided device buffer with consecutive integers
     * @param d_buffer Device pointer to buffer to be filled
     * @param length Length of the buffer (how many unsigned ints can it hold)
     * @param stream CUDA stream to be used for async CUDA operations
     */
    static void fillTIDArray_async(unsigned int *d_buffer, unsigned int length, cudaStream_t stream);
    /**
     * Sorts a buffer by the positions array, used for multi variable agent sorts
     * @param dest Device pointer to buffer for sorted data to be placed
     * @param src Device pointer to buffer to be sorted
     * @param position Positions buffer
     * @param typeLen sizeof the type stored in the buffer (e.g. sizeof(int))
     * @param length Length of the buffer (how many items it can it hold)
     * @param stream CUDA stream to be used for async CUDA operations
     */
    static void sortBuffer_async(void *dest, void*src, unsigned int *position, size_t typeLen, unsigned int length, cudaStream_t stream);
    /**
     * Wraps cub::DeviceReduce::Sum()
     * @param variable The agent variable to perform the sum reduction across
     * @param result Variable which will store the result (note method is async, result may not arrive until stream is synchronised)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @tparam OutT The template arg, 'OutT' can be used if the sum is expected to exceed the representation of the type being summed
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Method is async, result may not arrive until stream is synchronised
     */
    template<typename InT, typename OutT>
    void sum_async(const std::string& variable, OutT& result, cudaStream_t stream, unsigned int streamId) const;
    /**
     * Returns the mean and standard deviation of the specified variable in the agent population
     * The return value is a pair, where the first item holds the mean and the second item the standard deviation.
     * @param variable The agent variable to perform the sum reduction across
     * @param result Variable which will store the result
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note If you only require the mean, it is more efficient to use sum()/count()
     * @note Not actually async, would need a big rewrite (and to stop using the shared device symbol?)
     */
    template<typename InT>
    void meanStandardDeviation_async(const std::string& variable, std::pair<double, double>& result, cudaStream_t stream, unsigned int streamId) const;
    /**
     * Wraps cub::DeviceReduce::Min()
     * @param variable The agent variable to perform the lowerBound reduction across
     * @param result Variable which will store the result (note method is async, result may not arrive until stream is synchronised)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Method is async, result may not arrive until stream is synchronised
     */
    template<typename InT>
    void min_async(const std::string& variable, InT& result, cudaStream_t stream, unsigned int streamId) const;
    /**
     * Wraps cub::DeviceReduce::Max()
     * @param variable The agent variable to perform the upperBound reduction across
     * @param result Variable which will store the result (note method is async, result may not arrive until stream is synchronised)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Method is async, result may not arrive until stream is synchronised
     */
    template<typename InT>
    void max_async(const std::string& variable, InT& result, cudaStream_t stream, unsigned int streamId) const;
    /**
     * Wraps thrust::count(), to count the number of occurences of the provided value
     * @param variable The agent variable to perform the count reduction across
     * @param value The value to count occurrences of
     * @param stream The CUDAStream to use for CUDA operations
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Not actually async, uses thrust method that doesn't support async, uses specified stream though
     */
    template<typename InT>
    unsigned int count_async(const std::string& variable, InT value, cudaStream_t stream) const;
    /**
     * Wraps cub::DeviceHistogram::HistogramEven()
     * @param variable The agent variable to perform the reduction across
     * @param histogramBins The number of bins the histogram should have
     * @param lowerBound The (inclusive) lower sample value boundary of lowest bin
     * @param upperBound The (exclusive) upper sample value boundary of upper bin
     * @param result Variable which will store the result (note method is async, result may not arrive until stream is synchronised)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @tparam OutT The type of the histogram bin variables
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Method is async, result may not arrive until stream is synchronised
     */
    template<typename InT, typename OutT>
    void histogramEven_async(const std::string& variable, unsigned int histogramBins, InT lowerBound, InT upperBound, std::vector<OutT> &result, cudaStream_t stream, unsigned int streamId) const;
    /**
     * Wraps cub::DeviceReduce::Reduce(), to perform a reduction with a custom operator
     * @param variable The agent variable to perform the reduction across
     * @param reductionOperator The custom reduction function
     * @param init Initial value of the reduction
     * @param result Variable which will store the result (note method is async, result may not arrive until stream is synchronised)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Method is async, result may not arrive until stream is synchronised
     */
    template<typename InT, typename reductionOperatorT>
    void reduce_async(const std::string& variable, reductionOperatorT reductionOperator, InT init, InT &result, cudaStream_t stream, unsigned int streamId) const;
    /**
     * Wraps thrust::transformReduce(), to perform a custom transform on values before performing a custom reduction
     * @param variable The agent variable to perform the reduction across
     * @param transformOperator The custom unary transform function
     * @param reductionOperator The custom binary reduction function
     * @param init Initial value of the reduction
     * @tparam InT The type of the variable as specified in the model description hierarchy
     * @param stream The CUDAStream to use for CUDA operations
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Not actually async, uses thrust method that doesn't support async, uses specified stream though
     */
    template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
    OutT transformReduce_async(const std::string& variable, transformOperatorT transformOperator, reductionOperatorT reductionOperator, OutT init, cudaStream_t stream) const;
    /**
     * Sorts agents according to the named variable
     * @param variable The agent variable to sort the agents according to
     * @param order Whether the agents should be sorted in ascending or descending order of the variable
     * @param beginBit Advanced Option, see note
     * @param endBit Advanced Option, see note
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId The index of the stream resources to use
     * @tparam VarT The type of the variable as specified in the model description hierarchy
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note An optional bit subrange [begin_bit, end_bit) of differentiating variable bits can be specified. This can reduce overall sorting overhead and yield a corresponding performance improvement.
     * @note The sort provides no guarantee of stability
     */
    template<typename VarT>
    void sort_async(const std::string& variable, Order order, int beginBit, int endBit, cudaStream_t stream, unsigned int streamId);
    /**
     * Sort agents according to two variables e.g. [1:c, 3:b, 1:b, 1:a] -> [1:a, 1:b, 1:c, 3:b]
     * @param variable1 This variable will be the main direction that agents are sorted
     * @param order1 The order that variable 1 should be sorted according to
     * @param variable2 Agents with equal variable1's, will be sorted according this this variable
     * @param order2 The order that variable 2 should be sorted according to
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId The index of the stream resources to use
     * @throws exception::UnsupportedVarType Array variables are not supported
     * @tparam Var1T The type of variable1 as specified in the model description hierarchy
     * @tparam Var2T The type of variable2 as specified in the model description hierarchy
     * @throws exception::InvalidAgentVar If the agent does not contain a variable of the same name
     * @throws exception::InvalidVarType If the passed variable type does not match that specified in the model description hierarchy
     * @note Not actually async, uses thrust method that doesn't support async, uses specified stream though
     */
    template<typename Var1T, typename Var2T>
    void sort_async(const std::string& variable1, Order order1, const std::string& variable2, Order order2, cudaStream_t stream, unsigned int streamId);
    /**
     * Parent HostAPI
     */
    HostAPI &api;
    /**
     * Main object containing agent data
     * Probably type CUDAAgent
     */
    detail::AgentInterface &agent;
    /**
     * Agent state being accessed
     */
    const std::string stateName;
    /**
     * Holds offsets for accessing newAgentData
     * @see newAgent()
     */
    const VarOffsetStruct& agentOffsets;
    /**
     * Compact data store for efficient host agent creation
     * @see newAgent()
     */
    HostAPI::AgentDataBuffer& newAgentData;
};

//
// Implementation
//

template<typename InT>
InT HostAgentAPI::sum(const std::string &variable) const {
    InT rtn;
    sum_async<InT, InT>(variable, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
    return rtn;
}
template<typename InT, typename OutT>
OutT HostAgentAPI::sum(const std::string& variable) const {
    OutT rtn;
    sum_async<InT, OutT>(variable, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
    return rtn;
}
template<typename InT, typename OutT>
void HostAgentAPI::sum_async(const std::string &variable, OutT &result, const cudaStream_t stream, const unsigned int streamId) const {
    static_assert(sizeof(InT) <= sizeof(OutT), "Template arg OutT should not be of a smaller size than InT");
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != 1) {
        THROW exception::UnsupportedVarType("HostAgentAPI::sum() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::sum(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.getVariableType(variable).name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Check if we need to resize cub storage
    auto &cub_temp = api.scatter.CubTemp(streamId);
    size_t tempByte = 0;
    gpuErrchk(cub::DeviceReduce::Sum(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), static_cast<int>(agentCount), stream));
    cub_temp.resize(tempByte);
    // Resize output storage
    api.resizeOutputSpace<OutT>();
    gpuErrchk(cub::DeviceReduce::Sum(cub_temp.getPtr(), cub_temp.getSize(), reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), static_cast<int>(agentCount), stream));
    gpuErrchkLaunch();
    gpuErrchk(cudaMemcpyAsync(&result, api.d_output_space, sizeof(OutT), cudaMemcpyDeviceToHost, stream));
}
// Suppress GCC >= 10.1 diagnostic due to ABI change in C++17 mode on aarch64 for parameter passing of std::pair<double, double>
#if defined(__GNUC__) && (( __GNUC__ == 10 && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 1) || __GNUC__ > 10)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpsabi"
#endif
template<typename InT>
std::pair<double, double> HostAgentAPI::meanStandardDeviation(const std::string& variable) const {
    std::pair<double, double> rtn;
    meanStandardDeviation_async<InT>(variable, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));  // Redundant, meanStandardDeviation_async() is not truly async
    return rtn;
}
#if defined(__GNUC__) && (( __GNUC__ == 10 && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 1) || __GNUC__ > 10)
    #pragma GCC diagnostic pop
#endif
template<typename InT>
void HostAgentAPI::meanStandardDeviation_async(const std::string& variable, std::pair<double, double> &result, const cudaStream_t stream, const unsigned int streamId) const {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != 1) {
        THROW exception::UnsupportedVarType("HostAgentAPI::meanStandardDeviation() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::meanStandardDeviation(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.getVariableType(variable).name(), typeid(InT).name());
    }
    const auto agentCount = agent.getStateSize(stateName);
    if (agentCount == 0) {
        result = std::make_pair(0.0, 0.0);
    }
    // Calculate mean (We could make this more efficient by leaving sum in device mem?)
    typename sum_input_t<InT>::result_t sum_result;
    sum_async<InT, typename sum_input_t<InT>::result_t>(variable, sum_result, stream, streamId);
    gpuErrchk(cudaStreamSynchronize(stream));
    const double mean = sum_result / static_cast<double>(agentCount);
    // Then for each number: subtract the Mean and square the result
    // Then work out the mean of those squared differences.
    auto lock = std::unique_lock<std::mutex>(detail::STANDARD_DEVIATION_MEAN_mutex);
    gpuErrchk(cudaMemcpyToSymbolAsync(detail::STANDARD_DEVIATION_MEAN, &mean, sizeof(double), 0, cudaMemcpyHostToDevice, stream));
    const double variance = transformReduce_async<InT, double>(variable, detail::standard_deviation_subtract_mean, detail::standard_deviation_add, 0, stream) / static_cast<double>(agentCount);
    lock.unlock();
    // Take the square root of that and we are done!
    result = std::make_pair(mean, sqrt(variance));
}
template<typename InT>
InT HostAgentAPI::min(const std::string& variable) const {
    InT rtn;
    min_async<InT>(variable, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
    return rtn;
}
template<typename InT>
void HostAgentAPI::min_async(const std::string &variable, InT& result, const cudaStream_t stream, const unsigned int streamId) const {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != 1) {
        THROW exception::UnsupportedVarType("HostAgentAPI::lowerBound() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::min(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.getVariableType(variable).name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Check if we need to resize cub storage
    auto& cub_temp = api.scatter.CubTemp(streamId);
    // Resize cub storage
    size_t tempByte = 0;
    gpuErrchk(cub::DeviceReduce::Min(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount), stream));
    gpuErrchkLaunch();
    cub_temp.resize(tempByte);
    // Resize output storage
    api.resizeOutputSpace<InT>();
    gpuErrchk(cub::DeviceReduce::Min(cub_temp.getPtr(), cub_temp.getSize(), reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount), stream));
    gpuErrchkLaunch();
    gpuErrchk(cudaMemcpyAsync(&result, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost, stream));
}
template<typename InT>
InT HostAgentAPI::max(const std::string& variable) const {
    InT rtn;
    max_async<InT>(variable, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
    return rtn;
}
template<typename InT>
void HostAgentAPI::max_async(const std::string &variable, InT &result, const cudaStream_t stream, const unsigned int streamId) const {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != 1) {
        THROW exception::UnsupportedVarType("HostAgentAPI::max() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::max(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.getVariableType(variable).name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Check if we need to resize cub storage
    auto& cub_temp = api.scatter.CubTemp(streamId);
    // Resize cub storage
    size_t tempByte = 0;
    gpuErrchk(cub::DeviceReduce::Max(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount), stream));
    cub_temp.resize(tempByte);
    // Resize output storage
    api.resizeOutputSpace<InT>();
    gpuErrchk(cub::DeviceReduce::Max(cub_temp.getPtr(), cub_temp.getSize(), reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space), static_cast<int>(agentCount), stream));
    gpuErrchkLaunch();
    gpuErrchk(cudaMemcpyAsync(&result, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost, stream));
}
template<typename InT>
unsigned int HostAgentAPI::count(const std::string &variable, InT value) const {
    return count_async<InT>(variable, value, this->api.stream);
}
template<typename InT>
unsigned int HostAgentAPI::count_async(const std::string& variable, InT value, const cudaStream_t stream) const {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != 1) {
        THROW exception::UnsupportedVarType("HostAgentAPI::count() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::count(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.getVariableType(variable).name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Cast return from ptrdiff_t (int64_t) to (uint32_t)
    unsigned int rtn = static_cast<unsigned int>(thrust::count(thrust::cuda::par.on(stream), thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr)), thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr) + agentCount), value));
    gpuErrchkLaunch();
    return rtn;
}
template<typename InT>
std::vector<unsigned int> HostAgentAPI::histogramEven(const std::string &variable, unsigned int histogramBins, InT lowerBound, InT upperBound) const {
    std::vector<unsigned int> rtn;
    histogramEven_async<InT, unsigned int>(variable, histogramBins, lowerBound, upperBound, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
    return rtn;
}
template<typename InT, typename OutT>
std::vector<OutT> HostAgentAPI::histogramEven(const std::string &variable, unsigned int histogramBins, InT lowerBound, InT upperBound) const {
    std::vector<OutT> rtn;
    histogramEven_async<InT, OutT>(variable, histogramBins, lowerBound, upperBound, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
    return rtn;
}
template<typename InT, typename OutT>
void HostAgentAPI::histogramEven_async(const std::string &variable, unsigned int histogramBins, InT lowerBound, InT upperBound, std::vector<OutT>& result, const cudaStream_t stream, const unsigned int streamId) const {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    if (lowerBound >= upperBound) {
        THROW exception::InvalidArgument("lowerBound (%s) must be lower than < upperBound (%s) in HostAgentAPI::histogramEven().",
            std::to_string(lowerBound).c_str(), std::to_string(upperBound).c_str());
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != 1) {
        THROW exception::UnsupportedVarType("HostAgentAPI::histogramEven() does not support agent array variables.");
    }
    if (std::type_index(typeid(InT)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::histogramEven(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.getVariableType(variable).name(), typeid(InT).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Check if we need to resize cub storage
    auto& cub_temp = api.scatter.CubTemp(streamId);
    // Resize cub storage
    size_t tempByte = 0;
    gpuErrchk(cub::DeviceHistogram::HistogramEven(nullptr, tempByte,
        reinterpret_cast<InT*>(var_ptr), reinterpret_cast<int*>(api.d_output_space), histogramBins + 1, lowerBound, upperBound, static_cast<int>(agentCount), stream));
    gpuErrchkLaunch();
    cub_temp.resize(tempByte);
    // Resize output storage
    api.resizeOutputSpace<OutT>(histogramBins);
    gpuErrchk(cub::DeviceHistogram::HistogramEven(cub_temp.getPtr(), cub_temp.getSize(),
        reinterpret_cast<InT*>(var_ptr), reinterpret_cast<OutT*>(api.d_output_space), histogramBins + 1, lowerBound, upperBound, static_cast<int>(agentCount), stream));
    gpuErrchkLaunch();
    result.resize(histogramBins);
    gpuErrchk(cudaMemcpyAsync(result.data(), api.d_output_space, histogramBins * sizeof(OutT), cudaMemcpyDeviceToHost, stream));
}
template<typename InT, typename reductionOperatorT>
InT HostAgentAPI::reduce(const std::string &variable, reductionOperatorT reductionOperator, InT init) const {
    InT rtn;
    reduce_async<InT, reductionOperatorT>(variable, reductionOperator, init, rtn, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
    return rtn;
}
template<typename InT, typename reductionOperatorT>
void HostAgentAPI::reduce_async(const std::string & variable, reductionOperatorT /*reductionOperator*/, InT init, InT &result, const cudaStream_t stream, const unsigned int streamId) const {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != detail::type_decode<InT>::len_t) {
        THROW exception::UnsupportedVarType("HostAgentAPI::reduce() does not support agent array variables.");
    }
    if (std::type_index(typeid(typename detail::type_decode<InT>::type_t)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::reduce(). "
            "This call expects '%s', but '%s' was requested.",
            typ.name(), typeid(typename detail::type_decode<InT>::type_t).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    // Check if we need to resize cub storage
    auto& cub_temp = api.scatter.CubTemp(streamId);
    // Resize cub storage
    size_t tempByte = 0;
    gpuErrchk(cub::DeviceReduce::Reduce(nullptr, tempByte, reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space),
        static_cast<int>(agentCount), typename reductionOperatorT::template binary_function<InT>(), init, stream));
    gpuErrchkLaunch();
    cub_temp.resize(tempByte);
    // Resize output storage
    api.resizeOutputSpace<InT>();
    gpuErrchk(cub::DeviceReduce::Reduce(cub_temp.getPtr(), cub_temp.getSize(), reinterpret_cast<InT*>(var_ptr), reinterpret_cast<InT*>(api.d_output_space),
        static_cast<int>(agentCount), typename reductionOperatorT::template binary_function<InT>(), init, stream));
    gpuErrchkLaunch();
    gpuErrchk(cudaMemcpyAsync(&result, api.d_output_space, sizeof(InT), cudaMemcpyDeviceToHost, stream));
}
template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
OutT HostAgentAPI::transformReduce(const std::string &variable, transformOperatorT transformOperator, reductionOperatorT reductionOperator, OutT init) const {
    return transformReduce_async<InT, OutT, transformOperatorT, reductionOperatorT>(variable, transformOperator, reductionOperator, init, this->api.stream);
}
template<typename InT, typename OutT, typename transformOperatorT, typename reductionOperatorT>
OutT HostAgentAPI::transformReduce_async(const std::string &variable, transformOperatorT /*transformOperator*/, reductionOperatorT /*reductionOperator*/, OutT init, cudaStream_t stream) const {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    const CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != detail::type_decode<InT>::len_t) {
        THROW exception::UnsupportedVarType("HostAgentAPI::transformReduce() does not support agent array variables.");
    }
    if (std::type_index(typeid(typename detail::type_decode<InT>::type_t)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::transformReduce(). "
            "This call expects '%s', but '%s' was requested.",
            typ.name(), typeid(typename detail::type_decode<InT>::type_t).name());
    }
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const auto agentCount = agent.getStateSize(stateName);
    OutT rtn = thrust::transform_reduce(thrust::cuda::par.on(stream), thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr)), thrust::device_ptr<InT>(reinterpret_cast<InT*>(var_ptr) + agentCount),
        typename transformOperatorT::template unary_function<InT, OutT>(), init, typename reductionOperatorT::template binary_function<OutT>());
    gpuErrchkLaunch();
    return rtn;
}


template<typename VarT>
void HostAgentAPI::sort(const std::string &variable, Order order, int beginBit, int endBit) {
    sort_async<VarT>(variable, order, beginBit, endBit, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
}
template<typename VarT>
void HostAgentAPI::sort_async(const std::string & variable, Order order, int beginBit, int endBit, const cudaStream_t stream, const unsigned int streamId) {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream || this->api.streamId != streamId) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    auto &scatter = api.agentModel.singletons->scatter;
    auto &scan = scatter.Scan();
    // Check variable is valid
    CAgentDescription agentDesc(agent.getAgentDescription());
    std::type_index typ = agentDesc.getVariableType(variable);  // This will throw name exception
    if (agentDesc.getVariableLength(variable) != 1) {
        THROW exception::UnsupportedVarType("HostAgentAPI::sort() does not support agent array variables.");
    }
    if (std::type_index(typeid(VarT)) != typ) {
        THROW exception::InvalidVarType("Wrong variable type passed to HostAgentAPI::sort(). "
            "This call expects '%s', but '%s' was requested.",
            agentDesc.getVariableType(variable).name(), typeid(VarT).name());
    }
    // We will use scan_flag agent_death/message_output here so resize
    const unsigned int agentCount = agent.getStateSize(stateName);
    void *var_ptr = agent.getStateVariablePtr(stateName, variable);
    const size_t total_variable_buffer_size = sizeof(VarT) * agentCount;
    const unsigned int fake_num_agent = static_cast<unsigned int>(total_variable_buffer_size/sizeof(unsigned int)) +1;
    scan.resize(fake_num_agent, detail::CUDAScanCompaction::AGENT_DEATH, streamId);
    scan.resize(agentCount, detail::CUDAScanCompaction::MESSAGE_OUTPUT, streamId);
    VarT *keys_in = reinterpret_cast<VarT *>(scan.Config(detail::CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.scan_flag);
    VarT *keys_out = reinterpret_cast<VarT *>(scan.Config(detail::CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.position);
    unsigned int *vals_in = scan.Config(detail::CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.scan_flag;
    unsigned int *vals_out = scan.Config(detail::CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.position;
    // Create array of TID (use scanflag_death.position)
    fillTIDArray_async(vals_in, agentCount, stream);
    // Create array of agent values (use scanflag_death.scan_flag)
    gpuErrchk(cudaMemcpyAsync(keys_in, var_ptr, total_variable_buffer_size, cudaMemcpyDeviceToDevice, stream));
    // Check if we need to resize cub storage
    auto& cub_temp = api.scatter.CubTemp(streamId);
    // Resize cub storage
    size_t tempByte = 0;
    if (order == Asc) {
        gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, tempByte, keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit, stream));
    } else {
        gpuErrchk(cub::DeviceRadixSort::SortPairsDescending(nullptr, tempByte, keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit, stream));
    }
    cub_temp.resize(tempByte);
    // pair sort
    if (order == Asc) {
        gpuErrchk(cub::DeviceRadixSort::SortPairs(cub_temp.getPtr(), cub_temp.getSize(), keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit, stream));
    } else {
        gpuErrchk(cub::DeviceRadixSort::SortPairsDescending(cub_temp.getPtr(), cub_temp.getSize(), keys_in, keys_out, vals_in, vals_out, agentCount, beginBit, endBit, stream));
    }
    // Scatter all agent variables
    api.agentModel.agent_map.at(agentDesc.getName())->scatterSort_async(stateName, scatter, streamId, stream);
    if (population) {
        // If the user has a DeviceAgentVector out, purge cache so it redownloads new data on next use
        population->purgeCache();
    }
}


template<typename Var1T, typename Var2T>
void HostAgentAPI::sort(const std::string &variable1, Order order1, const std::string &variable2, Order order2) {
    sort_async<Var1T, Var2T>(variable1, order1, variable2, order2, this->api.stream, this->api.streamId);
    gpuErrchk(cudaStreamSynchronize(this->api.stream));
}
template<typename Var1T, typename Var2T>
void HostAgentAPI::sort_async(const std::string & variable1, Order order1, const std::string & variable2, Order order2, const cudaStream_t stream, const unsigned int streamId) {
    std::shared_ptr<DeviceAgentVector_impl> population = agent.getPopulationVec(stateName);
    if (population) {
        if (this->api.stream != stream || this->api.streamId != streamId) {
            THROW exception::InvalidOperation("Attempting to sync DeviceAgentVector with wrong stream!\nThis should not be possible.\n");
        }
        // If the user has a DeviceAgentVector out, sync changes
        population->syncChanges();
    }
    auto &scatter = api.agentModel.singletons->scatter;
    auto &scan = scatter.Scan();
    const CAgentDescription agentDesc(agent.getAgentDescription());
    {  // Check variable 1 is valid
        std::type_index typ = agentDesc.getVariableType(variable1);  // This will throw name exception
        if (agentDesc.getVariableLength(variable1) != 1) {
            THROW exception::UnsupportedVarType("HostAgentAPI::sort() does not support agent array variables.");
        }
        if (std::type_index(typeid(Var1T)) != typ) {
            THROW exception::InvalidVarType("Wrong type for variable '%s' passed to HostAgentAPI::sort(). "
                "This call expects '%s', but '%s' was requested.",
                variable1.c_str(), agentDesc.getVariableType(variable1).name(), typeid(Var1T).name());
        }
    }
    {  // Check variable 2 is valid
        std::type_index typ = agentDesc.getVariableType(variable2);  // This will throw name exception
        if (agentDesc.getVariableLength(variable2) != 1) {
            THROW exception::UnsupportedVarType("HostAgentAPI::sort() does not support agent array variables.");
        }
        if (std::type_index(typeid(Var2T)) != typ) {
            THROW exception::InvalidVarType("Wrong type for variable '%s' passed to HostAgentAPI::sort(). "
                "This call expects '%s', but '%s' was requested.",
                variable2.c_str(), agentDesc.getVariableType(variable2).name(), typeid(Var2T).name());
        }
    }
    const unsigned int agentCount = agent.getStateSize(stateName);
    // Fill array with var1 keys
    {
        // Resize
        const size_t total_variable_buffer_size = sizeof(Var1T) * agentCount;
        const unsigned int fake_num_agent = static_cast<unsigned int>(total_variable_buffer_size/sizeof(unsigned int)) +1;
        scan.resize(fake_num_agent, detail::CUDAScanCompaction::AGENT_DEATH, streamId);
        // Fill
        void *keys1b = scan.Config(detail::CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.position;
        void *var_ptr = agent.getStateVariablePtr(stateName, variable1);
        gpuErrchk(cudaMemcpyAsync(keys1b, var_ptr, total_variable_buffer_size, cudaMemcpyDeviceToDevice, stream));
    }
    // Fill array with var2 keys
    {
        // Resize
        const size_t total_variable_buffer_size = sizeof(Var2T) * agentCount;
        const unsigned int fake_num_agent = static_cast<unsigned int>(total_variable_buffer_size/sizeof(unsigned int)) +1;
        scan.resize(std::max(agentCount, fake_num_agent), detail::CUDAScanCompaction::MESSAGE_OUTPUT, streamId);
        // Fill
        void *keys2 = scan.Config(detail::CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.scan_flag;
        void *var_ptr = agent.getStateVariablePtr(stateName, variable2);
        gpuErrchk(cudaMemcpyAsync(keys2, var_ptr, total_variable_buffer_size, cudaMemcpyDeviceToDevice, stream));
    }
    // Define our buffers (here, after resize)
    Var1T *keys1 = reinterpret_cast<Var1T *>(scan.Config(detail::CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.scan_flag);
    Var1T *keys1b = reinterpret_cast<Var1T *>(scan.Config(detail::CUDAScanCompaction::Type::AGENT_DEATH, streamId).d_ptrs.position);
    Var2T *keys2 = reinterpret_cast<Var2T *>(scan.Config(detail::CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.scan_flag);
    unsigned int *vals = scan.Config(detail::CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId).d_ptrs.position;
    // Init value array
    fillTIDArray_async(vals, agentCount, stream);
    // Process variable 2 first
    {
        // pair sort values
        if (order2 == Asc) {
            thrust::stable_sort_by_key(thrust::cuda::par.on(stream), thrust::device_ptr<Var2T>(keys2), thrust::device_ptr<Var2T>(keys2 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::less<Var2T>());
        } else {
            thrust::stable_sort_by_key(thrust::cuda::par.on(stream), thrust::device_ptr<Var2T>(keys2), thrust::device_ptr<Var2T>(keys2 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::greater<Var2T>());
        }
        gpuErrchkLaunch();
        // sort keys1 based on this order
        sortBuffer_async(keys1, keys1b, vals, sizeof(Var1T), agentCount, stream);
    }
    // Process variable 1 second
    {
        // pair sort
        if (order1 == Asc) {
            thrust::stable_sort_by_key(thrust::cuda::par.on(stream), thrust::device_ptr<Var1T>(keys1), thrust::device_ptr<Var1T>(keys1 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::less<Var1T>());
        } else {
            thrust::stable_sort_by_key(thrust::cuda::par.on(stream), thrust::device_ptr<Var1T>(keys1), thrust::device_ptr<Var1T>(keys1 + agentCount),
            thrust::device_ptr<unsigned int>(vals), thrust::greater<Var1T>());
        }
        gpuErrchkLaunch();
    }
    // Scatter all agent variables
    api.agentModel.agent_map.at(agentDesc.getName())->scatterSort_async(stateName, scatter, streamId, stream);

    if (population) {
        // If the user has a DeviceAgentVector out, purge cache so it redownloads new data on next use
        population->purgeCache();
    }
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENT_HOSTAGENTAPI_CUH_

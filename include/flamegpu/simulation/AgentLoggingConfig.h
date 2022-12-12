#ifndef INCLUDE_FLAMEGPU_SIMULATION_AGENTLOGGINGCONFIG_H_
#define INCLUDE_FLAMEGPU_SIMULATION_AGENTLOGGINGCONFIG_H_

#include <string>
#include <memory>
#include <set>
#include <mutex>
#include <utility>

#include "flamegpu/simulation/LoggingConfig.h"
#include "flamegpu/simulation/AgentLoggingConfig_Reductions.cuh"
#include "flamegpu/simulation/AgentLoggingConfig_SumReturn.h"
#include "flamegpu/runtime/agent/HostAgentAPI.cuh"

namespace flamegpu {

struct ModelData;

/**
 * Interface to the data structure for controlling how a specific agent-state's data is logged
 */
class AgentLoggingConfig {
     /**
      * Requires access for calling constructor
      */
    friend AgentLoggingConfig LoggingConfig::agent(const std::string &, const std::string &);
    /**
     * Private constructor, directly initialises the two member variables
     * Used by LoggingConfig::agent(const std::string &, const std::string &)
     */
    AgentLoggingConfig(std::shared_ptr<const AgentData> agent,
                       std::pair<std::shared_ptr<std::set<LoggingConfig::NameReductionFn>>, bool> &agent_set);

 public:
    /**
     * Log the number of agents in this specific agent state
     */
    void logCount() { log_count = true; }
    /**
     * Mark the mean of the named agent variable to be logged
     * @param variable_name Name of the agent variable to have it's mean logged
     * @tparam T The type of the named variable
     * @throws exception::InvalidAgentVar If the agent var was not found inside the specified agent
     * @throws exception::InvalidArgument If the agent var's mean has already been marked for logging
     */
    template<typename T>
    void logMean(const std::string &variable_name);
    /**
     * Mark the standard deviation of the named agent variable to be logged
     * @param variable_name Name of the agent variable to have it's standard deviation logged
     * @tparam T The type of the named variable
     * @throws exception::InvalidAgentVar If the agent var was not found inside the specified agent
     * @throws exception::InvalidArgument If the agent var's standard deviation has already been marked for logging
     */
    template<typename T>
    void logStandardDev(const std::string &variable_name);
    /**
     * Mark the min of the named agent variable to be logged
     * @param variable_name Name of the agent variable to have it's min logged
     * @tparam T The type of the named variable
     * @throws exception::InvalidAgentVar If the agent var was not found inside the specified agent
     * @throws exception::InvalidArgument If the agent var's min has already been marked for logging
     */
    template<typename T>
    void logMin(const std::string &variable_name);
    /**
     * Mark the max of the named agent variable to be logged
     * @param variable_name Name of the agent variable to have it's max logged
     * @tparam T The type of the named variable
     * @throws exception::InvalidAgentVar If the agent var was not found inside the specified agent
     * @throws exception::InvalidArgument If the agent var's max has already been marked for logging
     */
    template<typename T>
    void logMax(const std::string &variable_name);
    /**
     * Mark the sum of the named agent variable to be logged
     * @param variable_name Name of the agent variable to have it's sum logged
     * @tparam T The type of the named variable
     * @throws exception::InvalidAgentVar If the agent var was not found inside the specified agent
     * @throws exception::InvalidArgument If the agent var's sum has already been marked for logging
     */
    template<typename T>
    void logSum(const std::string &variable_name);

 private:
    /**
     * Generic logging method
     * Returns false if that property combo already exists
     * @throws exception::InvalidAgentVar If the agent var was not found inside the specified agent
     */
    void log(const LoggingConfig::NameReductionFn &name, const std::type_index &variable_type, const std::string &method_name);
    /**
     * Reference to the agent data structure, for validation agent variable names
     */
    const std::shared_ptr<const AgentData> agent;
    /**
     * Reference to the set for storing this agent-states logging choices
     */
    const std::shared_ptr<std::set<LoggingConfig::NameReductionFn>> agent_set;
    /**
     * Bool flag tracking whether the count of agents in the targeted state will be logged
     */
    bool &log_count;
};

/**
 * @brief FLAMEGPU log reduction function pointer definition
 *  this runs on the host as an init/step/exit or host layer function
 */
template<typename T>
detail::Any getAgentVariableMeanFunc(HostAgentAPI &ai, const std::string &variable_name) {
    if (ai.count() > 0)
        return detail::Any(ai.sum<T, typename sum_input_t<T>::result_t>(variable_name) / static_cast<double>(ai.count()));
    return detail::Any(static_cast<double>(0));
}
template<typename T>
detail::Any getAgentVariableSumFunc(HostAgentAPI &ai, const std::string &variable_name) {
    return detail::Any(ai.sum<T, typename sum_input_t<T>::result_t>(variable_name));
}
template<typename T>
detail::Any getAgentVariableMinFunc(HostAgentAPI &ai, const std::string &variable_name) {
    return detail::Any(ai.min<T>(variable_name));
}
template<typename T>
detail::Any getAgentVariableMaxFunc(HostAgentAPI &ai, const std::string &variable_name) {
    return detail::Any(ai.max<T>(variable_name));
}

template<typename T>
detail::Any getAgentVariableStandardDevFunc(HostAgentAPI &ai, const std::string &variable_name) {
    // Todo, workout how to make this more multi-thread/deviceable.
    // Todo, streams for the memcpy?
    if (ai.count() == 0)
        return detail::Any(0.0);
    // Work out the Mean
    const double mean = ai.sum<T, typename sum_input_t<T>::result_t>(variable_name) / static_cast<double>(ai.count());
    // Then for each number: subtract the Mean and square the result
    // Then work out the mean of those squared differences.
    auto lock = std::unique_lock<std::mutex>(detail::STANDARD_DEVIATION_MEAN_mutex);
    gpuErrchk(cudaMemcpyToSymbol(detail::STANDARD_DEVIATION_MEAN, &mean, sizeof(double)));
    const double variance = ai.transformReduce<T, double>(variable_name, detail::standard_deviation_subtract_mean, detail::standard_deviation_add, 0) / static_cast<double>(ai.count());
    lock.unlock();
    // Take the square root of that and we are done!
    return detail::Any(sqrt(variance));
}

template<typename T>
void AgentLoggingConfig::logMean(const std::string &variable_name) {
    // Instantiate the template function for calculating the mean
    LoggingConfig::ReductionFn *fn = getAgentVariableMeanFunc<T>;
    // Log the property (validation occurs in this common log method)
    log({variable_name, LoggingConfig::Mean, fn}, std::type_index(typeid(T)), "Mean");
}
template<typename T>
void AgentLoggingConfig::logStandardDev(const std::string &variable_name) {
    // Instantiate the template function for calculating the mean
    LoggingConfig::ReductionFn *fn = getAgentVariableStandardDevFunc<T>;
    // Log the property (validation occurs in this common log method)
    log({variable_name, LoggingConfig::StandardDev, fn}, std::type_index(typeid(T)), "StandardDev");
}
template<typename T>
void AgentLoggingConfig::logMin(const std::string &variable_name) {
    // Instantiate the template function for calculating the mean
    LoggingConfig::ReductionFn *fn = getAgentVariableMinFunc<T>;
    // Log the property (validation occurs in this common log method)
    log({variable_name, LoggingConfig::Min, fn}, std::type_index(typeid(T)), "Min");
}
template<typename T>
void AgentLoggingConfig::logMax(const std::string &variable_name) {
    // Instantiate the template function for calculating the mean
    LoggingConfig::ReductionFn *fn = getAgentVariableMaxFunc<T>;
    // Log the property (validation occurs in this common log method)
    log({variable_name, LoggingConfig::Max, fn}, std::type_index(typeid(T)), "Max");
}
template<typename T>
void AgentLoggingConfig::logSum(const std::string &variable_name) {
    // Instantiate the template function for calculating the mean
    LoggingConfig::ReductionFn *fn = getAgentVariableSumFunc<T>;
    // Log the property (validation occurs in this common log method)
    log({variable_name, LoggingConfig::Sum, fn}, std::type_index(typeid(T)), "Sum");
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_AGENTLOGGINGCONFIG_H_

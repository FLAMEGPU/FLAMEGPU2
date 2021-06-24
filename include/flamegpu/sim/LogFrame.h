#ifndef INCLUDE_FLAMEGPU_SIM_LOGFRAME_H_
#define INCLUDE_FLAMEGPU_SIM_LOGFRAME_H_

#include "AgentLoggingConfig.h"

#include <map>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "flamegpu/sim/LoggingConfig.h"
#include "flamegpu/util/Any.h"
#include "flamegpu/exception/FGPUException.h"

namespace flamegpu {

struct AgentLogFrame;

/**
 * Generic frame of logging data
 * This can contain logged data related to agents or the environment
 */
struct LogFrame {
    friend class CUDASimulation;
    /**
     * Default constructor, creates an empty log
     */
    LogFrame();
    /**
     * Creates a log with pre-populated data
     */
    LogFrame(const std::map<std::string, util::Any> &&_environment,
    const std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>> &&_agents,
    const unsigned int &_step_count);
    /**
     * Returns the step count of the log
     * 0 is the state prior to the first step
     */
    unsigned int getStepCount() const { return step_count; }
    /**
     * Environment log accessors
     */
    bool hasEnvironmentProperty(const std::string &property_name) const;
    template<typename T>
    T getEnvironmentProperty(const std::string &property_name) const;
    template<typename T, unsigned int N>
    std::array<T, N> getEnvironmentProperty(const std::string &property_name) const;
#ifdef SWIG
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the elements of the environment property array
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T>
    std::vector<T> getEnvironmentPropertyArray(const std::string &property_name) const;
#endif
    /**
     * Agent log accessor
     */
    AgentLogFrame getAgent(const std::string &agent_name, const std::string &state_name = ModelData::DEFAULT_STATE) const;
    /**
     * Raw access to environment log map
     */
    const std::map<std::string, util::Any> &getEnvironment() const { return environment; }
    /**
     * Raw access to agent log map
     */
    const std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>> &getAgents() const { return agents; }

 private:
    std::map<std::string, util::Any> environment;
    std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>> agents;
    unsigned int step_count;
};
/**
 * A collection of LogFrame's related to a single model run
 * The data available depends on the LoggingConfig used at runtime
 */
struct RunLog {
    friend class CUDASimulation;
    /**
     * Constructs an empty RunLog
     */
    RunLog() { }
    /**
     * Constructs a RunLog from existing data frames
     * @param _exit Exit LogFrame
     * @param _step Ordered list of step LogFrames
     */
    RunLog(const LogFrame &_exit, const std::list<LogFrame> &_step)
        : exit(_exit)
        , step(_step) { }
     /**
      * Return the exit LogFrame
      * @return The logging information collected after completion of the model run
      */
    const LogFrame &getExitLog() const { return exit; }
    /**
     * Return the ordered list of step LogFrames
     * @return The logging information collected after each model step
     * @note If logging frequency was changed in the StepLoggingConfig, there may be less than 1 LogFrame per step.
     * @see getStepLogFrequency()
     */
    const std::list<LogFrame> &getStepLog() const {return step; }
    /**
     * Returns the random seed used for this run
     */
    unsigned int getRandomSeed() const { return random_seed; }
    /**
     * Returns the frequency that steps were logged
     * @note This value is configured via StepLoggingConfig::setFrequency()
     */
    unsigned int getStepLogFrequency() const { return step_log_frequency; }

 private:
    /**
     * Exit LogFrame
     */
    LogFrame exit;
    /**
     * Ordered list of step LogFrames
     */
    std::list<LogFrame> step;
    /**
     * Random seed
     */
    unsigned int random_seed = 0;
    /**
     * Step log frequency
     */
    unsigned int step_log_frequency = 0;
};
/**
 * Frame of logging data related to a specific agent type and state.
 * The data available depends on the AgentLoggingConfig used for the agent type at runtime
 */
struct AgentLogFrame {
    /**
     * Constructs an AgentLogFrame from existing data
     * @param data Map of reduction data
     * @param count Population size (alive agents)
     */
    explicit AgentLogFrame(const std::map<LoggingConfig::NameReductionFn, util::Any> &data, const unsigned int &count);
    /**
     * Return the number of alive agents in the population
     * @return The population size
     */
    unsigned int getCount() const;
    /**
     * Return the result of a min reduction performed on the specified agent variable
     * This returns the maximum value of the specified variable found in the agent population
     * @param variable_name The agent variable that was reduced
     * @tparam T The type of agent variable variable_name
     * @return The result of the min reduction
     * @throws exception::InvalidAgentVar If a min reduction of the agent variable of name variable_name was not found within the log.
     * @throws exception::InvalidVarType If the agent variable variable_name does not have type T within the agent.
     */
    template<typename T>
    T getMin(const std::string &variable_name) const;
    /**
     * Return the result of a max reduction performed on the specified agent variable
     * This returns the minimum value of the specified variable found in the agent population
     * @param variable_name The agent variable that was reduced
     * @tparam T The type of agent variable variable_name
     * @return The result of the max reduction
     * @throws exception::InvalidAgentVar If a max reduction of the agent variable of name variable_name was not found within the log.
     * @throws exception::InvalidVarType If the agent variable variable_name does not have type T within the agent.
     */
    template<typename T>
    T getMax(const std::string &variable_name) const;
    /**
     * Return the result of a sum reduction performed on the specified agent variable
     * This returns the result of summing every agent's copy of the specified variable
     * @param variable_name The agent variable that was summed
     * @tparam T The type of agent variable variable_name
     * @return The result of the sum (The type of this return value is the highest range type of the same format)
     * @throws exception::InvalidAgentVar If a sum reduction of the agent variable of name variable_name was not found within the log.
     * @throws exception::InvalidVarType If the agent variable variable_name does not have type T within the agent.
     */
    template<typename T>
    typename sum_input_t<T>::result_t getSum(const std::string &variable_name) const;
    /**
     * Return the result of a mean reduction performed on the specified agent variable
     * This returns the mean average value of the specified agent variable
     * @param variable_name The agent variable that was averaged
     * @tparam T The type of agent variable variable_name
     * @return The result of the average
     * @throws exception::InvalidAgentVar If a mean reduction of the agent variable of name variable_name was not found within the log.
     * @throws exception::InvalidVarType If the agent variable variable_name does not have type T within the agent.
     */
    double getMean(const std::string &variable_name) const;
    /**
     * Return the result of a standard deviation reduction performed on the specified agent variable
     * This returns the standard deviation of the set of agent's copies of the specified agent variable
     * @param variable_name The agent variable that was reduced
     * @tparam T The type of agent variable variable_name
     * @return The result of the standard deviation reduction
     * @throws exception::InvalidAgentVar If a standard deviation reduction of the agent variable of name variable_name was not found within the log.
     * @throws exception::InvalidVarType If the agent variable variable_name does not have type T within the agent.
     */
    double getStandardDev(const std::string &variable_name) const;

 private:
    /**
     * Logging data
     */
    const std::map<LoggingConfig::NameReductionFn, util::Any> &data;
    /**
     * Population size of the related agent state
     */
    const unsigned int &count;
};

template<typename T>
T LogFrame::getEnvironmentProperty(const std::string &property_name) const {
    const auto &it = environment.find(property_name);
    if (it == environment.end()) {
      THROW exception::InvalidEnvProperty("Environment property '%s' was not found in the log, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW exception::InvalidEnvPropertyType("Environment property '%s' has type %s, but requested type %s, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.elements != 1) {
      THROW exception::InvalidEnvPropertyType("Environment property '%s' is an array, use alternate function with array interface, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    return *static_cast<T*>(it->second.ptr);
}
template<typename T, unsigned int N>
std::array<T, N> LogFrame::getEnvironmentProperty(const std::string &property_name) const {
    const auto &it = environment.find(property_name);
    if (it == environment.end()) {
      THROW exception::InvalidEnvProperty("Environment property '%s' was not found in the log, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW exception::InvalidEnvPropertyType("Environment property '%s' has type %s, but requested type %s, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.elements != N) {
      THROW exception::InvalidEnvPropertyType("Environment property array '%s' has %u elements, but requested array with %u, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str(), it->second.elements, N);
    }
    std::array<T, N> rtn;
    memcpy(rtn.data(), it->second.ptr, it->second.length);
    return rtn;
}
#ifdef SWIG
template<typename T>
std::vector<T> LogFrame::getEnvironmentPropertyArray(const std::string& property_name) const {
    const auto &it = environment.find(property_name);
    if (it == environment.end()) {
      THROW exception::InvalidEnvProperty("Environment property '%s' was not found in the log, "
          "in LogFrame::getEnvironmentPropertyArray()\n",
          property_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW exception::InvalidEnvPropertyType("Environment property '%s' has type %s, but requested type %s, "
          "in LogFrame::getEnvironmentPropertyArray()\n",
          property_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    // Copy old data to return
    std::vector<T> rtn(static_cast<size_t>(it->second.elements));
    memcpy(rtn.data(), it->second.ptr, it->second.length);
    return rtn;
}
#endif

template<typename T>
T AgentLogFrame::getMin(const std::string &variable_name) const {
    const auto &it = data.find({variable_name, LoggingConfig::Min});
    if (it == data.end()) {
        THROW exception::InvalidAgentVar("Min of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getMin()\n",
            variable_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW exception::InvalidVarType("Agent variable '%s' has type %s, but requested type %s, "
          "in AgentLogFrame::getMin()\n",
          variable_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    return *static_cast<T *>(it->second.ptr);
}
template<typename T>
T AgentLogFrame::getMax(const std::string &variable_name) const {
    const auto &it = data.find({variable_name, LoggingConfig::Max});
    if (it == data.end()) {
        THROW exception::InvalidAgentVar("Max of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getMax()\n",
            variable_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW exception::InvalidVarType("Agent variable '%s' has type %s, but requested type %s, "
          "in AgentLogFrame::getMax()\n",
          variable_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    return *static_cast<T *>(it->second.ptr);
}
template<typename T>
typename sum_input_t<T>::result_t AgentLogFrame::getSum(const std::string &variable_name) const {
    const auto &it = data.find({variable_name, LoggingConfig::Sum});
    if (it == data.end()) {
        THROW exception::InvalidAgentVar("Sum of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getSum()\n",
            variable_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(typename sum_input_t<T>::result_t))) {
      THROW exception::InvalidVarType("Agent variable is not of type '%s', but requested type %s, "
          "in AgentLogFrame::getSum()\n",
          variable_name.c_str(), std::type_index(typeid(T)).name());
    }
    return *static_cast<typename sum_input_t<T>::result_t *>(it->second.ptr);
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_LOGFRAME_H_

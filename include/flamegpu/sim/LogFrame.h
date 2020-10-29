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

struct AgentLogFrame;

/**
 * Perhaps separate the data and interface in these classes?
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
    LogFrame(const std::map<std::string, Any> &&_environment,
    const std::map<LoggingConfig::NameStatePair, std::pair<std::map<LoggingConfig::NameReductionFn, Any>, unsigned int>> &&_agents,
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
     * @throws InvalidEnvProperty If a property array of the name does not exist
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
    const std::map<std::string, Any> &getEnvironment() const { return environment; }
    /**
     * Raw access to agent log map
     */
    const std::map<LoggingConfig::NameStatePair, std::pair<std::map<LoggingConfig::NameReductionFn, Any>, unsigned int>> &getAgents() const { return agents; }

 private:
    std::map<std::string, Any> environment;
    std::map<LoggingConfig::NameStatePair, std::pair<std::map<LoggingConfig::NameReductionFn, Any>, unsigned int>> agents;
    unsigned int step_count;
};

struct RunLog {
    friend class CUDASimulation;
    /**
     * Constructs an empty RunLog
     */
    RunLog() { }
    RunLog(const LogFrame &_exit, const std::list<LogFrame> &_step)
        : exit(_exit)
        , step(_step) { }

    const LogFrame &getExitLog() const { return exit; }
    const std::list<LogFrame> &getStepLog() const {return step; }
    unsigned int getRandomSeed() const { return random_seed; }

 private:
    LogFrame exit;
    std::list<LogFrame> step;
    unsigned int random_seed = 0;
    unsigned int step_log_frequency = 0;
};
struct AgentLogFrame {
    explicit AgentLogFrame(const std::map<LoggingConfig::NameReductionFn, Any> &data, const unsigned int &count);
    unsigned int getCount() const;
    template<typename T>
    T getMin(const std::string &variable_name) const;
    template<typename T>
    T getMax(const std::string &variable_name) const;
    template<typename T>
    typename sum_input_t<T>::result_t getSum(const std::string &variable_name) const;
    double getMean(const std::string &variable_name) const;
    double getStandardDev(const std::string &variable_name) const;

 private:
    const std::map<LoggingConfig::NameReductionFn, Any> &data;
    const unsigned int &count;
};

template<typename T>
T LogFrame::getEnvironmentProperty(const std::string &property_name) const {
    const auto &it = environment.find(property_name);
    if (it == environment.end()) {
      THROW InvalidEnvProperty("Environment property '%s' was not found in the log, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW InvalidEnvPropertyType("Environment property '%s' has type %s, but requested type %s, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.elements != 1) {
      THROW InvalidEnvPropertyType("Environment property '%s' is an array, use alternate function with array interface, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    return *static_cast<T*>(it->second.ptr);
}
template<typename T, unsigned int N>
std::array<T, N> LogFrame::getEnvironmentProperty(const std::string &property_name) const {
    const auto &it = environment.find(property_name);
    if (it == environment.end()) {
      THROW InvalidEnvProperty("Environment property '%s' was not found in the log, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW InvalidEnvPropertyType("Environment property '%s' has type %s, but requested type %s, "
          "in LogFrame::getEnvironmentProperty()\n",
          property_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.elements != N) {
      THROW InvalidEnvPropertyType("Environment property array '%s' has %u elements, but requested array with %u, "
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
      THROW InvalidEnvProperty("Environment property '%s' was not found in the log, "
          "in LogFrame::getEnvironmentPropertyArray()\n",
          property_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW InvalidEnvPropertyType("Environment property '%s' has type %s, but requested type %s, "
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
        THROW InvalidAgentVar("Min of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getMin()\n",
            variable_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW InvalidVarType("Agent variable '%s' has type %s, but requested type %s, "
          "in AgentLogFrame::getMin()\n",
          variable_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    return *static_cast<T *>(it->second.ptr);
}
template<typename T>
T AgentLogFrame::getMax(const std::string &variable_name) const {
    const auto &it = data.find({variable_name, LoggingConfig::Max});
    if (it == data.end()) {
        THROW InvalidAgentVar("Max of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getMax()\n",
            variable_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(T))) {
      THROW InvalidVarType("Agent variable '%s' has type %s, but requested type %s, "
          "in AgentLogFrame::getMax()\n",
          variable_name.c_str(), it->second.type.name(), std::type_index(typeid(T)).name());
    }
    return *static_cast<T *>(it->second.ptr);
}
template<typename T>
typename sum_input_t<T>::result_t AgentLogFrame::getSum(const std::string &variable_name) const {
    const auto &it = data.find({variable_name, LoggingConfig::Sum});
    if (it == data.end()) {
        THROW InvalidAgentVar("Sum of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getSum()\n",
            variable_name.c_str());
    }
    if (it->second.type != std::type_index(typeid(typename sum_input_t<T>::result_t))) {
      THROW InvalidVarType("Agent variable is not of type '%s', but requested type %s, "
          "in AgentLogFrame::getSum()\n",
          variable_name.c_str(), std::type_index(typeid(T)).name());
    }
    return *static_cast<typename sum_input_t<T>::result_t *>(it->second.ptr);
}

#endif  // INCLUDE_FLAMEGPU_SIM_LOGFRAME_H_

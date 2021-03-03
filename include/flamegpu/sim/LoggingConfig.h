#ifndef INCLUDE_FLAMEGPU_SIM_LOGGINGCONFIG_H_
#define INCLUDE_FLAMEGPU_SIM_LOGGINGCONFIG_H_

#include <string>
#include <map>
#include <set>
#include <utility>
#include <memory>

#include "flamegpu/runtime/HostAgentAPI.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/gpu/CUDAEnsemble.h"

class AgentLoggingConfig;

/**
 * Interface to the data structure for controlling how model data is logged
 */
class LoggingConfig {
    /**
     * Requires access to typedefs and enums
     */
    friend class AgentLoggingConfig;
    /**
     * Requires access to model for validating model description hierarchy match
     */
    // friend void CUDASimulation::setStepLog(const StepLoggingConfig &);
    // friend void CUDASimulation::setExitLog(const LoggingConfig &);
    friend void CUDAEnsemble::setStepLog(const StepLoggingConfig &);
    friend void CUDAEnsemble::setExitLog(const LoggingConfig &);
    /**
     * CUDASimulation::processStepLog() Requires access for reading the config
     */
    friend class CUDASimulation;

 public:
    /**
     * Enum representing the available reduction types for agent variables
     */
    enum Reduction{ Mean, StandardDev, Min, Max, Sum };
    static constexpr const char *toString(const Reduction &r) {
        switch (r) {
        case Mean: return "mean";
        case StandardDev:  return "standard_deviation";;
        case Min: return "min";
        case Max: return "max";
        case Sum: return "sum";
        default: return "unknown";
        }
    }
    typedef std::pair<std::string, std::string> NameStatePair;
    typedef Any (ReductionFn)(HostAgentAPI &ai, const std::string &variable_name);
    struct NameReductionFn {
        std::string name;
        Reduction reduction;
        ReductionFn *function;
        bool operator<(const NameReductionFn &other) const {
            if (name == other.name) {
                return reduction < other.reduction;
            }
            return name < other.name;
        }
    };
    /**
     * Constructor
     * @param model The ModelDescriptionHierarchy to produce a logging config for
     */
    explicit LoggingConfig(const ModelDescription &model);
    explicit LoggingConfig(const ModelData &model);
    /**
     * Copy Constructor
     */
    explicit LoggingConfig(const LoggingConfig &other);
    /**
     * Returns an interface to the logging config for the named agent state
     */
    AgentLoggingConfig agent(const std::string &agent_name, const std::string &agent_state = ModelData::DEFAULT_STATE);
    /**
     * Mark a named environment property to be logged
     * @param property_name Name of the environment property to be logged
     * @throws InvalidEnvironment
     */
    void logEnvironment(const std::string &property_name);

 private:
    std::shared_ptr<const ModelData> model;
    std::set<std::string> environment;
    /**
     * map<<agent_name:agent_state, variable_reductions:log_count>
     */
    std::map<NameStatePair, std::pair<std::shared_ptr<std::set<NameReductionFn>>, bool>> agents;
};

class StepLoggingConfig : public LoggingConfig {
    /**
     * CUDASimulation::processStepLog() requires access for reading the config
     */
    friend class CUDASimulation;
 public:
    /**
     * Constructor
     * @param model The ModelDescriptionHierarchy to produce a logging config for
     */
    explicit StepLoggingConfig(const ModelDescription &model);
    explicit StepLoggingConfig(const ModelData &model);
    /**
     * Copy Constructor
     */
    explicit StepLoggingConfig(const StepLoggingConfig &model);
    explicit StepLoggingConfig(const LoggingConfig &model);
    /**
     * Set the frequency of step log collection
     * How many steps between each log collection, defaults to 1, so a log is collected every step
     * A value of 0 disables step log collection
     */
    void setFrequency(const unsigned int &steps);

 private:
    unsigned int frequency;
};

#endif  // INCLUDE_FLAMEGPU_SIM_LOGGINGCONFIG_H_

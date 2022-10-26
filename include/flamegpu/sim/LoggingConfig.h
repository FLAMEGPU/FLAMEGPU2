#ifndef INCLUDE_FLAMEGPU_SIM_LOGGINGCONFIG_H_
#define INCLUDE_FLAMEGPU_SIM_LOGGINGCONFIG_H_

#include <string>
#include <map>
#include <set>
#include <utility>
#include <memory>

#include "flamegpu/util/StringPair.h"
#include "flamegpu/runtime/HostAgentAPI.cuh"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/gpu/CUDAEnsemble.h"

namespace flamegpu {

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
    /**
     * Requires access to log_timing
     */
    friend unsigned int CUDAEnsemble::simulate(const RunPlanVector& plans);

 public:
    /**
     * Enum representing the available reduction types for agent variables
     */
    enum Reduction{ Mean, StandardDev, Min, Max, Sum };
    /**
     * Converts a Reduction enum to a string representation
     */
    static constexpr const char *toString(const Reduction &r) {
        switch (r) {
        case Mean: return "mean";
        case StandardDev:  return "standard_deviation";
        case Min: return "min";
        case Max: return "max";
        case Sum: return "sum";
        default: return "unknown";
        }
    }
    /**
     * ReductionFn is a prototype for reduction functions
     * Typedef'ing function prototypes like this allows for cleaner function pointers
     * @note - this leads to a swig warning 504 which is suppressed.
     */
    typedef util::Any (ReductionFn)(HostAgentAPI &ai, const std::string &variable_name);
    /**
     * A user configured reduction to be logged
     */
    struct NameReductionFn {
        /**
         * Variable name to reduce over
         */
        std::string name;
        /**
         * The type of reduction
         */
        Reduction reduction;
        /**
         * Pointer to instantiated reduction function
         * (Reduction functions are templated so much be instantiated)
         */
        ReductionFn *function;
        /**
         * Generic ordering function, to allow instances of this type to be stored in ordered collections
         * The defined order is not important
         * @param other The other instance to compare vs
         * @return Returns whether this instance should come before other
         */
        bool operator<(const NameReductionFn &other) const {
            if (name == other.name) {
                return reduction < other.reduction;
            }
            return name < other.name;
        }
    };
    /**
     * Constructor
     * @param model The ModelDescription hierarchy to produce a logging config for
     */
    explicit LoggingConfig(const ModelDescription &model);
    /**
     * Constructor
     * @param model The ModelDescription hierarchy to produce a logging config for
     */
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
    /**
     * Log timing information for the complete simulation to file
     * In the case of step log, this will cause the step time to be logged
     * @param doLogTiming True if timing data should be logged
     * @note By default, timing information is not logged to file
     * @note Timing information will always be available in the programmatic logs accessed via code
     */
    void logTiming(bool doLogTiming);

 private:
    /**
     * The ModelDescription hierarchy to setup the logging for
     */
    std::shared_ptr<const ModelData> model;
    /**
     * Set of environment properties to be logged
     */
    std::set<std::string> environment;
    /**
     * Map of variable reductions per agent state to be logged
     * map<<agent_name:agent_state, variable_reductions:log_count>
     */
    std::map<util::StringPair, std::pair<std::shared_ptr<std::set<NameReductionFn>>, bool>> agents;
    /**
     * Flag denoting whether timing information for the simulation/steps should be logged
     */
    bool log_timing;
};

/**
 * Interface to the data structure for controlling how model data is logged each step
 */
class StepLoggingConfig : public LoggingConfig {
    /**
     * CUDASimulation::processStepLog() requires access for reading the config
     */
    friend class CUDASimulation;
 public:
    /**
     * Constructor
     * @param model The ModelDescription hierarchy to produce a logging config for
     */
    explicit StepLoggingConfig(const ModelDescription &model);
    /**
     * Constructor
     * @param model The ModelDescription hierarchy to produce a logging config for
     */
    explicit StepLoggingConfig(const ModelData &model);
    /**
     * Copy Constructor
     * @param model Other
     */
    explicit StepLoggingConfig(const StepLoggingConfig &model);
    /**
     * Copy Constructor
     * @param model Other
     */
    explicit StepLoggingConfig(const LoggingConfig &model);
    /**
     * Set the frequency of step log collection
     * How many steps between each log collection, defaults to 1, so a log is collected every step
     * A value of 0 disables step log collection
     */
    void setFrequency(unsigned int steps);

 private:
    /**
     * Frequency that step logs should be collected
     * A value of 1 will collect a log every step, a value of 2 will collect a log every 2 steps, etc
     */
    unsigned int frequency;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_LOGGINGCONFIG_H_

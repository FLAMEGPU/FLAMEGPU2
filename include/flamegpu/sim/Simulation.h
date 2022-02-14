#ifndef INCLUDE_FLAMEGPU_SIM_SIMULATION_H_
#define INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

#include <memory>
#include <string>
#include <ctime>
#include <utility>
#include <unordered_map>

#include "flamegpu/sim/AgentInterface.h"
#include "flamegpu/util/StringUint32Pair.h"


namespace flamegpu {

class AgentVector;
class HostAPI;
class ModelDescription;
struct ModelData;
struct RunLog;


/**
 * Abstract representation of a ModelDescription that can be executed
 * @see CUDASimulation The CUDA implementation of a Simulation
 */
class Simulation {
 public:
    /**
     * General simulation runner specific config
     */
    struct Config {
        Config() : random_seed(static_cast<uint64_t>(time(nullptr))) {
        }
        void operator=(const Config &other) {
            input_file = other.input_file;
            step_log_file = other.step_log_file;
            exit_log_file = other.exit_log_file;
            common_log_file = other.common_log_file;
            truncate_log_files = other.truncate_log_files;
            random_seed = other.random_seed;
            steps = other.steps;
            verbose = other.verbose;
            timing = other.timing;
#ifdef VISUALISATION
            console_mode = other.console_mode;
#endif
        }
        std::string input_file;
        std::string step_log_file;
        std::string exit_log_file;
        std::string common_log_file;
        bool truncate_log_files = true;
        uint64_t random_seed;
        unsigned int steps = 1;
        bool verbose = false;
        bool timing = false;
#ifdef VISUALISATION
        bool console_mode = false;
#else
        const bool console_mode = true;
#endif
    };
    virtual ~Simulation() = default;
    /**
     * This constructor takes a clone of the ModelData hierarchy
     */
    explicit Simulation(const std::shared_ptr<const ModelData> &model);

 protected:
    /**
     * This constructor is for use with submodels, it does not call clone
     * Additionally sets the submodel ptr
     */
    explicit Simulation(const std::shared_ptr<SubModelData> &sub_model, CUDASimulation *master_model);

 public:
    void initialise(int argc, const char** argv);

    virtual void initFunctions() = 0;
    virtual bool step() = 0;
    virtual void exitFunctions() = 0;
    virtual void simulate() = 0;
    /**
     * Returns the simulation to a clean state
     * This clears all agents and message lists, resets environment properties and reseeds random generation.
     * Also calls resetStepCounter();
     * @note If triggered on a submodel, agent states and environment properties mapped to a parent agent, and random generation are not affected.
     * @note If random was manually seeded, it will return to it's original state. If random was seeded from time, it will return to a new random state.
     */
    void reset();
    virtual unsigned int getStepCounter() = 0;
    virtual void resetStepCounter() = 0;

    const ModelData& getModelDescription() const;
    /**
     * Export model state to file
     * Export includes config structures, environment and agent data
     * @param path The file to output (must end '.json' or '.xml')
     * @param prettyPrint Whether to include indentation and line breaks to aide human reading
     * @note XML export does not currently includes config structures, only the same data present in FLAMEGPU1
     */
    void exportData(const std::string &path, bool prettyPrint = true);
    /**
     * Export the data logged by the last call to simulate() (and/or step) to the given path
     * @param path The file to output (must end '.json' or '.xml')
     * @param steps Whether the step log should be included in the log file
     * @param exit Whether the exit log should be included in the log file
     * @param stepTime Whether the step time should be included in the log file (always treated as false if steps=false)
     * @param exitTime Whether the simulation time should be included in the log file (always treated as false if exit=false)
     * @param prettyPrint Whether the log file should be minified or not
     * @note The config (possibly just random seed) is always output
     */
    void exportLog(const std::string &path, bool steps, bool exit, bool stepTime, bool exitTime, bool prettyPrint = true);

    virtual void setPopulationData(AgentVector& population, const std::string& state_name = ModelData::DEFAULT_STATE) = 0;
    virtual void getPopulationData(AgentVector& population, const std::string& state_name = ModelData::DEFAULT_STATE) = 0;

    virtual const RunLog &getRunLog() const = 0;
    virtual AgentInterface &getAgent(const std::string &name) = 0;

    Config &SimulationConfig();
    const Config &getSimulationConfig() const;

    void applyConfig();

 protected:
    /**
     * Returns the model to a clean state
     * This clears all agents and message lists, resets environment properties and reseeds random generation.
     * Also calls resetStepCounter();
     * @param submodelReset This should only be set to true when called automatically when a submodel reaches it's exit condition during execution. This performs a subset of the regular reset procedure.
     * @note If triggered on a submodel, agent states and environment properties mapped to a parent agent, and random generation are not affected.
     * @note If random was manually seeded, it will return to it's original state. If random was seeded from time, it will return to a new random state.
     */
    virtual void reset(bool submodelReset) = 0;
    virtual void applyConfig_derived() = 0;
    virtual bool checkArgs_derived(int argc, const char** argv, int &i) = 0;
    virtual void printHelp_derived() = 0;
    virtual void resetDerivedConfig() = 0;
    /**
     * Returns the unique instance id of this CUDASimulation instance
     * @note This value is used internally for environment property storage
     */
    unsigned int getInstanceID() const { return instance_id; }

    /**
     * returns the width of the widest layer in model.
     * @return the width of the widest layer.
     */
    unsigned int getMaximumLayerWidth() const { return maxLayerWidth; }

    const std::shared_ptr<const ModelData> model;

    /**
     * This is only used when model is a submodel, otherwise it is empty
     * If it is set, it causes simulate() to additionally reset/cull unmapped populations
     */
    const std::shared_ptr<const SubModelData> submodel;
    /**
     * Only used by submodels, only required to fetch the name of master model when initialising environment (as this occurs after constructor)
     */
    CUDASimulation const * mastermodel;

    Config config;
    /**
     * If this matches config.input_file, apply_config() will not load the input file
     */
    std::string loaded_input_file;
    /**
     * Unique index of Simulation instance
     */
    const unsigned int instance_id;
    /**
     * Initial environment items if they have been loaded from file, prior to device selection
     */
    util::StringUint32PairUnorderedMap<util::Any> env_init;
    /**
     * the width of the widest layer in the concrete version of the model (calculated once)
     */
    unsigned int maxLayerWidth;

 private:
    /**
     * Generates a unique id number for the instance
     */
    static unsigned int get_instance_id();
    void printHelp(const char *executable);
    int checkArgs(int argc, const char** argv);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

#ifndef INCLUDE_FLAMEGPU_SIM_SIMULATION_H_
#define INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

#include <memory>
#include <string>
#include <ctime>

#include "flamegpu/sim/AgentInterface.h"

class FLAMEGPU_HOST_API;
class ModelDescription;
class AgentPopulation;
struct ModelData;

class Simulation {
 public:
    struct Config {
        Config() : random_seed(static_cast<unsigned int>(time(nullptr))) {
        }
        std::string xml_input_file;
        unsigned int random_seed;
        unsigned int steps = 0;
        bool verbose = false;
        bool timing = false;
    };
    virtual ~Simulation() = default;
    /**
     * This constructor takes a clone of the ModelData hierarchy
     */
    explicit Simulation(const ModelDescription& model);

 protected:
    /**
     * This constructor is for use with submodels, it does not call clone
     * Additionally sets the submodel ptr
     */
    explicit Simulation(const std::shared_ptr<SubModelData> &sub_model, CUDAAgentModel *master_model);

 public:
    void initialise(int argc, const char** argv);

    virtual bool step() = 0;
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

    void exportData(const std::string &path);

    virtual void setPopulationData(AgentPopulation& population) = 0;
    virtual void getPopulationData(AgentPopulation& population) = 0;

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
     * Returns the unique instance id of this CUDAAgentModel instance
     * @note This value is used internally for environment property storage
     */
    unsigned int getInstanceID() const { return instance_id; }
    const std::shared_ptr<const ModelData> model;

    /**
     * This is only used when model is a submodel, otherwise it is empty
     * If it is set, it causes simulate() to additionally reset/cull unmapped populations
     */
    const std::shared_ptr<const SubModelData> submodel;
    /**
     * Only used by submodels, only required to fetch the name of master model when initialising environment (as this occurs after constructor)
     */
    CUDAAgentModel const * mastermodel;

    Config config;
    /**
     * Unique index of Simulation instance
     */
    const unsigned int instance_id;

 private:
    /**
     * Generates a unique id number for the instance
     */
    static unsigned int get_instance_id();
    void printHelp(const char *executable);
    int checkArgs(int argc, const char** argv);
};

#endif  // INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

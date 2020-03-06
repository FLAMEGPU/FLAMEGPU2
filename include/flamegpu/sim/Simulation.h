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
    };
    virtual ~Simulation() = default;
    explicit Simulation(const ModelDescription& model);
    void initialise(int argc, const char** argv);

    virtual bool step() = 0;
    virtual void simulate() = 0;
    virtual unsigned int getStepCounter() = 0;
    virtual void resetStepCounter() = 0;

    const ModelData& getModelDescription() const;

    void output(int argc, const char** argv);

    virtual void setPopulationData(AgentPopulation& population) = 0;
    virtual void getPopulationData(AgentPopulation& population) = 0;

    virtual AgentInterface &getAgent(const std::string &name) = 0;

    Config &SimulationConfig();
    const Config &getSimulationConfig() const;

    void applyConfig();

 protected:
    virtual void applyConfig_derived() = 0;
    virtual bool checkArgs_derived(int argc, const char** argv, int &i) = 0;
    virtual void printHelp_derived() = 0;
    virtual void resetDerivedConfig() = 0;
    const std::shared_ptr<const ModelData> model;

    Config config;

 private:
    void printHelp(const char *executable);
    int checkArgs(int argc, const char** argv);
};

#endif  // INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

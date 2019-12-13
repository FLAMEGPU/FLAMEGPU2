#ifndef INCLUDE_FLAMEGPU_SIM_SIMULATION_H_
#define INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

#include <memory>
#include <string>

#include "flamegpu/sim/AgentInterface.h"

class FLAMEGPU_HOST_API;
class ModelDescription;
class AgentPopulation;
struct ModelData;

class Simulation {
 public:
    virtual ~Simulation() = default;
    Simulation(const ModelDescription& model);
    void initialise(int argc, const char** argv);
    
    virtual bool step() = 0;
    virtual void simulate() = 0;

    void setSimulationSteps(unsigned int steps);
    unsigned int getSimulationSteps() const;
    const ModelData& getModelDescription() const;

    void output(int argc, const char** argv);

    virtual void setPopulationData(AgentPopulation& population) = 0;
    virtual void getPopulationData(AgentPopulation& population) = 0;

    virtual AgentInterface &getAgent(const std::string &name) = 0;
 protected:
    virtual void _initialise() = 0;
    virtual bool checkArgs_derived(int argc, const char** argv) = 0;
    virtual void printHelp_derived() = 0;
    const std::shared_ptr<const ModelData> model;
    /**
     * One instance of host api is used for entire model
     */
    std::unique_ptr<FLAMEGPU_HOST_API> host_api;

 private:
     void printHelp(const char *executable);
    int checkArgs(int argc, const char** argv);
    unsigned int simulation_steps;
    std::string xml_input_path;
    bool has_seed;
    unsigned int random_seed;

};

#endif  // INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

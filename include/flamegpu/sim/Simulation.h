#ifndef INCLUDE_FLAMEGPU_SIM_SIMULATION_H_
#define INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

#include <vector>
#include <string.h>

// include class dependencies
#include <flamegpu/sim/SimulationLayer.h>

// forward declare dependencies from other modules
class ModelDescription;
class StateReader;

typedef std::vector<std::reference_wrapper<SimulationLayer>> SimulationLayerVector;

/**
 * A much nice user method of specifying layer order is required!
 */
class Simulation {

 public:
    Simulation(const ModelDescription& model);
    ~Simulation(void);

    unsigned int addSimulationLayer(SimulationLayer& layer);
    void setSimulationSteps(unsigned int steps);
    unsigned int getSimulationSteps() const;
    const ModelDescription& getModelDescritpion() const;

    const FunctionDescriptionVector& getFunctionsAtLayer(unsigned int layer) const;
    unsigned int getLayerCount() const;

    int checkArgs(int argc, char** argv);
    
    void initialise(int argc, char** argv); // void initialise(const char * input);
    // void initialise(StateReader& reader);
    void output(int argc, char** argv);

 private:
    SimulationLayerVector layers;
    const ModelDescription& model_description;
    unsigned int simulation_steps;
};

#endif //INCLUDE_FLAMEGPU_SIM_SIMULATION_H_


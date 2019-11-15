#ifndef INCLUDE_FLAMEGPU_SIM_SIMULATION_H_
#define INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

#include <vector>
#include <string>
#include <set>

// include class dependencies
#include "flamegpu/sim/SimulationLayer.h"
#include "flamegpu/runtime/flamegpu_host_api.h"

// forward declare dependencies from other modules
class ModelDescription;
class StateReader;


/**
 * A much nice user method of specifying layer order is required!
 */
class Simulation {
 public:
    typedef std::vector<std::reference_wrapper<SimulationLayer>> SimulationLayerVector;
    typedef std::set<FLAMEGPU_INIT_FUNCTION_POINTER> InitFunctionSet;
    typedef std::set<FLAMEGPU_STEP_FUNCTION_POINTER> StepFunctionSet;
    typedef std::set<FLAMEGPU_EXIT_FUNCTION_POINTER> ExitFunctionSet;
    typedef std::set<FLAMEGPU_EXIT_CONDITION_POINTER> ExitConditionSet;

    explicit Simulation(const ModelDescription& model);
    ~Simulation(void);

    unsigned int addSimulationLayer(SimulationLayer &layer);
    /**
     * Adds an init function to the simulation
     * Init functions execute once before the simulation begins
     * @param func_p Pointer to the desired init function
     */
    void addInitFunction(const FLAMEGPU_INIT_FUNCTION_POINTER *func_p);
    /**
     * Adds a step function to the simulation
     * Step functions execute once per step, after all layers have been executed, before exit conditions
     * @param func_p Pointer to the desired step function
     */
    void addStepFunction(const FLAMEGPU_STEP_FUNCTION_POINTER *func_p);
    /**
     * Adds an exit function to the simulation
     * Exit functions execute once after the simulation ends
     * @param func_p Pointer to the desired exit function
     */
    void addExitFunction(const FLAMEGPU_EXIT_FUNCTION_POINTER *func_p);
    /**
     * Adds an exit condition function to the simulation
     * Exit conditions execute once per step, after all layers and step functions have been executed
     * If the condition returns false, the simulation exits early
     * @param func_p Pointer to the desired exit condition function
     */
    void addExitCondition(const FLAMEGPU_EXIT_CONDITION_POINTER *func_p);
    void setSimulationSteps(unsigned int steps);
    unsigned int getSimulationSteps() const;
    const ModelDescription& getModelDescritpion() const;

    /**
     * Returns the set of host functions for the given layer
     * @throws InvalidMemoryCapacity If layer number is invalid
     */
    const SimulationLayer::FunctionDescriptionVector& getFunctionsAtLayer(const unsigned int &layer) const;
    /**
     * Returns the set of host functions for the given layer
     * @throws InvalidMemoryCapacity If layer number is invalid
     */
    const SimulationLayer::HostFunctionSet& getHostFunctionsAtLayer(const unsigned int &layer) const;
    /**
     * Returns the set of init functions attached to the simulation
     */
    const InitFunctionSet& getInitFunctions() const;
    /**
     * Returns the set of step functions attached to the simulation
     */
    const StepFunctionSet& getStepFunctions() const;
    /**
     * Returns the set of exit functions attached to the simulation
     */
    const ExitFunctionSet& getExitFunctions() const;
    /**
     * Returns the set of exit condition attached to the simulation
     */
    const ExitConditionSet& getExitConditions() const;

    unsigned int getLayerCount() const;

    int checkArgs(int argc, const char** argv, std::string &xml_model_path);

    void initialise(int argc, const char** argv);

    // void initialise(StateReader& reader);
    void output(int argc, const char** argv);

 private:
    static void printHelp(const char *executable);

    SimulationLayerVector layers;
    InitFunctionSet initFunctions;
    StepFunctionSet stepFunctions;
    ExitFunctionSet exitFunctions;
    ExitConditionSet exitConditions;
    const ModelDescription& model_description;
    unsigned int simulation_steps;
};

#endif  // INCLUDE_FLAMEGPU_SIM_SIMULATION_H_

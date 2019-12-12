#ifndef INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_

#include <string>
#include <map>
#include <memory>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/sim/Simulation.h"

class AgentDescription;
class MessageDescription;
class LayerDescription;
struct ModelData;

class ModelDescription {
    friend Simulation::Simulation(const ModelDescription& model);
 public:
    /**
     * Constructors
     */
    ModelDescription(const std::string &model_name);
    // Copy Construct
    ModelDescription(const ModelDescription &other_model);
    // Move Construct
    ModelDescription(ModelDescription &&other_model);
    // Copy Assign
    ModelDescription& operator=(const ModelDescription &other_model);
    // Move Assign
    ModelDescription& operator=(ModelDescription &&other_model);

    /**
     * Accessors
     */    
    AgentDescription& newAgent(const std::string &agent_name);
    AgentDescription& Agent(const std::string &agent_name);
    AgentDescription& cloneAgent(const AgentDescription &agent);
    
    MessageDescription& newMessage(const std::string &message_name);
    MessageDescription& Message(const std::string &message_name);
    MessageDescription& cloneMessage(const MessageDescription &message);
    
    EnvironmentDescription& Environment();
    EnvironmentDescription& cloneEnvironment(const EnvironmentDescription &environment);

    LayerDescription& newLayer(const std::string &name = "");
    LayerDescription& Layer(const std::string &name);
    LayerDescription& Layer(const ModelData::size_type &layer_index);
    
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

    /**
     * Const Accessors
     */
    std::string getName() const;
    
    const AgentDescription& getAgent(const std::string &agent_name) const;
    const MessageDescription& getMessage(const std::string &message_name) const;
    const EnvironmentDescription& getEnvironment() const;
    const LayerDescription& getLayer(const std::string &name) const;
    const LayerDescription& getLayer(const ModelData::size_type &layer_index) const;

    bool hasAgent(const std::string &agent_name) const;
    bool hasMessage(const std::string &message_name) const;
    bool hasLayer(const std::string &name) const;
    bool hasLayer(const ModelData::size_type &layer_index) const;

    ModelDescription clone(const std::string &cloned_model_name) const;

 private:
     ModelData *const model;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_

#ifndef INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_

#include <map>
#include <memory>
#include <set>
#include <string>


#include "flamegpu/gpu/CUDAEnsemble.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceHost.h"

namespace flamegpu {

class AgentDescription;
class CAgentDescription;
class CLayerDescription;
class LayerDescription;
class CSubModelDescription;
class SubModelDescription;
class CEnvironmentDescription;
class EnvironmentDescription;
class DependencyNode;
struct ModelData;

/**
 * This class represents the hierarchy of components for a FLAMEGPU model
 * This is the initial class that should be created by a modeller
 * @see ModelData The internal data store for this class
 */
class ModelDescription {
    /**
     * Simulation accesses the classes internals to convert it to a constant ModelData
     */
    friend CUDASimulation::CUDASimulation(const ModelDescription& _model, int argc, const char** argv);
    friend CUDAEnsemble::CUDAEnsemble(const ModelDescription& model, int argc, const char** argv);
    friend class RunPlanVector;
    friend class RunPlan;
    friend class LoggingConfig;
 public:
    /**
     * Constructor
     * @param model_name Name of the model, this must be unique between models currently held by instances of CUDASimulation
     */
    explicit ModelDescription(const std::string &model_name);
    /**
     * Default copy constructor, not implemented
     */
    ModelDescription(const ModelDescription &other_model) = delete;
    /**
     * Default move constructor, not implemented
     */
    ModelDescription(ModelDescription &&other_model) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    ModelDescription& operator=(const ModelDescription &other_model) = delete;
    /**
     * Default move assignment, not implemented
     */
    ModelDescription& operator=(ModelDescription &&other_model) noexcept = delete;
    /**
     * Equality operator, checks whether ModelDescription hierarchies are functionally the same
     * @returns True when models are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const ModelDescription& rhs) const;
    /**
     * Equality operator, checks whether ModelDescription hierarchies are functionally different
     * @returns True when models are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const ModelDescription& rhs) const;

    /**
     * Creates a new agent with the specified name
     * @param agent_name Name which can be used to refer to the agent within the model description hierarchy
     * @return A mutable reference to the specified AgentDescription
     * @throws exception::InvalidAgentName If an agent with the same name already exists within the model description hierarchy
     */
    AgentDescription newAgent(const std::string &agent_name);
    /**
     * Returns a mutable reference to the named agent, which can be used to configure the agent
     * @param agent_name Name which can be used to the refer to the desired agent within the model description hierarchy
     * @return A mutable reference to the specified AgentDescription
     * @throws exception::InvalidAgentName If an agent with the name does not exist within the model description hierarchy
     * @see ModelDescription::getAgent(const std::string &) for the immutable version
     */
    AgentDescription Agent(const std::string &agent_name);

    /**
     * Creates a new message with the specified name
     * @param message_name Name which can be used to refer to the message within the model description hierarchy
     * @return A mutable reference to the new MessageDescription
     * @throws exception::InvalidMessageName If a message with the same name already exists within the model description hierarchy
     */
    template<typename MessageType>
    typename MessageType::Description newMessage(const std::string &message_name) {
        if (!hasMessage<MessageType>(message_name)) {
            auto rtn = std::shared_ptr<typename MessageType::Data>(new typename MessageType::Data(model, message_name));
            model->messages.emplace(message_name, rtn);
            return typename MessageType::Description(rtn);
        }
        THROW exception::InvalidMessageName("Message with name '%s' already exists, "
            "in ModelDescription::newMessage().",
            message_name.c_str());
    }
    MessageBruteForce::Description newMessage(const std::string &message_name);
    /**
     * Returns a mutable reference to the named message, which can be used to configure the message
     * @param message_name Name used to refer to the desired message within the model description hierarchy
     * @return A mutable reference to the specified MessageDescription
     * @throws exception::InvalidMessageName If a message with the name does not exist within the model description hierarchy
     * @see ModelDescription::getMessage(const std::string &) for the immutable version
     */
    template<typename MessageType>
    typename MessageType::Description Message(const std::string &message_name) {
        auto rtn = model->messages.find(message_name);
        if (rtn != model->messages.end()) {
            if (auto r = std::dynamic_pointer_cast<typename MessageType::Data>(rtn->second)) {
                return typename MessageType::Description(r);
            }
            THROW exception::InvalidMessageName("Message ('%s') is not of correct type, "
                "in ModelDescription::Message().",
                message_name.c_str());
        }
        THROW exception::InvalidMessageName("Message ('%s') was not found, "
            "in ModelDescription::Message().",
            message_name.c_str());
    }
    MessageBruteForce::Description Message(const std::string &message_name);
    /**
     * Returns a mutable reference to the environment description for the model description hierarchy
     * This can be used to configure environment properties
     * @see ModelDescription::getEnvironment() for the immutable version
     */
    EnvironmentDescription Environment();
    /**
     * Add a submodel to the Model Description hierarchy
     * The return value can be used to map agent variables
     * @param submodel_name The name used to refer to the submodel (e.g. when adding it to the layer)
     * @param submodel_description The actual definition of the submodel
     */
    SubModelDescription newSubModel(const std::string &submodel_name, const ModelDescription &submodel_description);
    /**
     * Returns a mutable reference to the named submodel
     * @param submodel_name Name which can be used to the refer to the desired submodel within the model description hierarchy
     * @return A mutable reference to the specified SubModelDescription
     * @throws exception::InvalidSubModelName If a submodel with the name does not exist within the model description hierarchy
     * @see ModelDescription::getSubModel(const std::string &) for the immutable version
     */
    SubModelDescription SubModel(const std::string &submodel_name);

    /**
     * Creates a new layer with the specified name
     * @param name Name which can be used to refer to the message within the model description hierarchy
     * @return A mutable reference to the new LayerDescription
     * @throws exception::InvalidFuncLayerIndx If a layer with the same name already exists within the model description hierarchy
     * @note Layer names are not required, passing empty string will not set a name
     */
    LayerDescription newLayer(const std::string &name = "");
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param name Name used to refer to the desired layer within the model description hierarchy
     * @return A mutable reference to the specified LayerDescription
     * @throws exception::InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::Layer(const flamegpu::size_type &)
     * @see ModelDescription::getLayer(const std::string &) for the immutable version
     */
    LayerDescription Layer(const std::string &name);
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param layer_index Index of the desired layer within the model description hierarchy
     * @return A mutable reference to the specified LayerDescription
     * @throws exception::InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::Layer(const std::string &)
     * @see ModelDescription::getLayer(const flamegpu::size_type &) for the immutable version
     */
    LayerDescription Layer(const flamegpu::size_type &layer_index);

    /**
     * Adds an init function to the simulation
     * Init functions execute once before the simulation begins
     * @param func_p Pointer to the desired init function
     * @throws exception::InvalidHostFunc If the init function has already been added to this model description
     * @note Init functions are executed in the order they were added to the model
     */
    void addInitFunction(FLAMEGPU_INIT_FUNCTION_POINTER func_p);
    /**
     * Adds a step function to the simulation
     * Step functions execute once per step, after all layers have been executed, before exit conditions
     * @param func_p Pointer to the desired step function
     * @throws exception::InvalidHostFunc If the step function has already been added to this model description
     * @note Step functions are executed in the order they were added to the model
     */
    void addStepFunction(FLAMEGPU_STEP_FUNCTION_POINTER func_p);
    /**
     * Adds an exit function to the simulation
     * Exit functions execute once after all simulation steps have completed or an exit conditions has returned EXIT
     * @param func_p Pointer to the desired exit function
     * @throws exception::InvalidHostFunc If the exit function has already been added to this model description
     * @note Exit functions are executed in the order they were added to the model
     */
    void addExitFunction(FLAMEGPU_EXIT_FUNCTION_POINTER func_p);
#ifdef SWIG
    /**
     * Adds an init function callback to the simulation. The callback objects is similar to adding via addInitFunction
     * however the runnable function is encapsulated within an object which permits cross language support in swig.
     * Init functions execute once before the simulation begins
     * @param func_callback Pointer to the desired init function callback
     * @throws exception::InvalidHostFunc If the init function has already been added to this model description
     * @note Init functions are executed in the order they were added to the model
     */
    inline void addInitFunctionCallback(HostFunctionCallback *func_callback);
    /**
     * Adds an step function callback to the simulation. The callback objects is similar to adding via addStepFunction
     * however the runnable function is encapsulated within an object which permits cross language support in swig.
     * Exit functions execute once after the simulation ends
     * @param func_callback Pointer to the desired exit function callback
     * @throws exception::InvalidHostFunc If the step function has already been added to this model description
     * @note Step functions are executed in the order they were added to the model
     */
    inline void addStepFunctionCallback(HostFunctionCallback *func_callback);
    /**
     * Adds an exit function callback to the simulation. The callback objects is similar to adding via addExitFunction
     * however the runnable function is encapsulated within an object which permits cross language support in swig.
     * Exit functions execute once after all simulation steps have completed or an exit conditions has returned EXIT
     * @param func_callback Pointer to the desired exit function callback
     * @throws exception::InvalidHostFunc If the exit function has already been added to this model description
     * @note Exit functions are executed in the order they were added to the model
     */
    inline void addExitFunctionCallback(HostFunctionCallback *func_callback);
#endif
    /**
     * Adds an exit condition function to the simulation
     * Exit conditions execute once per step, after all layers and step functions have been executed
     * If the condition returns false, the simulation exits early
     * @param func_p Pointer to the desired exit condition function
     * @throws exception::InvalidHostFunc If the exit condition has already been added to this model description
     * @note Exit conditions are the last functions to operate each step and can still make changes to the model
     * @note The step counter is updated after exit conditions have completed
     * @note Exit conditions are executed in the order they were added to the model
     */
    void addExitCondition(FLAMEGPU_EXIT_CONDITION_POINTER func_p);
#ifdef SWIG
    /**
     * Adds an exit condition callback to the simulation
     * Exit conditions execute once per step, after all layers and step functions have been executed
     * If the condition returns false, the simulation exits early
     * @param func_callback Pointer to the desired exit condition callback
     * @throws exception::InvalidHostFunc If the exit condition has already been added to this model description
     * @note Exit conditions are the last functions to operate each step and can still make changes to the model
     * @note The step counter is updated after exit conditions have completed
     * @note Exit conditions are executed in the order they were added to the model
     */
    inline void addExitConditionCallback(HostFunctionConditionCallback *func_callback);
#endif

    /**
     * @return The model's name
     */
    std::string getName() const;
    /**
     * @return A reference to the this model's DependencyGraph
     */
    const DependencyGraph& getDependencyGraph() const;
    /**
     * Sets root as an execution root of the model. Multiple roots can be used which represent independent chains of dependencies.
     * @param root The DependencyNode which will be used as an execution root
     */
    void addExecutionRoot(DependencyNode& root);
    /**
     * Generates layers from the dependency graph
     */
    void generateLayers();
    /**
     * Generates a .gv file containing the DOT representation of the dependencies specified
     * @param outputFileName The name of the output file
     */
    void generateDependencyGraphDOTDiagram(std::string outputFileName) const;
    /**
     * Returns a string representation of the constructed layers
     * @returns A string representation of the constructed layers
     */
    std::string getConstructedLayersString() const;
    /**
     * Returns an immutable reference to the specified agent, which can be used to view the agent's configuration
     * @param agent_name Name which can be used to the refer to the desired agent within the model description hierarchy
     * @return An immutable reference to the specified AgentDescription
     * @throws exception::InvalidAgentName If an agent with the name does not exist within the model description hierarchy
     * @see ModelDescription::Agent(const std::string &) for the mutable version
     */
    CAgentDescription getAgent(const std::string& agent_name) const;
    /**
     * Returns a mutable reference to the named message, which can be used to configure the message
     * @param message_name Name used to refer to the desired message within the model description hierarchy
     * @return An immutable reference to the specified MessageDescription
     * @throws exception::InvalidMessageName If a message with the name does not exist within the model description hierarchy
     * @see ModelDescription::Message(const std::string &) for the mutable version
     */
    template<typename MessageType>
    typename MessageType::CDescription getMessage(const std::string &message_name) const {
        auto rtn = model->messages.find(message_name);
        if (rtn != model->messages.end()) {
            if (auto r = std::dynamic_pointer_cast<typename MessageType::Data>(rtn->second)) {
                return MessageType::CDescription(r);
            }
            THROW exception::InvalidMessageType("Message ('%s') is not of correct type, "
                "in ModelDescription::getMessage().",
                message_name.c_str());
        }
        THROW exception::InvalidMessageName("Message ('%s') was not found, "
            "in ModelDescription::getMessage().",
            message_name.c_str());
    }
    MessageBruteForce::CDescription getMessage(const std::string &message_name) const;
    /**
     * Returns an immutable reference to the specified submodel, which can be used to view the submodel's configuration
     * @param submodel_name Name which can be used to the refer to the desired submodel within the model description hierarchy
     * @return An immutable reference to the specified SubModelDescription
     * @throws exception::InvalidSubModelName If a submodel with the name does not exist within the model description hierarchy
     * @see ModelDescription::SubModel(const std::string &) for the mutable version
     */
    CSubModelDescription getSubModel(const std::string &submodel_name) const;
    /**
     * Returns a mutable reference to the environment description for the model description hierarchy
     * This can be used to configure environment properties
     * @see ModelDescription::Environment() for the mutable version
     */
    CEnvironmentDescription getEnvironment() const;
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param name Name used to refer to the desired layer within the model description hierarchy
     * @return An immutable reference to the specified LayerDescription
     * @throws exception::InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::getLayer(const flamegpu::size_type &)
     * @see ModelDescription::Layer(const std::string &) for the mutable version
     */
    CLayerDescription getLayer(const std::string &name) const;
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param layer_index Index of the desired layer within the model description hierarchy
     * @return An immutable reference to the specified LayerDescription
     * @throws exception::InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::getLayer(const std::string &)
     * @see ModelDescription::Layer(const flamegpu::size_type &) for the mutable version
     */
    CLayerDescription getLayer(const flamegpu::size_type &layer_index) const;

    /**
     * @param agent_name Name of the agent to check
     * @return True when an agent with the specified name exists within the model's hierarchy
     */
    bool hasAgent(const std::string &agent_name) const;
    /**
     * @param message_name Name of the message to check
     * @return True when a message with the specified name exists within the model's hierarchy
     */
    template<typename MessageType>
    bool hasMessage(const std::string &message_name) const {
        auto a = model->messages.find(message_name);
        if (a != model->messages.end()) {
            if (std::dynamic_pointer_cast<typename MessageType::Data>(a->second))
                return true;
        }
        return false;
    }
    bool hasMessage(const std::string &message_name) const;
    /**
     * @param name Name of the layer to check
     * @return True when a layer with the specified name exists within the model's hierarchy
     */
    bool hasLayer(const std::string &name) const;
    /**
     * @param layer_index Index of the agent to check
     * @return True when a layer with the specified index exists within the model's hierarchy
     */
    bool hasLayer(const flamegpu::size_type &layer_index) const;
    /**
     * @param submodel_name Name of the submodel to check
     * @return True when a submodel with the specified name exists within the model's hierarchy
     */
    bool hasSubModel(const std::string &submodel_name) const;

    /**
     * @return The number of agents within the model's hierarchy
     */
    flamegpu::size_type getAgentsCount() const;
    /**
     * @return The number of messages within the model's hierarchy
     */
    flamegpu::size_type getMessagesCount() const;
    /**
     * @return The number of layers within the model's hierarchy
     */
    flamegpu::size_type getLayersCount() const;

 private:
    /**
     * The class which stores all of the model hierarchies data.
     */
     std::shared_ptr<ModelData> model;
};

#ifdef SWIG
void ModelDescription::addInitFunctionCallback(HostFunctionCallback* func_callback) {
    if (std::find(model->initFunctionCallbacks.begin(), model->initFunctionCallbacks.end(), func_callback) != model->initFunctionCallbacks.end()) {
            THROW exception::InvalidHostFunc("Attempted to add same init function callback twice,"
                "in ModelDescription::addInitFunctionCallback()");
    }
    model->initFunctionCallbacks.push_back(func_callback);
}
void ModelDescription::addStepFunctionCallback(HostFunctionCallback* func_callback) {
    if (std::find(model->stepFunctionCallbacks.begin(), model->stepFunctionCallbacks.end(), func_callback) != model->stepFunctionCallbacks.end()) {
            THROW exception::InvalidHostFunc("Attempted to add same step function callback twice,"
                "in ModelDescription::addStepFunctionCallback()");
    }
    model->stepFunctionCallbacks.push_back(func_callback);
}
void ModelDescription::addExitFunctionCallback(HostFunctionCallback* func_callback) {
    if (std::find(model->exitFunctionCallbacks.begin(), model->exitFunctionCallbacks.end(), func_callback) != model->exitFunctionCallbacks.end()) {
            THROW exception::InvalidHostFunc("Attempted to add same exit function callback twice,"
                "in ModelDescription::addExitFunctionCallback()");
    }
    model->exitFunctionCallbacks.push_back(func_callback);
}
void ModelDescription::addExitConditionCallback(HostFunctionConditionCallback *func_callback) {
    if (std::find(model->exitConditionCallbacks.begin(), model->exitConditionCallbacks.end(), func_callback) != model->exitConditionCallbacks.end()) {
            THROW exception::InvalidHostFunc("Attempted to add same exit condition callback twice,"
                "in ModelDescription::addExitConditionCallback()");
    }
    model->exitConditionCallbacks.push_back(func_callback);
}
#endif

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_

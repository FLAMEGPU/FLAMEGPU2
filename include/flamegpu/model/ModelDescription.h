#ifndef INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_

#include <map>
#include <memory>
#include <set>
#include <string>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/sim/Simulation.h"
#include "flamegpu/runtime/messaging/BruteForce.h"

class AgentDescription;
class LayerDescription;
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
    friend Simulation::Simulation(const ModelDescription& model);

 public:
    /**
     * Constructor
     * @param model_name Name of the model, this must be unique between models currently held by instances of CUDAAgentModel
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
     * @throws InvalidAgentName If an agent with the same name already exists within the model description hierarchy
     */
    AgentDescription& newAgent(const std::string &agent_name);
    /**
     * Returns a mutable reference to the named agent, which can be used to configure the agent
     * @param agent_name Name which can be used to the refer to the desired agent within the model description hierarchy
     * @return A mutable reference to the specified AgentDescription
     * @throws InvalidAgentName If an agent with the name does not exist within the model description hierarchy
     * @see ModelDescription::getAgent(const std::string &) for the immutable version
     */
    AgentDescription& Agent(const std::string &agent_name);

    /**
     * Creates a new message with the specified name
     * @param message_name Name which can be used to refer to the message within the model description hierarchy
     * @return A mutable reference to the new MessageDescription
     * @throws InvalidMessageName If a message with the same name already exists within the model description hierarchy
     */
    template<typename MsgType>
    typename MsgType::Description& newMessage(const std::string &message_name) {
        if (!hasMessage<MsgType>(message_name)) {
            auto rtn = std::shared_ptr<typename MsgType::Data>(new typename MsgType::Data(model, message_name));
            model->messages.emplace(message_name, rtn);
            return *reinterpret_cast<typename MsgType::Description*>(rtn->description.get());
        }
        THROW InvalidMessageName("Message with name '%s' already exists, "
            "in ModelDescription::newMessage().",
            message_name.c_str());
    }
    MsgBruteForce::Description& newMessage(const std::string &message_name);
    /**
     * Returns a mutable reference to the named message, which can be used to configure the message
     * @param message_name Name used to refer to the desired message within the model description hierarchy
     * @return A mutable reference to the specified MessageDescription
     * @throws InvalidMessageName If a message with the name does not exist within the model description hierarchy
     * @see ModelDescription::getMessage(const std::string &) for the immutable version
     */
    template<typename MsgType>
    typename MsgType::Description& Message(const std::string &message_name) {
        auto rtn = model->messages.find(message_name);
        if (rtn != model->messages.end()) {
            if (auto r = std::dynamic_pointer_cast<typename MsgType::Data>(rtn->second)) {
                return *reinterpret_cast<typename MsgType::Description*>(r->description.get());
            }
            THROW InvalidMessageName("Message ('%s') is not of correct type, "
                "in ModelDescription::Message().",
                message_name.c_str());
        }
        THROW InvalidMessageName("Message ('%s') was not found, "
            "in ModelDescription::Message().",
            message_name.c_str());
    }
    MsgBruteForce::Description& Message(const std::string &message_name);
    /**
     * Returns a mutable reference to the environment description for the model description hierarchy
     * This can be used to configure environment properties
     * @see ModelDescription::getEnvironment() for the immutable version
     */
    EnvironmentDescription& Environment();

    /**
     * Creates a new layer with the specified name
     * @param name Name which can be used to refer to the message within the model description hierarchy
     * @return A mutable reference to the new LayerDescription
     * @throws InvalidFuncLayerIndx If a layer with the same name already exists within the model description hierarchy
     * @note Layer names are not required, passing empty string will not set a name
     */
    LayerDescription& newLayer(const std::string &name = "");
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param name Name used to refer to the desired layer within the model description hierarchy
     * @return A mutable reference to the specified LayerDescription
     * @throws InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::Layer(const ModelData::size_type &)
     * @see ModelDescription::getLayer(const std::string &) for the immutable version
     */
    LayerDescription& Layer(const std::string &name);
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param layer_index Index of the desired layer within the model description hierarchy
     * @return A mutable reference to the specified LayerDescription
     * @throws InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::Layer(const std::string &)
     * @see ModelDescription::getLayer(const ModelData::size_type &) for the immutable version
     */
    LayerDescription& Layer(const ModelData::size_type &layer_index);

    /**
     * Adds an init function to the simulation
     * Init functions execute once before the simulation begins
     * @param func_p Pointer to the desired init function
     * @throws InvalidHostFunc If the init function has already been added to this model description
     */
    void addInitFunction(FLAMEGPU_INIT_FUNCTION_POINTER func_p);
    /**
     * Adds a step function to the simulation
     * Step functions execute once per step, after all layers have been executed, before exit conditions
     * @param func_p Pointer to the desired step function
     * @throws InvalidHostFunc If the step function has already been added to this model description
     */
    void addStepFunction(FLAMEGPU_STEP_FUNCTION_POINTER func_p);
    /**
     * Adds an exit function to the simulation
     * Exit functions execute once after the simulation ends
     * @param func_p Pointer to the desired exit function
     * @throws InvalidHostFunc If the exit function has already been added to this model description
     */
    void addExitFunction(FLAMEGPU_EXIT_FUNCTION_POINTER func_p);
    /**
     * Adds an exit condition function to the simulation
     * Exit conditions execute once per step, after all layers and step functions have been executed
     * If the condition returns false, the simulation exits early
     * @param func_p Pointer to the desired exit condition function
     * @throws InvalidHostFunc If the exit condition has already been added to this model description
     */
    void addExitCondition(FLAMEGPU_EXIT_CONDITION_POINTER func_p);

    /**
     * @return The model's name
     */
    std::string getName() const;
    /**
     * Returns an immutable reference to the specified agent, which can be used to view the agent's configuration
     * @param agent_name Name which can be used to the refer to the desired agent within the model description hierarchy
     * @return An immutable reference to the specified AgentDescription
     * @throws InvalidAgentName If an agent with the name does not exist within the model description hierarchy
     * @see ModelDescription::Agent(const std::string &) for the mutable version
     */
    const AgentDescription& getAgent(const std::string &agent_name) const;
    /**
     * Returns a mutable reference to the named message, which can be used to configure the message
     * @param message_name Name used to refer to the desired message within the model description hierarchy
     * @return An immutable reference to the specified MessageDescription
     * @throws InvalidMessageName If a message with the name does not exist within the model description hierarchy
     * @see ModelDescription::Message(const std::string &) for the mutable version
     */
    template<typename MsgType>
    const typename MsgType::Description& getMessage(const std::string &message_name) const {
        auto rtn = model->messages.find(message_name);
        if (rtn != model->messages.end()) {
            if (auto r = std::dynamic_pointer_cast<typename MsgType::Data>(rtn->second)) {
                return *reinterpret_cast<typename MsgType::Description*>(r->description.get());
            }
            THROW InvalidMessageName("Message ('%s') is not of correct type, "
                "in ModelDescription::getMessage().",
                message_name.c_str());
        }
        THROW InvalidMessageName("Message ('%s') was not found, "
            "in ModelDescription::getMessage().",
            message_name.c_str());
    }
    const MsgBruteForce::Description& getMessage(const std::string &message_name) const;
    /**
     * Returns a mutable reference to the environment description for the model description hierarchy
     * This can be used to configure environment properties
     * @see ModelDescription::Environment() for the mutable version
     */
    const EnvironmentDescription& getEnvironment() const;
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param name Name used to refer to the desired layer within the model description hierarchy
     * @return An immutable reference to the specified LayerDescription
     * @throws InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::getLayer(const ModelData::size_type &)
     * @see ModelDescription::Layer(const std::string &) for the mutable version
     */
    const LayerDescription& getLayer(const std::string &name) const;
    /**
     * Returns a mutable reference to the named layer, which can be used to configure the layer
     * @param layer_index Index of the desired layer within the model description hierarchy
     * @return An immutable reference to the specified LayerDescription
     * @throws InvalidFuncLayerIndx If a layer with the name does not exist within the model description hierarchy
     * @see ModelDescription::getLayer(const std::string &)
     * @see ModelDescription::Layer(const ModelData::size_type &) for the mutable version
     */
    const LayerDescription& getLayer(const ModelData::size_type &layer_index) const;

    /**
     * @param agent_name Name of the agent to check
     * @return True when an agent with the specified name exists within the model's hierarchy
     */
    bool hasAgent(const std::string &agent_name) const;
    /**
     * @param message_name Name of the message to check
     * @return True when a message with the specified name exists within the model's hierarchy
     */
    template<typename MsgType>
    bool hasMessage(const std::string &message_name) const {
        auto a = model->messages.find(message_name);
        if (a != model->messages.end()) {
            if (std::dynamic_pointer_cast<typename MsgType::Data>(a->second))
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
    bool hasLayer(const ModelData::size_type &layer_index) const;

    /**
     * @return The number of agents within the model's hierarchy
     */
    ModelData::size_type getAgentsCount() const;
    /**
     * @return The number of messages within the model's hierarchy
     */
    ModelData::size_type getMessagesCount() const;
    /**
     * @return The number of layers within the model's hierarchy
     */
    ModelData::size_type getLayersCount() const;

 private:
    /**
     * The class which stores all of the model hierarchies data.
     */
     ModelData *const model;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_

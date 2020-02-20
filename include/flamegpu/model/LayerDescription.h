#ifndef INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_

#include <string>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"

struct ModelData;
struct LayerData;

/**
 * Within the model hierarchy, this class represents an execution layer for a FLAMEGPU model
 * This class is used to configure which agent and host functions execute at a stage of the model's execution
 * @see ModelDescription::newLayer(const std::string&) For creating instances of this class
 */
class LayerDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct LayerData;
    /**
    * Constructors
    */
    LayerDescription(ModelData *const _model, LayerData *const data);
    /**
     * Default copy constructor, not implemented
     */
    LayerDescription(const LayerDescription &other_layer) = delete;
    /**
     * Default move constructor, not implemented
     */
    LayerDescription(LayerDescription &&other_layer) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    LayerDescription& operator=(const LayerDescription &other_layer) = delete;
    /**
     * Default move assignment, not implemented
     */
    LayerDescription& operator=(LayerDescription &&other_layer) noexcept = delete;

 public:
    /**
     * Equality operator, checks whether LayerDescription hierarchies are functionally the same
     * @returns True when layers are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const LayerDescription& rhs) const;
    /**
     * Equality operator, checks whether LayerDescription hierarchies are functionally different
     * @returns True when layers are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const LayerDescription& rhs) const;

    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param a Agent function as defined using FLAMEGPU_AGENT_FUNCTION notation
     * @tparam AgentFunction Struct containing agent function definition
     * @throw InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw InvalidAgentFunc If the agent function has already been added to the layer
     * @note The agent function must first be added to an Agent
     * @see AgentDescription::newFunction(const std::string &, AgentFunction)
     */
    template<typename AgentFunction>
    void addAgentFunction(AgentFunction a = AgentFunction());
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param afd Agent function description to execute during this layer
     * @throw InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw InvalidAgentFunc If the agent function has already been added to the layer
     */
    void addAgentFunction(const AgentFunctionDescription &afd);
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param name Name of the agent function description to execute during this layer
     * @throw InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw InvalidAgentFunc If the agent function has already been added to the layer
     */
    void addAgentFunction(const std::string &name);
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param name Name of the agent function description to execute during this layer
     * @throw InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw InvalidAgentFunc If the agent function has already been added to the layer
     * @note This version exists because the template overload was preventing implicit cast to std::string
     */
    void addAgentFunction(const char *name);
    /**
     * Adds a host function to this layer
     * The host function will be called during this stage of model execution
     * @param func_p Function pointer to the host function declared using FLAMEGPU_HOST_FUNCTION notation
     * @throw InvalidHostFunc If the function has already been added to the layer
     * @note This version exists because the template overload was preventing implicit cast to std::string
     */
    void addHostFunction(FLAMEGPU_HOST_FUNCTION_POINTER func_p);

    /**
     * @return The layer's name
     */
    std::string getName() const;
    /**
     * @return The index of the layer within the model's execution
     */
    ModelData::size_type getIndex() const;
    /**
     * @return The total number of agent functions within the layer
     */
    ModelData::size_type getAgentFunctionsCount() const;
    /**
     * @return The total number of host functions within the layer
     */
    ModelData::size_type getHostFunctionsCount() const;

    /**
     * @param index Index of the function to return
     * @return An immutable reference to the agent function at the provided index
     * @throw OutOfBoundsException When index exceeds number of agent functions in the layer
     * @see LayerDescription::getAgentFunctionsCount()
     * @note Functions are stored in a set, so order may change as new functions are added
     */
    const AgentFunctionDescription &getAgentFunction(unsigned int index) const;
    /**
     * @param index Index of the function to return
     * @return A function pointer to the host function at the provided index
     * @throw OutOfBoundsException When index exceeds number of host functions in the layer
     * @see LayerDescription::getAgentFunctionsCount()
     * @note Functions are stored in a set, so order may change as new functions are added
     */
    FLAMEGPU_HOST_FUNCTION_POINTER getHostFunction(unsigned int index) const;

 private:
    /**
     * Root of the model hierarchy
     */
    ModelData *const model;
    /**
     * The class which stores all of the layer's data.
     */
    LayerData *const layer;
};


template<typename AgentFunction>
void LayerDescription::addAgentFunction(AgentFunction /*af*/) {
    AgentFunctionWrapper * func_compare = AgentFunction::fnPtr();
    // Find the matching agent function in model hierarchy
    for (auto a : model->agents) {
        for (auto f : a.second->functions) {
            if (f.second->func == func_compare) {
                // Check that layer does not already contain function with same agent + states
                for (const auto &b : layer->agent_functions) {
                    if (auto parent = b->parent.lock()) {
                        // If agent matches
                        if (parent->name == a.second->name) {
                            // If they share a state
                            if (b->initial_state == f.second->initial_state ||
                                b->initial_state == f.second->end_state ||
                                b->end_state == f.second->initial_state ||
                                b->end_state == f.second->end_state) {
                                THROW InvalidAgentFunc("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                                    "within the layer shares an input or output state, this is not permitted, "
                                    "in LayerDescription::addAgentFunction()\n",
                                    f.second->name.c_str(), b->name.c_str());
                            }
                        }
                    }
                }
                // Add it and check it succeeded
                if (layer->agent_functions.emplace(f.second).second)
                    return;
                THROW InvalidAgentFunc("Attempted to add same agent function to same layer twice, "
                    "in LayerDescription::addAgentFunction()\n");
            }
        }
    }
    THROW InvalidAgentFunc("Agent function was not found, "
        "in LayerDescription::addAgentFunction()\n");
}

#endif  // INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_

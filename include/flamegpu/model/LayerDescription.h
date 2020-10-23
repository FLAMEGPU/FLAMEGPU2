#ifndef INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_

#include <string>
#include <memory>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/runtime/AgentFunction.h"
#include "flamegpu/model/AgentFunctionData.h"

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
    LayerDescription(const std::shared_ptr<const ModelData> &_model, LayerData *const data);
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
     * @throw InvalidLayerMember If the layer already contains a SubModel
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
     * @throw InvalidLayerMember If the layer already contains a SubModel
     */
    void addAgentFunction(const AgentFunctionDescription &afd);
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param agentName Name of the agent which owns the function to execute during this layer
     * @param functionName Name of the agent function description to execute during this layer
     * @throw InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw InvalidAgentFunc If the agent function has already been added to the layer
     * @throw InvalidLayerMember If the layer already contains a SubModel
     */
    void addAgentFunction(const std::string &agentName, const std::string &functionName);
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param agentName Name of the agent which owns the function to execute during this layer
     * @param functionName Name of the agent function description to execute during this layer
     * @throw InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw InvalidAgentFunc If the agent function has already been added to the layer
     * @throw InvalidLayerMember If the layer already contains a SubModel
     * @note This version exists because the template overload was preventing implicit cast to std::string
     */
    void addAgentFunction(const char *agentName, const char *functionName);
    /**
     * Adds a host function to this layer
     * The host function will be called during this stage of model execution
     * @param func_p Function pointer to the host function declared using FLAMEGPU_HOST_FUNCTION notation
     * @throw InvalidHostFunc If the function has already been added to the layer
     * @throw InvalidLayerMember If the layer already contains a SubModel
     * @note There is no guarantee on the order in which multiple host functions in the same layer will be executed
     */
    void addHostFunction(FLAMEGPU_HOST_FUNCTION_POINTER func_p);
    /**
     * Adds a submodel to a layer
     * If layer contains a submodel, it may contain nothing else
     * @param name Name of the submodel (passed to ModelDescription::newSubModel() was called)
     * @throw InvalidLayerMember If the layer already contains any agent functions or host functions
     * @throw InvalidSubModel If the layer already contains a submodel
     * @see addSubModel(const SubModelDescription &)
     */
    void addSubModel(const std::string &name);
    /**
     * Adds a submodel to a layer
     * If layer contains a submodel, it may contain nothing else
     * @param submodel SubModel description of the layer to be bound
     * @throw InvalidLayerMember If the layer already contains any agent functions or host functions
     * @throw InvalidSubModel If the layer already contains a submodel
     * @see addSubModel(const std::string &)
     */
    void addSubModel(const SubModelDescription &submodel);
#ifdef SWIG
    /**
     * Adds a host function to this layer, similar to addHostFunction
     * however the runnable function is encapsulated within an object which permits cross language support in swig.
     * The host function will be called during this stage of model execution
     * @param func_callback a Host function callback object
     * @throw InvalidHostFunc If the function has already been added to the layer
     */
    inline void addHostFunctionCallback(HostFunctionCallback *func_callback);
#endif
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
#ifdef SWIG
    /**
     * @return The total number of host function callbacks within the layer
     */
    inline ModelData::size_type getHostFunctionCallbackCount() const;
#endif

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
     * @see LayerDescription::getHostFunctionsCount()
     * @note Functions are stored in a set, so order may change as new functions are added
     */
    FLAMEGPU_HOST_FUNCTION_POINTER getHostFunction(unsigned int index) const;
#ifdef SWIG
    /**
     * @param index Index of the function to return
     * @return A function callback to the host function at the provided index
     * @throw OutOfBoundsException When index exceeds number of host functions in the layer
     * @see LayerDescription::getHostFunctionCallbackCount()
     * @note Functions are stored in a set, so order may change as new functions are added
     */
    inline HostFunctionCallback* getHostFunctionCallback(unsigned int index) const;
#endif

 private:
    /**
     * Root of the model hierarchy
     */
    std::weak_ptr<const ModelData> model;
    /**
     * The class which stores all of the layer's data.
     */
    LayerData *const layer;
};


template<typename AgentFunction>
void LayerDescription::addAgentFunction(AgentFunction /*af*/) {
    AgentFunctionWrapper * func_compare = AgentFunction::fnPtr();
    // Find the matching agent function in model hierarchy
    auto mdl = model.lock();
    if (!mdl) {
        THROW ExpiredWeakPtr();
    }
    for (auto a : mdl->agents) {
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
                                    "in LayerDescription::addAgentFunction().",
                                    f.second->name.c_str(), b->name.c_str());
                            }
                        }
                    }
                }
                // Add it and check it succeeded
                if (layer->agent_functions.emplace(f.second).second)
                    return;
                THROW InvalidAgentFunc("Attempted to add same agent function to same layer twice, "
                    "in LayerDescription::addAgentFunction().");
            }
        }
    }
    THROW InvalidAgentFunc("Agent function was not found, "
        "in LayerDescription::addAgentFunction().");
}

#ifdef SWIG
void LayerDescription::addHostFunctionCallback(HostFunctionCallback* func_callback) {
    if (!layer->host_functions_callbacks.insert(func_callback).second) {
            THROW InvalidHostFunc("Attempted to add same host function callback twice,"
                "in LayerDescription::addHostFunctionCallback()");
        }
}
ModelData::size_type LayerDescription::getHostFunctionCallbackCount() const {
    // Safe down-cast
    return static_cast<ModelData::size_type>(layer->host_functions_callbacks.size());
}
HostFunctionCallback* LayerDescription::getHostFunctionCallback(unsigned int index) const {
    if (index < layer->host_functions_callbacks.size()) {
        auto it = layer->host_functions_callbacks.begin();
        for (unsigned int i = 0; i < index; ++i)
            ++it;
        return *it;
    }
    THROW OutOfBoundsException("Index %d is out of bounds (only %d items exist) "
        "in LayerDescription.getHostFunctionCallback()\n",
        index, layer->host_functions_callbacks.size());
}
#endif

#endif  // INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_

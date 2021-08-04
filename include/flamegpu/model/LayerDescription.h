#ifndef INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_

#include <string>
#include <memory>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/runtime/AgentFunction.cuh"
#include "flamegpu/model/AgentFunctionData.cuh"

namespace flamegpu {

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
    friend class DependencyGraph;
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
     * @throw exception::InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw exception::InvalidAgentFunc If the agent function has already been added to the layer
     * @throw exception::InvalidLayerMember If the layer already contains a SubModel
     * @throw exception::InvalidLayerMember If the agent function outputs to a message list output to by an existing agent function of the layer
     * @note The agent function must first be added to an Agent
     * @see AgentDescription::newFunction(const std::string &, AgentFunction)
     */
    template<typename AgentFunction>
    void addAgentFunction(AgentFunction a = AgentFunction());
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param afd Agent function description to execute during this layer
     * @throw exception::InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw exception::InvalidAgentFunc If the agent function has already been added to the layer
     * @throw exception::InvalidLayerMember If the layer already contains a SubModel
     * @throw exception::InvalidLayerMember If the agent function outputs to a message list output to by an existing agent function of the layer
     * @throw exception::InvalidLayerMember If the agent function outputs an agent in the same agent state as an existing agent function's input state (or vice versa)
     */
    void addAgentFunction(const AgentFunctionDescription &afd);
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param agentName Name of the agent which owns the function to execute during this layer
     * @param functionName Name of the agent function description to execute during this layer
     * @throw exception::InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw exception::InvalidAgentFunc If the agent function has already been added to the layer
     * @throw exception::InvalidLayerMember If the layer already contains a SubModel
     * @throw exception::InvalidLayerMember If the agent function outputs to a message list output to by an existing agent function of the layer
     * @throw exception::InvalidLayerMember If the agent function outputs an agent in the same agent state as an existing agent function's input state (or vice versa)
     */
    void addAgentFunction(const std::string &agentName, const std::string &functionName);
    /**
     * Adds an agent function to this layer
     * The agent function will be called during this stage of model execution
     * @param agentName Name of the agent which owns the function to execute during this layer
     * @param functionName Name of the agent function description to execute during this layer
     * @throw exception::InvalidAgentFunc If the agent function does not exist within the model hierarchy
     * @throw exception::InvalidAgentFunc If the agent function has already been added to the layer
     * @throw exception::InvalidLayerMember If the layer already contains a SubModel
     * @throw exception::InvalidLayerMember If the agent function outputs to a message list output to by an existing agent function of the layer
     * @throw exception::InvalidLayerMember If the agent function outputs an agent in the same agent state as an existing agent function's input state (or vice versa)
     * @note This version exists because the template overload was preventing implicit cast to std::string
     */
    void addAgentFunction(const char *agentName, const char *functionName);
    /**
     * Adds a host function to this layer
     * The host function will be called during this stage of model execution
     * @param func_p Function pointer to the host function declared using FLAMEGPU_HOST_FUNCTION notation
     * @throw exception::InvalidHostFunc If the function has already been added to the layer
     * @throw exception::InvalidLayerMember If the layer already contains a SubModel
     * @note There is no guarantee on the order in which multiple host functions in the same layer will be executed
     */
    void addHostFunction(FLAMEGPU_HOST_FUNCTION_POINTER func_p);
    /**
     * Adds a submodel to a layer
     * If layer contains a submodel, it may contain nothing else
     * @param name Name of the submodel (passed to ModelDescription::newSubModel() was called)
     * @throw exception::InvalidLayerMember If the layer already contains any agent functions or host functions
     * @throw exception::InvalidSubModel If the layer already contains a submodel
     * @see addSubModel(const SubModelDescription &)
     */
    void addSubModel(const std::string &name);
    /**
     * Adds a submodel to a layer
     * If layer contains a submodel, it may contain nothing else
     * @param submodel SubModel description of the layer to be bound
     * @throw exception::InvalidLayerMember If the layer already contains any agent functions or host functions
     * @throw exception::InvalidSubModel If the layer already contains a submodel
     * @see addSubModel(const std::string &)
     */
    void addSubModel(const SubModelDescription &submodel);
    /**
     * Adds a host function to this layer, similar to addHostFunction
     * however the runnable function is encapsulated within an object which permits cross language support in swig.
     * The host function will be called during this stage of model execution
     * @param func_callback a Host function callback object
     * @throw exception::InvalidHostFunc If the function has already been added to the layer
     * @note ONLY USED INTERNALLY AND BY PYTHON API - DO NOT CALL IN C++ BUILD
     */
    void _addHostFunctionCallback(HostFunctionCallback *func_callback);

 public:
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
    /**
     * Adds a host function to this layer, similar to addHostFunction
     * however the runnable function is encapsulated within an object which permits cross language support in swig.
     * The host function will be called during this stage of model execution
     * @param func_callback a Host function callback object
     * @throw exception::InvalidHostFunc If the function has already been added to the layer
     * @note ONLY USED INTERNALLY AND BY PYTHON API - DO NOT CALL IN C++ BUILD
     */
    inline void addHostFunctionCallback(HostFunctionCallback *func_callback);
#endif

    /**
     * @param index Index of the function to return
     * @return An immutable reference to the agent function at the provided index
     * @throw exception::OutOfBoundsException When index exceeds number of agent functions in the layer
     * @see LayerDescription::getAgentFunctionsCount()
     * @note Functions are stored in a set, so order may change as new functions are added
     */
    const AgentFunctionDescription &getAgentFunction(unsigned int index) const;
    /**
     * @param index Index of the function to return
     * @return A function pointer to the host function at the provided index
     * @throw exception::OutOfBoundsException When index exceeds number of host functions in the layer
     * @see LayerDescription::getHostFunctionsCount()
     * @note Functions are stored in a set, so order may change as new functions are added
     */
    FLAMEGPU_HOST_FUNCTION_POINTER getHostFunction(unsigned int index) const;
#ifdef SWIG
    /**
     * @param index Index of the function to return
     * @return A function callback to the host function at the provided index
     * @throw exception::OutOfBoundsException When index exceeds number of host functions in the layer
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
    if (layer->sub_model) {
        THROW exception::InvalidLayerMember("A layer containing agent functions and/or host functions, may not also contain a submodel, "
            "in LayerDescription::addAgentFunction()\n");
    }
    if (layer->host_functions.size() || layer->host_functions_callbacks.size()) {
        THROW exception::InvalidLayerMember("A layer containing host functions, may not also contain agent functions, "
            "in LayerDescription::addAgentFunction()\n");
    }
    AgentFunctionWrapper * func_compare = AgentFunction::fnPtr();
    // Find the matching agent function in model hierarchy
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    unsigned int matches = 0;
    std::shared_ptr<AgentFunctionData> match_ptr;
    for (auto a : mdl->agents) {
        for (auto f : a.second->functions) {
            if (f.second->func == func_compare) {
                auto a_agent_out = f.second->agent_output.lock();
                auto a_message_out = f.second->message_output.lock();
                auto a_message_in = f.second->message_input.lock();
                for (const auto &b : layer->agent_functions) {
                    if (auto parent = b->parent.lock()) {
                        // Check that layer does not already contain function with same agent + states
                        // If agent matches
                        if (parent->name == a.second->name) {
                            // If they share a state
                            if (b->initial_state == f.second->initial_state ||
                                b->initial_state == f.second->end_state ||
                                b->end_state == f.second->initial_state ||
                                b->end_state == f.second->end_state) {
                                THROW exception::InvalidAgentFunc("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                                    "within the layer shares an input or output state, this is not permitted, "
                                    "in LayerDescription::addAgentFunction().",
                                    f.second->name.c_str(), b->name.c_str());
                            }
                        }
                        // Check that the layer does not already contain function for the agent + state being output to
                        if (a_agent_out) {
                            // If agent matches
                            if (parent->name == a_agent_out->name) {
                                // If state matches
                                if (b->initial_state == f.second->agent_output_state) {
                                    THROW exception::InvalidLayerMember("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                                        "within the layer requires the same agent state as an input, as this agent function births, "
                                        "in LayerDescription::addAgentFunction().",
                                        f.second->name.c_str(), b->name.c_str());
                                }
                            }
                        }
                        // Also check the inverse
                        auto b_agent_out = b->agent_output.lock();
                        if (b_agent_out) {
                            // If agent matches
                            if (a.second->name == b_agent_out->name) {
                                // If state matches
                                if (f.second->initial_state == b->agent_output_state) {
                                    THROW exception::InvalidLayerMember("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                                        "within the layer agent births to the same agent state as this agent function requires as an input, "
                                        "in LayerDescription::addAgentFunction().",
                                        f.second->name.c_str(), b->name.c_str());
                                }
                            }
                        }
                    }
                    // Check the layer does not already contain function which outputs to same message list
                    auto b_message_out = b->message_output.lock();
                    auto b_message_in = b->message_input.lock();
                    if ((a_message_out && b_message_out && a_message_out == b_message_out) ||
                        (a_message_out && b_message_in && a_message_out == b_message_in) ||
                        (a_message_in && b_message_out && a_message_in == b_message_out)) {  // Pointer comparison should be fine here
                        THROW exception::InvalidLayerMember("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                            "within the layer also inputs or outputs to the same messagelist, this is not permitted, "
                            "in LayerDescription::addAgentFunction().",
                            f.second->name.c_str(), b->name.c_str());
                    }
                }
                match_ptr = f.second;
                ++matches;
            }
        }
    }
    if (matches == 1) {
        // Add it and check it succeeded
        if (layer->agent_functions.emplace(match_ptr).second)
            return;
        THROW exception::InvalidAgentFunc("Attempted to add same agent function to same layer twice, "
            "in LayerDescription::addAgentFunction().");
    }
    if (matches > 1) {
        THROW exception::InvalidAgentFunc("There are %u possible agent functions to add to layer, please use a more specific method for adding this agent function to a layer, "
            "in LayerDescription::addAgentFunction().", matches);
    }
    THROW exception::InvalidAgentFunc("Agent function was not found, "
        "in LayerDescription::addAgentFunction().");
}

#ifdef SWIG
void LayerDescription::addHostFunctionCallback(HostFunctionCallback* func_callback) {
    this->_addHostFunctionCallback(func_callback);
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
    THROW exception::OutOfBoundsException("Index %d is out of bounds (only %d items exist) "
        "in LayerDescription.getHostFunctionCallback()\n",
        index, layer->host_functions_callbacks.size());
}
#endif  // SWIG

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_LAYERDESCRIPTION_H_

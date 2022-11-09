#ifndef INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_

#include <unordered_map>
#include <list>
#include <memory>
#include <typeindex>
#include <vector>
#include <string>

#include "flamegpu/defines.h"
#include "flamegpu/runtime/HostAPI_macros.h"
#include "flamegpu/runtime/messaging/MessageBruteForce.h"

namespace flamegpu {

class HostFunctionCallback;
class HostFunctionConditionCallback;
class DependencyGraph;
struct EnvironmentData;
struct AgentData;
struct LayerData;
struct SubModelData;

/**
 * This is the internal data store for ModelDescription
 * Users should only access that data stored within via an instance of ModelDescription
 */
struct ModelData : std::enable_shared_from_this<ModelData>{
    virtual ~ModelData() = default;
    /**
     * Default state, all agents and agent functions begin in/with this state
     */
    static const char *DEFAULT_STATE;  // "default"
    /**
     * Description needs full access
     */
    friend class ModelDescription;
    /**
     * Map of name:agent definition
     * map<string, AgentData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<AgentData>> AgentMap;
    /**
     * Map of name:message definition
     * map<string, MessageData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<MessageBruteForce::Data>> MessageMap;
    /**
     * Map of name:message definition
     * map<string, MessageData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<SubModelData>> SubModelMap;
    /**
     * List of layer definitions
     * list<LayerData>
     */
    typedef std::list<std::shared_ptr<LayerData>> LayerList;
    /**
     * Vector of Init function pointers.
     * Uses a vector rather than a set to preserve order.
     */
    typedef std::vector<FLAMEGPU_INIT_FUNCTION_POINTER> InitFunctionVector;
    /**
     * Vector of Step function pointers.
     * Uses a vector rather than a set to preserve order.
     */
    typedef std::vector<FLAMEGPU_STEP_FUNCTION_POINTER> StepFunctionVector;
    /**
     * Vector of Step function callback pointers.
     * Uses a vector rather than a set to preserve order.
     */
    typedef std::vector<HostFunctionCallback*> HostFunctionCallbackVector;
    /**
     * Vector of host condition callback pointers.
     * Uses a vector rather than a set to preserve order.
     */
    typedef std::vector<HostFunctionConditionCallback*> HostFunctionConditionCallbackVector;
    /**
     * Vector of Exit function pointers.
     * Uses a vector rather than a set to preserve order.
     */
    typedef std::vector<FLAMEGPU_EXIT_FUNCTION_POINTER> ExitFunctionVector;
    /**
     * Vector of Exit condition pointers.
     * Uses a vector rather than a set to preserve order.
     */
    typedef std::vector<FLAMEGPU_EXIT_CONDITION_POINTER> ExitConditionVector;

    /**
     * Holds all of the model's agent definitions
     */
    AgentMap agents;
    /**
     * Holds all of the model's message definitions
     */
    MessageMap messages;
    /**
     * Holds all of the model's sub models
     */
    SubModelMap submodels;
    /**
     * Holds all of the model's layer definitions
     */
    LayerList layers;
    /**
     * Holds pointers to all of the init functions used by the model
     */
    InitFunctionVector initFunctions;
    HostFunctionCallbackVector initFunctionCallbacks;
    /**
     * Holds pointers to all of the step functions used by the model
     */
    StepFunctionVector stepFunctions;
    HostFunctionCallbackVector stepFunctionCallbacks;
    /**
     * Holds pointers to all of the exit functions used by the model
     */
    ExitFunctionVector exitFunctions;
    HostFunctionCallbackVector exitFunctionCallbacks;
    /**
     * Holds pointers to all of the exit conditions used by the model
     */
    ExitConditionVector exitConditions;
    HostFunctionConditionCallbackVector exitConditionCallbacks;
    /**
     * Holds all of the model's environment property definitions
     */
    std::shared_ptr<EnvironmentData> environment;
    /**
     * The name of the model
     * This must be unique among Simulation (e.g. CUDASimulation) instances
     */
    std::string name;
    /**
     * The dependency graph representing the dependencies of agent functions, submodels and host functions for this model.
     */
    std::shared_ptr<DependencyGraph> dependencyGraph;
    /**
     * Creates a copy of the entire model definition hierarchy
     * This is called when a ModelDescription is passed to a Simulation (e.g. CUDASimulation)
     */
    std::shared_ptr<ModelData> clone() const;
    /**
     * Equality operator, checks whether ModelData hierarchies are functionally the same
     * @returns True when models are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const ModelData& rhs) const;
    /**
     * Equality operator, checks whether ModelData hierarchies are functionally different
     * @returns True when models are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const ModelData& rhs) const;

    /**
     * @return The maximum layer width within the model's hierarchy
     */
    flamegpu::size_type getMaxLayerWidth() const;

 protected:
    friend SubModelData;  // Uses private copy constructor
    /**
     * Copy constructor
     * This should only be called via clone();
     */
    explicit ModelData(const ModelData &other);
    /**
     * Normal constructor
     * This should only be called by ModelDescription
     */
    explicit ModelData(const std::string &model_name);

 private:
    /**
     * @param submodel_data The submodel's data to check
     * @return True when a submodel with the specified name exists within the model's hierarchy
     */
    bool hasSubModelRecursive(const std::shared_ptr<const ModelData> &submodel_data) const;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_

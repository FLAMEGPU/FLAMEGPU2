
#ifndef INCLUDE_FLAMEGPU_MODEL_LAYERDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_LAYERDATA_H_

#include <set>
#include <memory>
#include <string>

#include "flamegpu/runtime/HostAPI_macros.h"  // Todo replace with std/cub style fns (see AgentFunction.cuh)
#include "flamegpu/model/ModelData.h"

namespace flamegpu {

struct AgentFunctionData;
class LayerDescription;
class HostFunctionCallback;
class HostFunctionConditionCallback;

/**
 * This is the internal data store for LayerDescription
 * Users should only access that data stored within via an instance of LayerDescription
 */
struct LayerData {
    friend class ModelDescription;
    friend struct ModelData;
    /**
     * Parent model
     */
    std::weak_ptr<const ModelData> model;
    /**
     * Set of Agent Functions
     * set<AgentFunctionData>
     */
    std::set<std::shared_ptr<AgentFunctionData>> agent_functions;
    /**
     * Set of host function pointers
     * set<FLAMEGPU_HOST_FUNCTION_POINTER>
     */
    std::set<FLAMEGPU_HOST_FUNCTION_POINTER> host_functions;
    /**
     * Set of host function callbacks (used by SWIG interface)
     * set<HostFunctionCallback*>
     */
    std::set<HostFunctionCallback*> host_functions_callbacks;
    /**
     * SubModel
     * (If present, layer can hold no host_functions or agent_functions)
     */
    std::shared_ptr<SubModelData> sub_model;
    /**
     * Name of the agent function, used to refer to the agent function in many functions
     */
    std::string name;
    /**
     * Index of the layer in the stack
     * (Eventually this will be replaced when we move to a more durable mode of layers, e.g. dependency analysis)
     */
    flamegpu::size_type index;
    /**
     * Equality operator, checks whether LayerData hierarchies are functionally the same
     * @returns True when layers are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const LayerData &rhs) const;
    /**
     * Equality operator, checks whether LayerData hierarchies are functionally different
     * @returns True when layers are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const LayerData &rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    LayerData(const LayerData &other) = delete;

 protected:
    friend DependencyGraph;  // Uses normal constructor to directly create layers in ModelData - avoids requiring ModelDescription in DependencyGraph
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     */
    LayerData(const std::shared_ptr<const ModelData> &model, const LayerData &other);
    /**
     * Normal constructor, only to be called by ModelDescription
     */
    LayerData(const std::shared_ptr<const ModelData> &model, const std::string &name, const flamegpu::size_type &index);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_LAYERDATA_H_

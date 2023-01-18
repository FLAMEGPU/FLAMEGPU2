#ifndef INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDIRECTEDGRAPHDATA_CUH_
#define INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDIRECTEDGRAPHDATA_CUH_

#include <string>
#include <memory>

#include "flamegpu/model/Variable.h"
#include "flamegpu/model/ModelData.h"

namespace flamegpu {

class EnvironmentDescription;

/**
 * This is the internal data store for EnvironmentDirectedGraphDescription
 * Users should only access that data stored within via an instance of EnvironmentDirectedGraphDescription
 */
struct EnvironmentDirectedGraphData {
    friend class EnvironmentDescription;
    friend struct EnvironmentData;
    /**
     * Parent model
     */
    std::weak_ptr<const ModelData> model;
    /**
     * Holds all of the graphs's vertex property definitions
     */
    VariableMap vertexProperties{};
    /**
     * Holds all of the graphs's edge property definitions
     */
    VariableMap edgeProperties{};
    /**
     * Name of the graph, used to refer to the graph in many functions
     */
    std::string name;
    /**
     * Equality operator, checks whether EnvironmentDirectedGraphData hierarchies are functionally the same
     * @param rhs Right hand side
     * @returns True when directed graphs are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const EnvironmentDirectedGraphData& rhs) const;
    /**
     * Equality operator, checks whether EnvironmentDirectedGraphData hierarchies are functionally different
     * @param rhs Right hand side
     * @returns True when directed graphs are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const EnvironmentDirectedGraphData& rhs) const;
    /**
     * Default copy constructor should not be used
     */
    explicit EnvironmentDirectedGraphData(const EnvironmentDirectedGraphData& other) = delete;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     * @param model The parent model of the graph
     * @param other Other EnvironmentDirectedGraphData to copy data from
     */
    explicit EnvironmentDirectedGraphData(const std::shared_ptr<const ModelData>& model, const EnvironmentDirectedGraphData& other);
    /**
     * Normal constructor, only to be called by ModelDescription
     * @param parent The environment which owns the graph
     * @param graph_name Name of the graph
     */
    explicit EnvironmentDirectedGraphData(const std::shared_ptr<const EnvironmentData>& parent, const std::string& graph_name);
};
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDIRECTEDGRAPHDATA_CUH_

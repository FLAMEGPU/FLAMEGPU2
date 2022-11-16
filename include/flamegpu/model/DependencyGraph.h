#ifndef INCLUDE_FLAMEGPU_MODEL_DEPENDENCYGRAPH_H_
#define INCLUDE_FLAMEGPU_MODEL_DEPENDENCYGRAPH_H_

#include <functional>
#include <string>
#include <vector>

#include "DependencyNode.h"
#include "ModelDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/HostFunctionDescription.h"
#include "flamegpu/model/SubModelDescription.h"

namespace flamegpu {

/**
 * This class represents the dependency tree for agent functions, host functions and submodels. Each DependencyNode has its own dependencies/dependents, the purpose of this class is to
 * walk the dependency tree and provide utility such as validation/layering.
 * @see DependencyNode
 */

class DependencyGraph {
 public:
    /**
     * Constructors
     */
    /**
     * Used by ModelData for instantiation
     */
    explicit DependencyGraph(ModelData* _model);
    /**
     * Deep copy
     */
    explicit DependencyGraph(const DependencyGraph& other);
    /**
     * Equality operator, checks whether DependencyGraphs are functionally the same, i.e, do they represent the same execution graph
     * @returns True when both graphs represent the same execution graph
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const DependencyGraph& rhs);
    /**
     * Add an AgentFunctionDescription, host function or submodel as a root node
     * @param root The function or submodel to add to the graph as a root
     */
    void addRoot(DependencyNode& root);
    /**
     * Generates optimal layers based on the dependencies specified and adds them to the model
     * @param model The model the layers should be added to
     * @throws exception::InvalidDependencyGraph if the model already has layers attached
     */
    void generateLayers();
    /**
     * Generates a .gv file containing the DOT representation of the dependencies specified
     * @param outputFileName The name of the output file
     */
    void generateDOTDiagram(std::string outputFileName) const;
    /**
     * Checks the dependency graph for cycles and validates that all agent functions belong to the same model
     * @returns True when the graph is valid, i.e. it contains no cycles
     */
    bool validateDependencyGraph() const;
    /**
     * Returns a string representation of the constructed layers
     * @returns A string representation of the constructed layers
     */
    std::string getConstructedLayersString() const;

 private:
    /**
     * Roots of the dependency tree, i.e. functions/submodels which have no dependencies and should execute first
     */
    std::vector<DependencyNode*> roots;
    /**
     * Used internally for cycle checking
     */
    bool doesFunctionExistInStack(DependencyNode* function, std::vector<DependencyNode*>& functionStack) const;
    /**
     * Check the subtree from a node is valid
     * @param node the root of the subtree
     * @param functionStack functions are already present in the tree
     * @returns True if the subtree is valid
     */
    bool validateSubTree(DependencyNode* node, std::vector<DependencyNode*>& functionStack) const;
    /**
     * Issues a warning if the graph is missing agent functions which are present in the model this dependency graph is attached to
     */
    void checkForUnattachedFunctions();
    /**
     * Generates a new layer in the model the dependency graph is associated with
     */
    LayerDescription newModelLayer();
    /**
     * Returns the name of a given DependencyNode
     * @param node The node to get the name of
     * @returns std::string The name of the node
     */
    static std::string getNodeName(DependencyNode* node);
    /**
     * Structured representation of the layers added to the model
     */
    std::vector<std::vector<std::string>> constructedLayers;
    /**
     * Root of the model hierarchy, used for validating agent functions belong to correct model when added
     */
    ModelData* model;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_DEPENDENCYGRAPH_H_

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

/**
 * This class represents the dependency tree for agent functions, host functions and submodels. Each DependencyNode has its own dependencies/dependents, the purpose of this class is to
 * walk the dependency tree and provide utility such as validation/layering.
 * @see DependencyNode
 */

class DependencyGraph {
 public:
    DependencyGraph();
    explicit DependencyGraph(const ModelData* _model);
    explicit DependencyGraph(const DependencyGraph& other);
    bool operator==(const DependencyGraph& rhs);
    void addRoot(DependencyNode& root);
    bool validateDependencyGraph();
    void generateLayers(ModelDescription& model);
    void generateDOTDiagram(std::string outputFileName);
    std::string getConstructedLayersString();

 private:
    std::vector<DependencyNode*> roots;
    std::vector<DependencyNode*> functionStack;
    bool validateSubTree(DependencyNode* node);
    void checkForUnattachedFunctions();
    bool doesFunctionExistInStack(DependencyNode* function);
    static std::string getNodeName(DependencyNode* node);
    std::vector<std::vector<std::string>> constructedLayers;
    /**
     * Root of the model hierarchy, used for validating agent functions belong to correct model when added
     */
    const ModelData* model;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_DEPENDENCYGRAPH_H_

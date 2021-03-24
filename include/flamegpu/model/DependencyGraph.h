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

    void addRoot(DependencyNode* root);
    bool validateDependencyGraph();
    void generateLayers(ModelDescription& model);
    void printGraph() const;
    void generateDOTDiagram(std::string outputFileName);

 private:
    std::vector<DependencyNode*> roots;
    std::vector<DependencyNode*> functionStack;
    bool validateSubTree(DependencyNode* node);
    bool doesFunctionExistInStack(DependencyNode* function);
    static std::string getNodeName(DependencyNode* node);
};

#endif  // INCLUDE_FLAMEGPU_MODEL_DEPENDENCYGRAPH_H_

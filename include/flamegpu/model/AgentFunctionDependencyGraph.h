#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDEPENDENCYGRAPH_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDEPENDENCYGRAPH_H_

#include <functional> 

#include "DependencyNode.h"
#include "ModelDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

/**
 * This class represents the dependency tree for agent functions. Each DependencyNode has its own dependencies/dependents, the purpose of this class is to
 * walk the dependency tree and provide utility such as validation/layering.
 * @see DependencyNode
 */

class AgentFunctionDependencyGraph {
    public:
        AgentFunctionDependencyGraph();
        
        void addRoot(DependencyNode* root);
        bool validateDependencyGraph();
        void generateLayers(ModelDescription& model);
        void printGraph() const;
        void generateDOTDiagram(std::string outputFileName) const;
    
    private:
        std::vector<DependencyNode*> roots;
        std::vector<DependencyNode*> functionStack;
        bool validateSubTree(DependencyNode* node);
        bool doesFunctionExistInStack(DependencyNode* function);
};

#endif

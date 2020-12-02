#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDEPENDENCYGRAPH_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDEPENDENCYGRAPH_H_

#include <functional> 

#include "AgentFunctionDescription.h"
#include "ModelDescription.h"

/**
 * This class represents the dependency tree for agent functions. Each AgentFunctionDescription has its own dependencies/dependents, the purpose of this class is to
 * walk the dependency tree and provide utility such as validation/layering.
 * @see AgentFunctionDescription
 */

class AgentFunctionDependencyGraph {
    public:
        AgentFunctionDependencyGraph();
        
        void addRoot(AgentFunctionDescription* root);
        bool validateDependencyGraph();
        void generateLayers(ModelDescription& model);
        void printGraph() const;
        void generateDOTDiagram(std::string outputFileName) const;
    
    private:
        std::vector<AgentFunctionDescription*> roots;
        std::vector<AgentFunctionDescription*> functionStack;
        bool validateSubTree(AgentFunctionDescription* node);
        bool doesFunctionExistInStack(AgentFunctionDescription* function);
};

#endif

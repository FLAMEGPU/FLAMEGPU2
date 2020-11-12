#include "flamegpu/model/AgentFunctionDependencyGraph.h"

AgentFunctionDependencyGraph::AgentFunctionDependencyGraph() {
}

void AgentFunctionDependencyGraph::addRoot(AgentFunctionDescription* root) {
    roots.push_back(root);
}

bool AgentFunctionDependencyGraph::validateDependencyGraph() {
    if (roots.size() == 0) {
            std::cout << "Warning! Agent function dependency graph is empty!" << std::endl;
            THROW InvalidDependencyGraph();
    }
    for(auto& root : roots) {
        if (root->getDependencies().size() != 0) {
            std::cout << "Warning! Root agent function has dependencies!" << std::endl;
            THROW InvalidDependencyGraph();
        }
        if (!validateSubTree(root)) {
            std::cout << "Warning! Dependency graph validation failed! Does the graph have a cycle?" << std::endl;
            THROW InvalidDependencyGraph();
        }
    }
    return true;
}

std::vector<std::vector<std::string>> AgentFunctionDependencyGraph::generateLayers() {
    printf("generateLayers not yet implemented!\n");
    return std::vector<std::vector<std::string>>();
}

bool AgentFunctionDependencyGraph::validateSubTree(AgentFunctionDescription* node) {
    // Check if the function we are looking at already exists on the stack - if so we have a cycle
    if (doesFunctionExistInStack(node)) {
        return false;
    }
    
    // No cycle detected, check if all child nodes form valid subtrees
    functionStack.push_back(node);
    for (auto child : node->getDependents()) {
        if(!validateSubTree(child))
            return false;
    }
   
    // No problems, tree formed by this node is valid
    return true;
} 

bool AgentFunctionDependencyGraph::doesFunctionExistInStack(AgentFunctionDescription* function) {
    // Iterating vector probably faster than using constant lookup structure for likely number of elements
    for (auto fn : functionStack) {
        if (fn == function) {
            return true;
        }
    }
    return false;
}

void AgentFunctionDependencyGraph::printGraph() const {
    printf("printGraph not yet implemented!\n");
}

void AgentFunctionDependencyGraph::generateDOTDiagram(std::string outputFileName) const {
    //std::ofstream(outputFileName);
    printf("generateDOTDiagram not yet implemented!\n");  
}

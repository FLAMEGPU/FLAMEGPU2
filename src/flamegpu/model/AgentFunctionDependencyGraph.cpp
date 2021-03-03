#include "flamegpu/model/AgentFunctionDependencyGraph.h"

AgentFunctionDependencyGraph::AgentFunctionDependencyGraph() {
}

void AgentFunctionDependencyGraph::addRoot(AgentFunctionDescription* root) {
    roots.push_back(root);
}

bool AgentFunctionDependencyGraph::validateDependencyGraph() {
    if (roots.size() == 0) {
            THROW InvalidDependencyGraph("Warning! Agent function dependency graph is empty!");
    }
    for(auto& root : roots) {
        if (root->getDependencies().size() != 0) {
            THROW InvalidDependencyGraph("Warning! Root agent function has dependencies!");
        }
        if (!validateSubTree(root)) {
            THROW InvalidDependencyGraph("Warning! Dependency graph validation failed! Does the graph have a cycle?");
        }
    }
    return true;
}

void AgentFunctionDependencyGraph::generateLayers(ModelDescription& model) {
    
    // Check dependency graph is valid before we attempt to build layers
    validateDependencyGraph();
    
    // Lambda to walk the graph and set minimum layer depths of nodes
    std::function<void(AgentFunctionDescription*, int)> setMinLayerDepths;
    setMinLayerDepths = [&setMinLayerDepths] (AgentFunctionDescription* node, int depth) {
        if (depth >= node->getMinimumLayerDepth()) { 
            node->setMinimumLayerDepth(depth);
        }
        for (auto child : node->getDependents()) {
            setMinLayerDepths(child, depth + 1);
        }
    };

    // Set minimum layer depths
    for (auto root : roots) {
        setMinLayerDepths(root, 0);
    }

    // Build list of functions in their respective ideal layers assuming no conflicts
    std::vector<std::vector<AgentFunctionDescription*>> idealLayers;
    std::function<void(AgentFunctionDescription*)> buildIdealLayers;
    buildIdealLayers = [&buildIdealLayers, &idealLayers] (AgentFunctionDescription* node) {
        // New layers required
        int nodeDepth = node->getMinimumLayerDepth();
        if (nodeDepth >= idealLayers.size()) {
            idealLayers.push_back(std::vector<AgentFunctionDescription*>());
        }
          
        // Add node to relevant layer
        idealLayers[nodeDepth].push_back(node);

        // Repeat for children
        for (auto child : node->getDependents()) {
            buildIdealLayers(child);
        }
    }; 

    for (auto root : roots) {
        buildIdealLayers(root);
    } 

    // idealLayers now contains AgentFunctionDescription pointers in their ideal layers, i.e. assuming no conflicts. 
    // Now iterate structure attempting to add functions to layers.
    // If we encounter conflicts, introduce additional layers as necessary

    for (auto idealLayer : idealLayers) {
        // Request a new layer from the model
        LayerDescription* layer = &model.newLayer();
       
        // Attempt to add each function in the idealLayer to the layer
        for (auto agentFunction : idealLayer) { 
            try {
                layer->addAgentFunction(*agentFunction);
            } catch (const InvalidAgentFunc& e) {
                // Conflict, create new layer and add to that instead
                layer = &model.newLayer();
                layer->addAgentFunction(*agentFunction);
                printf("New function execution layer created - InvalidAgentFunc exception\n");
            } catch (const InvalidLayerMember& e) {
                layer = &model.newLayer();
                layer->addAgentFunction(*agentFunction);
                printf("New function execution layer created - InvalidLayerMember exception\n");
            }
        }
    } 
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
   
    // No problems, tree formed by this node and its children is valid
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
    std::ofstream DOTFile(outputFileName);
    if (DOTFile.is_open()) {
        // File preamble
        DOTFile << "digraph {" << std::endl;

        // Lambda to recursively print relations
        std::function<void(AgentFunctionDescription*)> printRelations;
        printRelations = [&printRelations, &DOTFile] (AgentFunctionDescription* node) {
            // Get this node's name
            std::string parentName = node->getName();
            
            // For each child, print DOT relation and recurse
            for (auto child : node->getDependents()) {
                DOTFile << "    " << parentName << " -> " << child->getName() << ";" << std::endl;
                printRelations(child);
            }
        }; 

        // Recursively print relations
        for (auto root : roots) {
            printRelations(root); 
        }

        // EOF
        DOTFile << "}";
    }
}

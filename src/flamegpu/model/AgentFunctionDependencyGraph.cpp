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
    
    // Lambda to walk the graph and set minimum layer depths of nodes
    std::function<void(AgentFunctionDescription*, int)> setMinLayerDepths;
    setMinLayerDepths = [&setMinLayerDepths] (AgentFunctionDescription* node, int depth) {
        if (depth >= node->getMinimumLayerDepth()) { 
            node->setMinimumLayerDepth(depth);
            printf("Setting depth to %d\n", depth);
        }
        for (auto child : node->getDependents()) {
            setMinLayerDepths(child, depth + 1);
        }
    };

    // Set minimum layer depths
    for (auto root : roots) {
        setMinLayerDepths(root, 0);
    }

    printf("Finished setting depths\n");
    
    // Build list of functions in their respective ideal layers assuming no conflicts
    std::vector<std::vector<AgentFunctionDescription*>> idealLayers;
    std::function<void(AgentFunctionDescription*)> buildIdealLayers;
    buildIdealLayers = [&buildIdealLayers, &idealLayers] (AgentFunctionDescription* node) {
        // New layers required
        int nodeDepth = node->getMinimumLayerDepth();
        printf("Node depth %d\n", nodeDepth);
        if (nodeDepth >= idealLayers.size()) {
            idealLayers.push_back(std::vector<AgentFunctionDescription*>());
            printf("Adding ideal layer\n");
        }
          
        // Add node to relevant layer
        idealLayers[nodeDepth].push_back(node);
        printf("Added node to layer %d\n", nodeDepth);

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

    int extraLayers = 0;
    for (auto idealLayer : idealLayers) {
        printf("Handling ideal layer\n");
        // Request a new layer from the model
        LayerDescription* layer = &model.newLayer();
       
        // Attempt to add each function in the idealLayer to the layer
        for (auto agentFunction : idealLayer) { 
            try {
                layer->addAgentFunction(*agentFunction);
                printf("AgentFunction added to existing layer - no conflict\n");
            } catch (const InvalidAgentFunc& e) {
                // Conflict, create new layer and add to that instead
                layer = &model.newLayer();
                layer->addAgentFunction(*agentFunction);
                printf("New layer created - InvalidAgentFunc exception\n");
            } catch (const InvalidLayerMember& e) {
                layer = &model.newLayer();
                layer->addAgentFunction(*agentFunction);
                printf("New layer created - InvalidLayerMember exception\n");
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

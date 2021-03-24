#include "flamegpu/model/DependencyGraph.h"

DependencyGraph::DependencyGraph() {
}

void DependencyGraph::addRoot(DependencyNode& root) {
    roots.push_back(&root);
}

bool DependencyGraph::validateDependencyGraph() {
    functionStack.clear();

    if (roots.size() == 0) {
            THROW InvalidDependencyGraph("Warning! Agent function dependency graph is empty!");
    }
    for (auto& root : roots) {
        if (root->getDependencies().size() != 0) {
            THROW InvalidDependencyGraph("Warning! Root agent function has dependencies!");
        }
        if (!validateSubTree(root)) {
            THROW InvalidDependencyGraph("Warning! Dependency graph validation failed! Does the graph have a cycle?");
        }
    }
    return true;
}

void DependencyGraph::generateLayers(ModelDescription& model) {
    // Check dependency graph is valid before we attempt to build layers
    validateDependencyGraph();

    // Lambda to walk the graph and set minimum layer depths of nodes
    std::function<void(DependencyNode*, int)> setMinLayerDepths;
    setMinLayerDepths = [&setMinLayerDepths] (DependencyNode* node, int depth) {
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
    std::vector<std::vector<DependencyNode*>> idealLayers;
    std::function<void(DependencyNode*)> buildIdealLayers;
    buildIdealLayers = [&buildIdealLayers, &idealLayers] (DependencyNode* node) {
        // New layers required
        std::vector<std::vector<DependencyNode*>>::size_type nodeDepth = node->getMinimumLayerDepth();
        if (nodeDepth >= idealLayers.size()) {
            idealLayers.push_back(std::vector<DependencyNode*>());
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

    // idealLayers now contains DependencyNode pointers in their ideal layers, i.e. assuming no conflicts.
    // Now iterate structure attempting to add functions to layers.
    // If we encounter conflicts, introduce additional layers as necessary

    for (auto idealLayer : idealLayers) {
        // Request a new layer from the model
        LayerDescription* layer = &model.newLayer();

        // Attempt to add each node in the idealLayer to the layer
        for (auto node : idealLayer) {
            // Add node based on its concrete type
            // Agent function
            if (AgentFunctionDescription* afd = dynamic_cast<AgentFunctionDescription*>(node)) {
                try {
                    layer->addAgentFunction(*afd);
                } catch (const InvalidAgentFunc&) {
                    // Conflict, create new layer and add to that instead
                    layer = &model.newLayer();
                    layer->addAgentFunction(*afd);
                    printf("New function execution layer created - InvalidAgentFunc exception\n");
                } catch (const InvalidLayerMember&) {
                    layer = &model.newLayer();
                    layer->addAgentFunction(*afd);
                    printf("New function execution layer created - InvalidLayerMember exception\n");
                }
            }

            // Submodel
            if (SubModelDescription* smd = dynamic_cast<SubModelDescription*>(node)) {
                try {
                    layer->addSubModel(*smd);
                } catch (const InvalidLayerMember&) {
                    layer = &model.newLayer();
                    layer->addSubModel(*smd);
                    printf("New submodel layer created - InvalidLayerMember exception\n");
                } catch (const InvalidSubModel&) {
                    layer = &model.newLayer();
                    layer->addSubModel(*smd);
                    printf("New submodel layer created - InvalidSubModel exception\n");
                }
            }

            // Host function
            if (HostFunctionDescription* hdf = dynamic_cast<HostFunctionDescription*>(node)) {
#ifndef SWIG
                try {
                    layer->addHostFunction(hdf->getFunctionPtr());
                } catch (const InvalidLayerMember&) {
                    layer = &model.newLayer();
                    layer->addHostFunction(hdf->getFunctionPtr());
                    printf("New host function layer created - InvalidLayerMember exception\n");
                }
#endif
#ifdef SWIG
                try {
                    layer->addHostFunctionCallback(hdf->getCallbackObject());
                } catch (const InvalidLayerMember& e) {
                    layer = &model.newLayer();
                    layer->addHostFunctionCallback(hdf->getCallbackObject());
                    printf("New host function layer created - InvalidLayerMember exception\n");

#endif
            }
        }
    }
}

bool DependencyGraph::validateSubTree(DependencyNode* node) {
    // Check if the function we are looking at already exists on the stack - if so we have a cycle
    if (doesFunctionExistInStack(node)) {
        return false;
    }

    // No cycle detected, check if all child nodes form valid subtrees
    functionStack.push_back(node);
    for (auto child : node->getDependents()) {
        if (!validateSubTree(child))
            return false;
    }

    // No problems, tree formed by this node and its children is valid
    // Pop this function from the stack and return
    functionStack.pop_back();
    return true;
}

bool DependencyGraph::doesFunctionExistInStack(DependencyNode* function) {
    // Iterating vector probably faster than using constant lookup structure for likely number of elements
    for (auto fn : functionStack) {
        if (fn == function) {
            return true;
        }
    }
    return false;
}

void DependencyGraph::printGraph() const {
    printf("printGraph not yet implemented!\n");
}

std::string DependencyGraph::getNodeName(DependencyNode* node) {
    if (AgentFunctionDescription* afd = dynamic_cast<AgentFunctionDescription*>(node)) {
        return afd->getName();
    } else if (HostFunctionDescription* hfd = dynamic_cast<HostFunctionDescription*>(node)) {
        return hfd->getName();
    } else if (SubModelDescription* smd = dynamic_cast<SubModelDescription*>(node)) {
        return smd->getName();
    } else {
        return std::string("DependencyNode without concrete type!");
    }
}

void DependencyGraph::generateDOTDiagram(std::string outputFileName) {
    validateDependencyGraph();
    std::ofstream DOTFile(outputFileName);
    if (DOTFile.is_open()) {
        // File preamble
        DOTFile << "digraph {" << std::endl;

        // Lambda to recursively print nodes
        std::function<void(DependencyNode*)> printNodes;
        printNodes = [&printNodes, &DOTFile] (DependencyNode* node) {
            std::string nodeName = DependencyGraph::getNodeName(node);
            if (dynamic_cast<AgentFunctionDescription*>(node)) {
                DOTFile << "    " << nodeName << "[style = filled, color = red];" << std::endl;
            } else if (dynamic_cast<HostFunctionDescription*>(node)) {
                DOTFile << "    " << nodeName << "[style = filled, color = yellow];" << std::endl;
            } else if (dynamic_cast<SubModelDescription*>(node)) {
                DOTFile << "    " << nodeName << "[style = filled, color = green];" << std::endl;
            }

            for (auto child : node->getDependents()) {
                printNodes(child);
            }
        };

        // Lambda to recursively print relations
        std::function<void(DependencyNode*)> printRelations;
        printRelations = [&printRelations, &DOTFile] (DependencyNode* node) {
            // Get this node's name
            std::string parentName = DependencyGraph::getNodeName(node);

            // For each child, print DOT relation and recurse
            for (auto child : node->getDependents()) {
                DOTFile << "    " << parentName << " -> " << getNodeName(child) << ";" << std::endl;
                printRelations(child);
            }
        };

        // Recursively print nodes
        for (auto root : roots) {
            printNodes(root);
        }

        // Recursively print relations
        for (auto root : roots) {
            printRelations(root);
        }

        // EOF
        DOTFile << "}";
    }
}

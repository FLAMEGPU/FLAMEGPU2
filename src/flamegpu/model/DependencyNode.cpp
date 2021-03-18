#include <iostream>
#include <string>

#include "flamegpu/model/DependencyNode.h"


/**
 * Constructors
 */

DependencyNode::~DependencyNode() {
}

/**
 * Accessors
 */

bool DependencyNode::hasDependents() const {
    return dependents.size() != 0;
}
const std::vector<DependencyNode*> DependencyNode::getDependents() const {
    return dependents;
}
bool DependencyNode::hasDependencies() const {
    return dependencies.size() != 0;
}
const std::vector<DependencyNode*> DependencyNode::getDependencies() const {
    return dependencies;
}

/**
 * Dependency functions
 */
void DependencyNode::setMinimumLayerDepth(const int minLayerDepth) {
    this->minLayerDepth = minLayerDepth;
}

int DependencyNode::getMinimumLayerDepth() {
    return minLayerDepth;
} 

void DependencyNode::dependsOn(DependencyNode* dependency) {
    if (dependency) {
        dependency->addDependent(this);
        dependencies.push_back(dependency);
    } else { 
        // TODO: Improve error message
        THROW InvalidAgentFunc("Attempting to set dependency of DependencyNode to nullptr");
    }
}

void DependencyNode::addDependent(DependencyNode* dependent) {
    dependents.push_back(dependent);
}

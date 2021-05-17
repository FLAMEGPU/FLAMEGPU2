#include <iostream>
#include <string>

#include "flamegpu/model/DependencyNode.h"

#include "flamegpu/model/AgentFunctionDescription.h"

namespace flamegpu {

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
void DependencyNode::setMinimumLayerDepth(const int _minLayerDepth) {
    this->minLayerDepth = _minLayerDepth;
}

int DependencyNode::getMinimumLayerDepth() {
    return minLayerDepth;
}

void DependencyNode::dependsOnImpl(DependencyNode& dependency) {
    if (auto thisAsAFD = dynamic_cast<AgentFunctionDescription*>(this)) {
        if (auto depAsAFD = dynamic_cast<AgentFunctionDescription*>(&dependency)) {
            if (thisAsAFD->model.expired() || !(thisAsAFD->model.lock() == depAsAFD->model.lock())) {
                THROW InvalidDependencyGraph("Attempting to add two agent functions from different models to dependency graph!");
            }
        }
    }
    dependency.addDependent(*this);
    dependencies.push_back(&dependency);
}

void DependencyNode::addDependent(DependencyNode& dependent) {
    dependents.push_back(&dependent);
}

}  // namespace flamegpu

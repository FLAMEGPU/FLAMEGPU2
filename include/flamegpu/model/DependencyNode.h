#ifndef INCLUDE_FLAMEGPU_MODEL_DEPENDENCYNODE_H_
#define INCLUDE_FLAMEGPU_MODEL_DEPENDENCYNODE_H_

#include <vector>
#include "flamegpu/exception/FGPUException.h"

class DependencyNode {
 public:
    virtual ~DependencyNode();

    /**
     * Specifies that this agent function depends on the completion of all of the provided functions
     * @param dependencyList The host functions, agent functions and submodels which this depends on
     */
    template<typename A>
    void dependsOn(A& dep) {
       dependsOnImpl(dep);
    }
    template<typename A, typename...Args>
    void dependsOn(A& dep, Args&...dependencyList) {
       dependsOnImpl(dep);
       dependsOn(dependencyList...);
    }

 protected:
    friend class DependencyGraph;
    /**
     * Vector storing the 'children' of this agent function in the dependency tree, i.e. those functions which depend on the completion of this one
     */
    std::vector<DependencyNode*> dependents;
    /**
     * Vector storing the 'parents' of this agent function in the dependency tree, i.e. those functions which must be completed before this one may begin
     */
    std::vector<DependencyNode*> dependencies;
    /**
     * Adds an agent function to this agent function's list of dependents
     */
    void addDependent(DependencyNode& dependent);
    /**
     * This functions minimum layer depth in the execution graph
     */
    int minLayerDepth = 0;
    /**
     * Auxillary function for dependency construction
     */
    void dependsOnImpl(DependencyNode& dependency);
    /**
     * Sets the minimum layer depth for this agent function
     */
    void setMinimumLayerDepth(const int minLayerDepth);

    /**
     * @return Whether this agent function has any dependents
     */
    bool hasDependents() const;
    /**
     * @return Immutable vector of the dependents of this agent function
     */
    const std::vector<DependencyNode*> getDependents() const;
    /**
     * @return Whether this agent function has any dependencies
     */
    bool hasDependencies() const;
    /**
     * @return The vector of the dependencies of this agent function
     */
    const std::vector<DependencyNode*> getDependencies() const;
     /**
     * @return The minimum layer depth for this agent function
     */
    int getMinimumLayerDepth();
};

#endif  // INCLUDE_FLAMEGPU_MODEL_DEPENDENCYNODE_H_

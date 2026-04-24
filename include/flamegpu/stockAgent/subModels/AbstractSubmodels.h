#ifndef INCLUDE_FLAMEGPU_STOCKAGENT_ABSTRACTSUBMODELS_H_
#define INCLUDE_FLAMEGPU_STOCKAGENT_ABSTRACTSUBMODELS_H_

#include "flamegpu/flamegpu.h"

namespace flamegpu {
namespace stockAgent {
namespace submodels {
    /**
     * Abstract base class for submodels.
     * Submodels are used to group together related agent functions and variables, and to allow for modularity and reusability of code.
     * Submodels can be nested within other submodels, allowing for hierarchical organization of code.
     */
    class AbstractSubmodel {
    public:

        
        virtual void validate() = 0;
        virtual ~AbstractSubmodel() = default;
    };


}  // namespace submodels
}  // namespace stockAgent
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_STOCKAGENT_ABSTRACTSUBMODELS_H_
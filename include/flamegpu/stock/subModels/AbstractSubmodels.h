#ifndef INCLUDE_FLAMEGPU_STOCK_SUBMODELS_ABSTRACTSUBMODELS_H_
#define INCLUDE_FLAMEGPU_STOCK_SUBMODELS_ABSTRACTSUBMODELS_H_

#include <string>
#include "flamegpu/flamegpu.h"

namespace flamegpu {
namespace stock {
namespace submodels {
    /**
     * Abstract base class for submodels.
     * Submodels are used to group together related agent functions and variables, and to allow for modularity and reusability of code.
     * Submodels can be nested within other submodels, allowing for hierarchical organization of code.
     */
    class AbstractSubmodel {
     public:
        virtual ~AbstractSubmodel() = default;

        /**
         * Returns the underlying FLAME GPU SubModelDescription.
         * Throws if the submodel hasn't been initialized/added to a model yet.
         */
        virtual flamegpu::SubModelDescription getSubModelDescription() const = 0;

        /**
         * Validates that all required agents and variables have been mapped.
         */
        virtual void validate() = 0;

        /**
         * Returns the name of the submodel instance.
         */
        virtual std::string getName() const = 0;
    };


}  // namespace submodels
}  // namespace stock
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_STOCK_SUBMODELS_ABSTRACTSUBMODELS_H_

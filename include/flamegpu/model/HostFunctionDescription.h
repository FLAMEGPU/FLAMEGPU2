#ifndef INCLUDE_FLAMEGPU_MODEL_HOSTFUNCTIONDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_HOSTFUNCTIONDESCRIPTION_H_

#include <string>
#include <memory>
#include <vector>

#ifdef SWIG
#include "flamegpu/runtime/HostFunctionCallback.h"
#endif
#include "flamegpu/model/DependencyNode.h"
#include "flamegpu/model/LayerDescription.h"

/**
 * Within the model hierarchy, this class represents a host function for a FLAMEGPU model
 */
class HostFunctionDescription : public DependencyNode {
    /**
     * Constructors
     */
    
    /**
     * Default copy constructor, not implemented
     */
    HostFunctionDescription(const HostFunctionDescription &other_function) = delete;
    /**
     * Default move constructor, not implemented
     */
    HostFunctionDescription(HostFunctionDescription &&other_function) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    HostFunctionDescription& operator=(const HostFunctionDescription &other_function) = delete;
    /**
     * Default move assignment, not implemented
     */
    HostFunctionDescription& operator=(HostFunctionDescription &&other_function) noexcept = delete;

 public:
   HostFunctionDescription(std::string host_function_name, FLAMEGPU_HOST_FUNCTION_POINTER host_function);
   HostFunctionDescription(std::string host_funcation_name, HostFunctionCallback *func_callback);

    /**
     * Equality operator, checks whether HostFunctionDescription hierarchies are functionally the same
     * @returns True when agent functions are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const HostFunctionDescription& rhs) const;
    /**
     * Equality operator, checks whether HostFunctionDescription hierarchies are functionally different
     * @returns True when agent functions are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const HostFunctionDescription& rhs) const;

    /**
     * @return The function's name
     */
    //std::string getName() const;
    
    /**
     * @return The cuda kernel entry point for executing the agent function
     */

    FLAMEGPU_HOST_FUNCTION_POINTER getFunctionPtr() const;
    HostFunctionCallback* getCallbackObject();
    std::string getName();


 private:

    FLAMEGPU_HOST_FUNCTION_POINTER function;
    HostFunctionCallback* callbackObject;
    std::string name;
    
};

#endif  // INCLUDE_FLAMEGPU_MODEL_HostFunctionDescription_H_

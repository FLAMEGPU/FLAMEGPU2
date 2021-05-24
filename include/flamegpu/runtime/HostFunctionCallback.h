#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_

#include "flamegpu/runtime/HostAPI_macros.h"

/**
 * Virtual Callback class which allows a user to create a callback function to be used as a host function in either step, exit or init functions.
 * This is mostly required to allow swig wrapping of host functions in a target language
 */
class HostFunctionCallback {
 public:
    /**
     * This is the method that will be implemented by a user in Python to define their host function
     */
    virtual void run(HostAPI*) = 0;
    /**
     * Virtual destructor for correct inheritance behaviour
     */
    virtual ~HostFunctionCallback() {}
};

/**
 * Virtual Callback class which allows a user to create a callback function for an exit condition.
 * Different to HostFunctionCallback as it returns a value.
 * This is mostly required to allow swig wrapping of host functions in a target language
 */
class HostFunctionConditionCallback {
 public:
     /**
      * This is the method that will be implemented by a user in Python to define their host condition
      * @return Either pyflamegpu.EXIT or pyflamegpu.CONTINUE, denoting whether the simulation should exit
      */
    virtual FLAME_GPU_CONDITION_RESULT run(HostAPI*) = 0;
    /**
     * Virtual destructor for correct inheritance behaviour
     */
    virtual ~HostFunctionConditionCallback() {}
};


#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_

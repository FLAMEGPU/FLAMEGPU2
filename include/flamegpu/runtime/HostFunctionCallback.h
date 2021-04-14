#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_

// class HostAPI;

class HostFunctionCallback {
  /**
   * Virtual Callback class which allows a user to create a callback function to be used as a host function in either step, exit or init functions.
   * This is mostly required to allow swig wrapping of host functions in a target language
   */
 public:
    virtual void run(HostAPI*) = 0;
    virtual ~HostFunctionCallback() {}
};

class HostFunctionConditionCallback {
  /**
   * Virtual Callback class which allows a user to create a callback function for an exit condition.
   * Different to HostFunctionCallback as it returns a value.
   * This is mostly required to allow swig wrapping of host functions in a target language
   */
 public:
    virtual FLAME_GPU_CONDITION_RESULT run(HostAPI*) = 0;
    virtual ~HostFunctionConditionCallback() {}
};


#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_

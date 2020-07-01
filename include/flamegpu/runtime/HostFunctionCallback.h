#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_

// class FLAMEGPU_HOST_API;

class HostFunctionCallback {
  /**
   * Virtual Callback class which allows a user to create a callback function to be used as a host function in either step, exit or init functions.
   * This is mostly required to allow swig wrapping of host functions in a target language
   */
 public:
    virtual void run(FLAMEGPU_HOST_API*) = 0;
    virtual ~HostFunctionCallback() {}
};


#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTFUNCTIONCALLBACK_H_

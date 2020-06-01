#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_H_

/**
 * This empty class is used when messaging is not enabled for an agent function
 * It also provides the best overview of the required components of a new messsaging type
 */
class MsgNone {
 public:
    /**
     * Common size type
     */
    typedef unsigned int size_type;
    // Host (Data and Description not required for None)
    class CUDAModelHandler;
    // Device
    class In;
    class Out;
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_H_

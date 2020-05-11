#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY_H_

#include "flamegpu/runtime/messaging/BruteForce.h"


/**
 * Array messaging functionality
 *
 * Like an array, each message is assigned an index within a known range
 * Only one message may exist at each index
 * Agent functions can access individual messages by requesting them with their index
 * 
 * Algorithm:
 * Every agent outputs a message to the array based on their thread index
 * They also set the __index variable with the intended output bin
 * When buildIndex() is called, messages are sorted and errors (multiple messages per bin) are detected
 */
class MsgArray {
 public:
    /**
     * Common size type
     */
    typedef MsgNone::size_type size_type;

    // Host
    struct Data;        // Forward declare inner classes
    class Description;  // Forward declare inner classes
    class CUDAModelHandler;

    // Device
    class In;
    class Out;

    /**
     * MetaData required by brute force during message reads
     */
    struct MetaData {
        /**
         * Length
         */
        size_type length;
    };
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY_H_

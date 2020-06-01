#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_

#include "flamegpu/runtime/messaging/None.h"

struct ModelData;

/**
 * Brute force messaging functionality
 *
 * Every agent accesses all messages
 * This technique is expensive, and other techniques are preferable if operating with more than 1000 messages.
 */
class MsgBruteForce {
 public:
    /**
     * Common size type
     */
    typedef MsgNone::size_type size_type;

    // Host
    struct Data;        // Forward declare inner classes
    class Description;  // Forward declare inner classes
    class CUDAModelHandler;  // Forward declare inner classes
    // Device
    class In;  // Forward declare inner classes
    class Out;  // Forward declare inner classes
    /**
     * MetaData required by brute force during message reads
     */
    struct MetaData {
        unsigned int length;
    };
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_

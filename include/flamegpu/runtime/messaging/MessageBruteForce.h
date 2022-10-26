#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCE_H_

#include "flamegpu/runtime/messaging/MessageNone.h"

namespace flamegpu {

struct ModelData;

/**
 * Brute force messaging functionality
 *
 * Every agent accesses all messages
 * This technique is expensive, and other techniques are preferable if operating with more than 1000 messages.
 */
class MessageBruteForce {
 public:
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
        unsigned int length = 0;
    };
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCE_H_

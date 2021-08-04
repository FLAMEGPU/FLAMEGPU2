#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY3D_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY3D_H_

#include "flamegpu/runtime/messaging/MessageBruteForce.h"
#include "flamegpu/runtime/messaging/MessageArray2D.h"

namespace flamegpu {

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
class MessageArray3D {
 public:
    /**
     * Common size type
     */
    typedef MessageNone::size_type size_type;

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
         * Dimensions of array
         */
        size_type dimensions[3];
        /**
         * Total number of elements
         */
        size_type length;
    };
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY3D_H_

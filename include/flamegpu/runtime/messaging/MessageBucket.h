#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBUCKET_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBUCKET_H_

#ifndef __CUDACC_RTC__
#include <memory>
#include <string>
#endif  // __CUDACC_RTC__

#include "flamegpu/runtime/messaging/MessageNone.h"
#include "flamegpu/runtime/messaging/MessageBruteForce.h"

namespace flamegpu {

typedef int IntT;

/**
 * Bucket messaging functionality
 *
 * User specifies an integer upper and lower bound, these form a set of consecutive indices which act as keys to buckets.
 * Each bucket may contain 0 to many messages, however an index is generated such that empty bins still consume a small amount of space.
 * As such, this is similar to a multi-map, however the key space must be a set of consecutive integers.
 *
 * By using your own hash function you can convert non-integer keys to suitable integer keys.
 */
class MessageBucket {
 public:
    // Host
    struct Data;
    class CDescription;
    class Description;
    class CUDAModelHandler;

    // Device
    class In;
    class Out;

    /**
     * MetaData required by bucket messaging during message reads
     */
    struct MetaData {
        /**
         * The inclusive minimum environment bound
         */
        IntT min;
        /**
         * The exclusive maximum environment bound
         */
        IntT max;
        /**
         * Pointer to the partition boundary matrix in device memory
         * The PBM is never stored on the host
         */
        unsigned int *PBM;
    };
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBUCKET_H_

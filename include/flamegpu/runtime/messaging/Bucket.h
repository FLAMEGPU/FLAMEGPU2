#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_H_

#ifndef __CUDACC_RTC__
#include <memory>
#include <string>

#include "flamegpu/runtime/cuRVE/curve.h"
#endif  // __CUDACC_RTC__

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"

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
class MsgBucket {
 public:
    /**
     * Common size type
     */
    typedef MsgNone::size_type size_type;

    // Host
    struct Data;
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

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_H_

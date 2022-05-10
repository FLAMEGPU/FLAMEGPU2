#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_MESSAGENONEDEVICE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_MESSAGENONEDEVICE_CUH_

#include "flamegpu/runtime/detail/curve/Curve.cuh"
#include "flamegpu/runtime/messaging/MessageNone.h"

namespace flamegpu {

/**
 * Provides message input functionality during agent functions
 * Constructed and owned by DeviceAPI
 */
class MessageNone::In {
 public:
    /**
     * Constructor
     * Takes a device pointer to a struct for metadata related to accessing the messages (e.g. an index data structure)
     */
    __device__ In(const void * /*metadata*/) {
    }
};
/**
 * Provides message output functionality during agent functions
 * Constructed and owned by DeviceAPI
 */
class MessageNone::Out {
 public:
    /**
     * Constructor
     * Takes a device pointer to a struct for metadata related to accessing the messages (e.g. an index data structure)
     */
    __device__ Out(const void * /*metadata*/, unsigned int * /*scan_flag_messageOutput*/){
    }
};

}  // namespace flamegpu


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_MESSAGENONEDEVICE_CUH_

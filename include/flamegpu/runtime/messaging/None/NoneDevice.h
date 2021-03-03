#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEDEVICE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEDEVICE_H_

#ifndef __CUDACC_RTC__
#include "flamegpu/runtime/cuRVE/curve.h"
#endif  // __CUDACC_RTC__

#include "flamegpu/runtime/messaging/None.h"


/**
 * Provides message input functionality during agent functions
 * Constructed and owned by DeviceAPI
 */
class MsgNone::In {
 public:
    /**
     * Constructor
     * Requires CURVE hashes for agent function and message name to retrieve variable memory locations
     * Takes a device pointer to a struct for metadata related to accessing the messages (e.g. an index data structure)
     */
    __device__ In(Curve::NamespaceHash /*agent fn hash*/, Curve::NamespaceHash /*message name hash*/, const void * /*metadata*/) {
    }
};
/**
 * Provides message output functionality during agent functions
 * Constructed and owned by DeviceAPI
 */
class MsgNone::Out {
 public:
    /**
     * Constructor
     * Requires CURVE hashes for agent function and message name to retrieve variable memory locations
     * Takes a device pointer to a struct for metadata related to accessing the messages (e.g. an index data structure)
     */
    __device__ Out(Curve::NamespaceHash /*agent fn hash*/, Curve::NamespaceHash /*message name hash*/, const void * /*metadata*/, unsigned int * /*scan_flag_messageOutput*/){
    }
};


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEDEVICE_H_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_H_

#include "flamegpu/defines.h"

namespace flamegpu {

/**
 * This empty class is used when messaging is not enabled for an agent function
 * It also provides the best overview of the required components of a new messsaging type
 */
class MessageNone {
 public:
    // Host (Data and Description not required for None)
    class CUDAModelHandler;
    // Device
    class In;
    class Out;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_H_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEHOST_H_

#include "flamegpu/runtime/messaging/MsgSpecialisationHandler.h"
#include "flamegpu/runtime/messaging/None.h"

namespace flamegpu {

class CUDAMessage;

/**
 * Provides specialisation behaviour for messages between agent functions
 * e.g. allocates/initialises additional data structure memory, sorts messages and builds an index
 * Created and owned by CUDAMessage
 */
class MsgNone::CUDAModelHandler : public MsgSpecialisationHandler {
 public:
    /**
     * Constructor
     */
    explicit CUDAModelHandler(CUDAMessage &a)
        : MsgSpecialisationHandler()
        , sim_message(a)
    { }
    /**
     * Owning CUDAMessage
     */
    CUDAMessage &sim_message;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEHOST_H_

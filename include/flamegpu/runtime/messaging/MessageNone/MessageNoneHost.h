#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_MESSAGENONEHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_MESSAGENONEHOST_H_

#include "flamegpu/runtime/messaging/MessageSpecialisationHandler.h"
#include "flamegpu/runtime/messaging/MessageNone.h"

namespace flamegpu {
namespace detail {
class CUDAMessage;
}  // namespace detail

/**
 * Provides specialisation behaviour for messages between agent functions
 * e.g. allocates/initialises additional data structure memory, sorts messages and builds an index
 * Created and owned by CUDAMessage
 */
class MessageNone::CUDAModelHandler : public MessageSpecialisationHandler {
 public:
    /**
     * Constructor
     */
    explicit CUDAModelHandler(detail::CUDAMessage &a)
        : MessageSpecialisationHandler()
        , sim_message(a)
    { }
    /**
     * Owning CUDAMessage
     */
    detail::CUDAMessage &sim_message;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGENONE_MESSAGENONEHOST_H_

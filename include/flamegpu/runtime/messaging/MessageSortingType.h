#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESORTINGTYPE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESORTINGTYPE_H_

namespace flamegpu {

/**
 * This empty class is used when messaging is not enabled for an agent function
 * It also provides the best overview of the required components of a new messsaging type
 */
enum class MessageSortingType {
    none,
    spatial2D,
    spatial3D
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESORTINGTYPE_H_

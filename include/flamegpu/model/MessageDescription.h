#ifndef INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_

#include <string>

#include "ModelDescription.h"

class MessageDescription {
    /**
     * Only way to construct an MessageDescription
     */
    friend MessageDescription& ModelDescription::newMessage(const std::string &);
    
    /**
     * Constructors
     */
    MessageDescription(const std::string &message_name);
    // Copy Construct
    MessageDescription(const MessageDescription &other_message);
    // Move Construct
    MessageDescription(MessageDescription &&other_message);
    // Copy Assign
    MessageDescription& operator=(const MessageDescription &other_message);
    // Move Assign
    MessageDescription& operator=(MessageDescription &&other_message);

    MessageDescription clone(const std::string &cloned_message_name) const;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
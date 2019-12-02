#ifndef INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_

#include <string>

#include "ModelDescription.h"

/**
 * Base-class, represents brute-force messages
 * Can be extended by more advanced message descriptors
 */
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
 public:
    /**
     * Typedefs
     */
    typedef unsigned int size_type;
    typedef std::map<const std::string, std::tuple<std::type_index, size_t, unsigned int>> VariableMap;

    /**
     * Accessors
     */
    template<typename T, size_type N = 1>
    void newVariable(const std::string &variable_name);

    /**
    * Const Accessors
    */
    std::string getName() const;

    std::type_index getVariableType(const std::string &variable_name) const;
    size_t getVariableSize(const std::string &variable_name) const;
    size_type getVariableLength(const std::string &variable_name) const;
    size_type getVariablesCount() const;

    const VariableMap &getVariables() const;

    bool hasVariable(const std::string &variable_name) const;

 private:
    std::string name;
    VariableMap variables;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
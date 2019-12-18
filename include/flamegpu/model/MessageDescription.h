#ifndef INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_

#include <string>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

struct ModelData;
struct MessageData;

/**
 * Base-class, represents brute-force messages
 * Can be extended by more advanced message descriptors
 */
class MessageDescription {
    friend struct MessageData;
    friend void AgentFunctionDescription::setMessageOutput(MessageDescription&);
    friend void AgentFunctionDescription::setMessageInput(MessageDescription&);

    /**
     * Constructors
     */
    MessageDescription(ModelData *const _model, MessageData *const data);
    // Copy Construct
    MessageDescription(const MessageDescription &other_message) = delete;
    // Move Construct
    MessageDescription(MessageDescription &&other_message) noexcept = delete;
    // Copy Assign
    MessageDescription& operator=(const MessageDescription &other_message) = delete;
    // Move Assign
    MessageDescription& operator=(MessageDescription &&other_message) noexcept = delete;

 public:
     bool operator==(const MessageDescription& rhs) const;
     bool operator!=(const MessageDescription& rhs) const;

    /**
     * Accessors
     */
    template<typename T, ModelData::size_type N = 1>
    void newVariable(const std::string &variable_name);

    /**
     * Const Accessors
     */
    std::string getName() const;

    std::type_index getVariableType(const std::string &variable_name) const;
    size_t getVariableSize(const std::string &variable_name) const;
    ModelData::size_type getVariableLength(const std::string &variable_name) const;
    ModelData::size_type getVariablesCount() const;

    bool hasVariable(const std::string &variable_name) const;

 private:
    ModelData *const model;
    MessageData *const message;
};

/**
 * Template implementation
 */
template<typename T, ModelData::size_type N>
void MessageDescription::newVariable(const std::string &variable_name) {
    if (message->variables.find(variable_name) == message->variables.end()) {
        message->variables.emplace(variable_name, ModelData::Variable(N, T()));
        return;
    }
    THROW InvalidAgentVar("Message ('%s') already contains variable '%s', "
        "in MessageDescription::newVariable().",
        message->name.c_str(), variable_name.c_str());
}

#endif  // INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_

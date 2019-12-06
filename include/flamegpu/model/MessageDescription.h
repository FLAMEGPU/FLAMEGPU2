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
    MessageDescription();
    MessageDescription(ModelDescription * const model, const std::string &message_name);
    // Copy Construct
    MessageDescription(const MessageDescription &other_message);
    // Move Construct
    MessageDescription(MessageDescription &&other_message) noexcept;
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
    struct Variable
    {
        /**
         * Cannot explicitly specify template args of constructor, so we take redundant arg for implicit template
         */
        template<typename T>
        Variable(size_type _elements, T)
            : type(typeid(T)), type_size(sizeof(T)), elements(elements) { }
        const std::type_index type;
        const size_t type_size;
        const  unsigned int elements;
    };
    typedef std::map<const std::string, Variable> VariableMap;

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
    ModelDescription * const model;
};

/**
 * Template implementation
 */
template<typename T, MessageDescription::size_type N>
void MessageDescription::newVariable(const std::string &variable_name) {
    if (variables.find(variable_name) == variables.end()) {
        variables.emplace(variable_name, Variable(N, T()));
        return;
    }
    THROW InvalidAgentVar("Message ('%s') already contains variable '%s', "
        "in MessageDescription::newVariable().",
        name.c_str(), variable_name.c_str());
}

#endif  // INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
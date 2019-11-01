
 /**
 * @file MessageDescription.h
 * @authors
 * @brief
 *
 * @see
 * @warning
 */

#ifndef INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_

#include <utility>
#include <string>
#include <typeinfo>
#include <map>

#define DEFAULT_MESSAGE_BUFFER_SIZE 1024

typedef std::map<const std::string, const std::type_info&> VariableMap;

typedef std::pair<const std::string, const std::type_info&> VariableMapPair;

typedef std::map<const std::type_info*, std::size_t> VarTypeSizeMap;    // to track size of data types

class MessageDescription {
 public:
    explicit MessageDescription(const std::string message_name, unsigned int initial_size = DEFAULT_MESSAGE_BUFFER_SIZE);

    virtual ~MessageDescription();

    const std::string getName() const;

    template <typename T> void addVariable(const std::string variable_name);

    VariableMap& getVariableMap();

    const VariableMap& getVariableMap() const;

    size_t getMessageVariableSize(const std::string variable_name) const;

    size_t getMemorySize() const;

    unsigned int getNumberMessageVariables() const;

    const std::type_info& getVariableType(const std::string variable_name) const;

    unsigned int getMaximumMessageListCapacity() const;

 private:
    const std::string name;
    VariableMap variables;
    VarTypeSizeMap sizes;
    unsigned int maximum_size;  // size is maximum buffer size for messages
};

template <typename T> void MessageDescription::addVariable(const std::string variable_name) {
    variables.insert(variables.end(), VariableMap::value_type(variable_name, typeid(T)));
    sizes.insert(VarTypeSizeMap::value_type(&typeid(T), (unsigned int)sizeof(T)));
}

#endif // INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_

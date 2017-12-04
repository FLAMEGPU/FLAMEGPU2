
 /**
 * @file MessageDescription.h
 * @authors
 * @brief
 *
 * @see
 * @warning
 */

#ifndef MESSAGEDESCRIPTION_H_
#define MESSAGEDESCRIPTION_H_

#include <string>
#include <typeinfo>
#include <map>

typedef std::map<const std::string, const std::type_info&> VariableMap;

typedef std::pair<const std::string, const std::type_info&> VariableMapPair;

typedef std::map<const std::type_info*, std::size_t> VarTypeSizeMap;	// to track size of data types


class MessageDescription {
public:
	MessageDescription(const std::string message_name);

	virtual ~MessageDescription();

	const std::string getName() const;

	template <typename T> void addVariable(const std::string variable_name);

	VariableMap& getVariableMap();

    const VariableMap& getVariableMap() const;

    const size_t getMessageVariableSize(const std::string variable_name) const;

	size_t getMemorySize() const;

    unsigned int getNumberMessageVariables() const;

	const std::type_info& getVariableType(const std::string variable_name) const;

private:
	const std::string name;
	VariableMap variables;
    VarTypeSizeMap sizes;
};

template <typename T> void MessageDescription::addVariable(const std::string variable_name){
	variables.insert(variables.end(), VariableMap::value_type(variable_name, typeid(T)));
	sizes.insert(VarTypeSizeMap::value_type(&typeid(T), (unsigned int)sizeof(T)));
}


#endif /* MESSAGEDESCRIPTION_H_ */

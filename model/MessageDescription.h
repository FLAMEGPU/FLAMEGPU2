
 /**
 * @file MessageDescription.h
 * @authors Paul
 * @date 5 Mar 2014
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

class MessageDescription {
public:
	MessageDescription(const std::string message_name);

	virtual ~MessageDescription();

	const std::string getName() const;

	template <typename T> void addVariable(const std::string variable_name);

private:
	const std::string name;
	VariableMap variables;
};

template <typename T> void MessageDescription::addVariable(const std::string variable_name){
	variables.insert(variables.end(), VariableMap::value_type(variable_name, typeid(T)));
}


#endif /* MESSAGEDESCRIPTION_H_ */

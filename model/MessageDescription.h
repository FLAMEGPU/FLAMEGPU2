/*
 * MessageDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef MESSAGEDESCRIPTION_H_
#define MESSAGEDESCRIPTION_H_

#include <string>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/container/map.hpp>
#include <typeinfo>

typedef std::map<const std::string, const std::type_info&> VariableMap;

class MessageDescription {
public:
	MessageDescription(const std::string message_name);

	virtual ~MessageDescription();

	const std::string getName() const;

	template <typename T> void addVariable(const std::string variable_name){
		variables.insert(variables.end(), VariableMap::value_type(variable_name, typeid(T)));
	}

private:
	const std::string name;
	VariableMap variables;
};

#endif /* MESSAGEDESCRIPTION_H_ */

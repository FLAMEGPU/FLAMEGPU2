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

typedef std::map<std::string, const std::type_info&> VariableMap;

class MessageDescription {
public:
	MessageDescription(std::string name) : variables() { this->name = name; }
	virtual ~MessageDescription() {}

	std::string getName() const;

	void setName(std::string name);

	template <typename T> void addVariable(std::string variable_name){
		variables.insert(variables.end(), VariableMap::value_type(variable_name, typeid(T)));
	}

private:
	std::string name;
	VariableMap variables;
};

#endif /* MESSAGEDESCRIPTION_H_ */

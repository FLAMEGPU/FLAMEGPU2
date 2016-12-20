/*
* AgentMemoryVector.h
*
*/


#ifndef AGENTMEMORYVECTOR_H_
#define AGENTMEMORYVECTOR_H_

#include <string>
#include <vector>
#include <ostream>
#include <typeinfo>

#include "../exception/FGPUException.h"

class GenericAgentMemoryVector{
public:

	virtual ~GenericAgentMemoryVector(){ ; }
	
	virtual const std::type_info& getType() = 0;

	virtual const void* getDataPtr() const = 0;

	virtual void* getVectorPtr() = 0;

	virtual unsigned int incrementVector() = 0;

	virtual GenericAgentMemoryVector* clone() const = 0;

	virtual void resize(unsigned int) = 0;

	template <typename T> std::vector<T>& getVector();

	template <typename T> std::vector<T> getVectorIteratorAt(unsigned int i);

};

template <typename T> 
class AgentMemoryVector : public GenericAgentMemoryVector
{

public:

	AgentMemoryVector() : GenericAgentMemoryVector(), type(typeid(T)) {
		default_value = T();
	}

	virtual ~AgentMemoryVector(){ ; }

	virtual const std::type_info& getType(){
		return type;
	}

	virtual const void* getDataPtr() const{
		if (vec.empty())
			return NULL;
		else
			return &(vec.front());
	}

	virtual void* getVectorPtr()
	{
		return static_cast<void*>(&vec);
	}

	virtual unsigned int incrementVector()
	{
		vec.push_back(default_value);
		return static_cast<unsigned int>(vec.size());
	}

	virtual AgentMemoryVector<T>* clone() const
	{
		return (new AgentMemoryVector<T>());
	}

	virtual void resize(unsigned int s)
	{
		vec.resize(s);
	}


protected:
	std::vector<T> vec;
	T default_value;
	const std::type_info& type;
};

template <typename T> std::vector<T>& GenericAgentMemoryVector::getVector(){

	if (getType() != typeid(T))
		//throw std::runtime_error("Bad variable type in agent instance set variable");
		throw InvalidVarType("bad");

	//must cast the vector as the correct type
	std::vector<T> *t_v = static_cast<std::vector<T>*>(getVectorPtr());
	//return reference
	return *t_v;
}

template <typename T> std::vector<T> GenericAgentMemoryVector::getVectorIteratorAt(unsigned int i){

	//return an iterator at correct position
	std::vector<T>& v = getVector<T>();
	return (v.begin() + i);
}


#endif //AGENTMEMORYVECTOR_H_
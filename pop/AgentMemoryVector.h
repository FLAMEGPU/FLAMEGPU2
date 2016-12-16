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


class GenericAgentMemoryVector{
public:

	virtual ~GenericAgentMemoryVector(){ ; }
	
	virtual const std::type_info& getType() = 0;

	virtual void* getDataPtr() = 0;

	virtual void* getVectorPtr() = 0;

	virtual unsigned int incrementVector() = 0;

	virtual GenericAgentMemoryVector* clone() const = 0;


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

	virtual void* getDataPtr(){
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



protected:
	std::vector<T> vec;
	T default_value;
	const std::type_info& type;
};

#endif //AGENTMEMORYVECTOR_H_
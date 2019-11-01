/*
* MemoryVector.h
*
*/


#ifndef AGENTMEMORYVECTOR_H_
#define AGENTMEMORYVECTOR_H_

#include <string>
#include <vector>
#include <ostream>
#include <typeinfo>
#include <map>
#include <memory>

#include <flamegpu/exception/FGPUException.h>

class GenericMemoryVector{
public:

    virtual ~GenericMemoryVector(){ ; }

    virtual const std::type_info& getType() = 0;

    virtual void* getDataPtr() = 0;

    virtual const void* getReadOnlyDataPtr() const = 0;

    virtual void* getVectorPtr() = 0;

    virtual GenericMemoryVector* clone() const = 0;

    virtual void resize(unsigned int) = 0;

    template <typename T> std::vector<T>& getVector();

    template <typename T> std::vector<T> getVectorIteratorAt(unsigned int i);

};

template <typename T>
class MemoryVector : public GenericMemoryVector
{

public:

    MemoryVector() : GenericMemoryVector(), type(typeid(T)) {
        default_value = T();
    }

    virtual ~MemoryVector(){ ; }

    virtual const std::type_info& getType(){
        return type;
    }

    virtual void* getDataPtr(){
        if (vec.empty())
            return NULL;
        else
            return &(vec.front());
    }

    virtual const void* getReadOnlyDataPtr() const{
        if (vec.empty())
            return NULL;
        else
            return &(vec.front());
    }

    virtual void* getVectorPtr()
    {
        return static_cast<void*>(&vec);
    }

    virtual MemoryVector<T>* clone() const
    {
        return (new MemoryVector<T>());
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

template <typename T> std::vector<T>& GenericMemoryVector::getVector(){

    if (getType() != typeid(T))
        throw InvalidVarType("Wrong variable type getting agent data vector");

    //must cast the vector as the correct type
    std::vector<T> *t_v = static_cast<std::vector<T>*>(getVectorPtr());
    //return reference
    return *t_v;
}

template <typename T> std::vector<T> GenericMemoryVector::getVectorIteratorAt(unsigned int i){

    //return an iterator at correct position
    std::vector<T>& v = getVector<T>();
    return (v.begin() + i);
}


//use this to store default values for a population, must be here to register the correct types at compile time
/*! Create a map with std::strings for keys (indexes) and GenericAgentMemoryVector object. A smart pointer has been used to automaticaly manage the object*/
typedef std::map<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMap;

/*! Create a pair with std::strings for keys (indexes) and GenericAgentMemoryVector object.  A smart pointer has been used to automaticaly manage the object*/
typedef std::pair<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMapPair;

#endif // AGENTMEMORYVECTOR_H_

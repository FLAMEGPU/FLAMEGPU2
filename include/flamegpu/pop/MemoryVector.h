/*
* MemoryVector.h
*
*/

#ifndef INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_
#define INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_

#include <vector>
#include <ostream>
#include <typeinfo>
#include <map>
#include <memory>
#include <utility>
#include <string>

#include "flamegpu/exception/FGPUException.h"

class GenericMemoryVector{
 public:
    /**
     * Acts as a multiplier
     */
    virtual ~GenericMemoryVector() { ; }

    virtual const std::type_index& getType() const = 0;

    virtual void* getDataPtr() = 0;

    virtual const void* getReadOnlyDataPtr() const = 0;

    virtual void* getVectorPtr() = 0;

    virtual GenericMemoryVector* clone() const = 0;

    virtual void resize(unsigned int) = 0;
};

template <typename T>
class MemoryVector : public GenericMemoryVector {
 public:
    /**
     * Memory vector is an array of variables
     * @param _sub_length the length of the array within a variable, most variables will be 1 (a lone variable)
     */
    explicit MemoryVector(unsigned int _sub_length = 1)
    : GenericMemoryVector()
    , sub_length(_sub_length)
    , type(typeid(T)) { }

    virtual ~MemoryVector() { ; }

    const std::type_index& getType() const override {
        return type;
    }

    void* getDataPtr() override {
        if (vec.empty())
            return nullptr;
        else
            return &(vec.front());
    }

    const void* getReadOnlyDataPtr() const override {
        if (vec.empty())
            return nullptr;
        else
            return &(vec.front());
    }

    virtual void* getVectorPtr() {
        return static_cast<void*>(&vec);
    }

    virtual MemoryVector<T>* clone() const {
        return (new MemoryVector<T>(sub_length));
    }

    void resize(unsigned int s) override {
        vec.resize(s * sub_length);
    }

 protected:
    /**
     * Multiplies with size to allocate enough memory
     */
    const unsigned int sub_length;
    std::vector<T> vec;
    const std::type_index type;
};

// use this to store default values for a population, must be here to register the correct types at compile time
/*! Create a map with std::strings for keys (indexes) and GenericAgentMemoryVector object. A smart pointer has been used to automaticaly manage the object*/
typedef std::map<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMap;

/*! Create a pair with std::strings for keys (indexes) and GenericAgentMemoryVector object.  A smart pointer has been used to automaticaly manage the object*/
typedef std::pair<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMapPair;

#endif  // INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_

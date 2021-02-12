/*
* MemoryVector.h
*
*/

#ifndef INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_
#define INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_

#include <vector>
#include <typeindex>
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
    virtual ~GenericMemoryVector() { }

    virtual const std::type_index& getType() const = 0;

    virtual unsigned int getElements() const = 0;

    virtual size_t getTypeSize() const = 0;

    virtual size_t getVariableSize() const = 0;

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
     * @param _elements the length of the array within a variable, most variables will be 1 (a lone variable)
     */
    explicit MemoryVector(unsigned int _elements = 1)
    : GenericMemoryVector()
    , elements(_elements)
    , type(typeid(T))
    , type_size(sizeof(T)) { }

    virtual ~MemoryVector() { ; }

    const std::type_index& getType() const override {
        return type;
    }
    unsigned int getElements() const {
        return elements;
    }
    size_t getTypeSize() const {
        return type_size;
    }
    size_t getVariableSize() const {
        return type_size * elements;
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
        return (new MemoryVector<T>(elements));
    }

    void resize(unsigned int s) override {
        vec.resize(s * elements);
    }

 protected:
    /**
     * Multiplies with size to allocate enough memory
     */
    const unsigned int elements;
    std::vector<T> vec;
    const std::type_index type;
    const size_t type_size;
};

// use this to store default values for a population, must be here to register the correct types at compile time
/*! Create a map with std::strings for keys (indexes) and GenericAgentMemoryVector object. A smart pointer has been used to automaticaly manage the object*/
typedef std::map<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMap;

/*! Create a pair with std::strings for keys (indexes) and GenericAgentMemoryVector object.  A smart pointer has been used to automaticaly manage the object*/
typedef std::pair<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMapPair;

#endif  // INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_

#ifndef INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_
#define INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_

#include <vector>
#include <typeindex>
#include <map>
#include <memory>
#include <utility>
#include <string>

#include "flamegpu/exception/FGPUException.h"

namespace flamegpu {

/**
 * Pure abstract memory vector
 */
class GenericMemoryVector{
 public:
    /**
     * Acts as a multiplier
     */
    virtual ~GenericMemoryVector() { }
    /**
     * Returns the type index of the base type of an element.
     */
    virtual const std::type_index& getType() const = 0;
    /**
     * Returns the number of elements in a variable
     * This is likely to be 1 unless the variable is an array variable
     * @see getTypeSize()
     * @see getVariableSize()
     */
    virtual unsigned int getElements() const = 0;
    /**
     * Returns the size (in bytes) of a base type of an element of the MemoryVector
     * e.g. = type size = variable size / elements
     * @see getElements()
     * @see getVariableSize()
     */
    virtual size_t getTypeSize() const = 0;
    /**
     * Returns the size (in bytes) of a single element of the MemoryVector.
     * e.g. variable size = type size * elements
     * @see getElements()
     * @see getTypeSize()
     */
    virtual size_t getVariableSize() const = 0;
    /**
     * Returns a pointer to the front of the internal buffer
     * @note If the vector is resized the pointer will become invalid
     */
    virtual void* getDataPtr() = 0;
    /**
     * Returns a const pointer to the front of the internal buffer
     * @note If the vector is resized the pointer will become invalid
     */
    virtual const void* getReadOnlyDataPtr() const = 0;
    /**
     * Returns a copy of the vector with the same contents
     */
    virtual GenericMemoryVector* clone() const = 0;
    /**
     * Resize the buffer to hold t items
     * @param t The size of the buffer (in terms of items, not bytes)
     */
    virtual void resize(unsigned int t) = 0;
};

/**
 * Storage class for host copies of variable buffers
 * @tparam T The base-type of the data in the vector
 */
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
    /**
     * Default destructor
     */
    virtual ~MemoryVector() { ; }
    /**
     * Returns the type index of the base type of an element.
     */
    const std::type_index& getType() const override {
        return type;
    }
    /**
     * Returns the number of elements in a variable
     * This is likely to be 1 unless the variable is an array variable
     * @see getTypeSize()
     * @see getVariableSize()
     */
    unsigned int getElements() const override {
        return elements;
    }
    /**
     * Returns the size (in bytes) of a base type of an element of the MemoryVector
     * e.g. = type size = variable size / elements
     * @see getElements()
     * @see getVariableSize()
     */
    size_t getTypeSize() const override {
        return type_size;
    }
    /**
     * Returns the size (in bytes) of a single element of the MemoryVector.
     * e.g. variable size = type size * elements
     * @see getElements()
     * @see getTypeSize()
     */
    size_t getVariableSize() const override {
        return type_size * elements;
    }
    /**
     * Returns a pointer to the front of the internal buffer
     * @note If the vector is resized the pointer will become invalid
     */
    void* getDataPtr() override {
        if (vec.empty())
            return nullptr;
        else
            return &(vec.front());
    }
    /**
     * Returns a const pointer to the front of the internal buffer
     * @note If the vector is resized the pointer will become invalid
     */
    const void* getReadOnlyDataPtr() const override {
        if (vec.empty())
            return nullptr;
        else
            return &(vec.front());
    }
    /**
     * Returns a copy of the vector with the same contents
     */
    MemoryVector<T>* clone() const override {
        return (new MemoryVector<T>(elements));
    }
    /**
     * Resize the buffer to hold s items
     * @param s The size of the buffer (in terms of items, not bytes)
     */
    void resize(unsigned int s) override {
        vec.resize(s * elements);
    }

 protected:
    /**
     * Number of elements per variable (1 if not an array variable)
     * Multiplies with size to allocate enough memory
     */
    const unsigned int elements;
    /**
     * Vector which manages data storage
     */
    std::vector<T> vec;
    /**
     * Type info about the vector base type
     * %typeid(T)
     */
    const std::type_index type;
    /**
     * Size of the base type of a variable
     * e.g. not accounting for the number of elements in the variable
     * %sizeof(T)
     */
    const size_t type_size;
};

// use this to store default values for a population, must be here to register the correct types at compile time
/*! Create a map with std::strings for keys (indexes) and GenericAgentMemoryVector object. A smart pointer has been used to automatically manage the object*/
typedef std::map<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMap;

/*! Create a pair with std::strings for keys (indexes) and GenericAgentMemoryVector object.  A smart pointer has been used to automatically manage the object*/
typedef std::pair<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMapPair;

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_POP_MEMORYVECTOR_H_

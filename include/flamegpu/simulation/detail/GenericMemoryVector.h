#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_GENERICMEMORYVECTOR_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_GENERICMEMORYVECTOR_H_

#include <vector>
#include <typeindex>
#include <map>
#include <memory>
#include <utility>
#include <string>

#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {
namespace detail {

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

// use this to store default values for a population, must be here to register the correct types at compile time
/*! Create a map with std::strings for keys (indexes) and GenericAgentMemoryVector object. A smart pointer has been used to automatically manage the object*/
typedef std::map<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMap;

/*! Create a pair with std::strings for keys (indexes) and GenericAgentMemoryVector object.  A smart pointer has been used to automatically manage the object*/
typedef std::pair<const std::string, std::unique_ptr<GenericMemoryVector>> StateMemoryMapPair;

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_GENERICMEMORYVECTOR_H_

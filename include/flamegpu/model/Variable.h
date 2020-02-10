#ifndef INCLUDE_FLAMEGPU_MODEL_VARIABLE_H_
#define INCLUDE_FLAMEGPU_MODEL_VARIABLE_H_

#include <typeindex>
#include <memory>
#include <unordered_map>
#include <cassert>
#include <string>

#include "flamegpu/pop/MemoryVector.h"

/**
* Common variable definition type
* Used internally by AgentData and MessageData
*/
struct Variable {
    /**
      * Constructs a new variable
      * @param _elements The number of elements, this will be 1 unless the variable is an array
      * @tparam T The type of the variable, it's size and std::type_index are derived from this
      * @note Cannot explicitly specify template args of constructor, so we take redundant arg for implicit template
      */
    template<typename T>
    Variable(unsigned int _elements, T)
        : type(typeid(T)), type_size(sizeof(T)), elements(_elements), memory_vector(new MemoryVector<T>()) {
        assert(_elements > 0);  // This should be enforced with static_assert where Variable's are defined, see MessageDescription::newVariable()
        // Limited to Arithmetic types
        // Compound types would allow host pointers inside structs to be passed
        static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
            "Only arithmetic types can be used as environmental properties");
    }
    /**
     * Unique identifier of the variables type as returned by std::type_index(typeid())
     */
    const std::type_index type;
    /**
     * Size of the type in bytes as returned by sizeof() (e.g. float == 4 bytes)
     */
    const size_t type_size;
    /**
     * The number of elements, this will be 1 unless the variable is an array
     */
    const unsigned int elements;
    /**
     * Holds the variables memory vector type so we can dynamically create them with clone()
     */
    const std::unique_ptr<GenericMemoryVector> memory_vector;
    /**
     * Copy constructor
     */
    Variable(const Variable &other)
        :type(other.type), type_size(other.type_size), elements(other.elements), memory_vector(other.memory_vector->clone()) { }
};
/**
 * Map of name:variable definition
 * map<string, Variable>
 */
typedef std::unordered_map<std::string, Variable> VariableMap;

#endif  // INCLUDE_FLAMEGPU_MODEL_VARIABLE_H_

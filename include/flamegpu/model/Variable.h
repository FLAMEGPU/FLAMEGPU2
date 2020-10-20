#ifndef INCLUDE_FLAMEGPU_MODEL_VARIABLE_H_
#define INCLUDE_FLAMEGPU_MODEL_VARIABLE_H_

#include <typeindex>
#include <memory>
#include <map>
#include <cassert>
#include <string>
#include <cstring>
#include <vector>

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
      * @note This constructor does not set default value
      */
    template<typename T>
    Variable(unsigned int _elements, const T)
        : type(typeid(T))
        , type_size(sizeof(T))
        , elements(_elements)
        , memory_vector(new MemoryVector<T>(_elements))
        , default_value(nullptr) {
        assert(_elements > 0);  // This should be enforced with static_assert where Variable's are defined, see MessageDescription::newVariable()
        // Limited to Arithmetic types
        // Compound types would allow host pointers inside structs to be passed
        static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
            "Only arithmetic types can be used");
    }
    template<typename T, std::size_t N>
    explicit Variable(const std::array<T, N> &_default_value)
        : type(typeid(T))
        , type_size(sizeof(T))
        , elements(N)
        , memory_vector(new MemoryVector<T>(N))
        , default_value(malloc(sizeof(T) * N)) {
        assert(N > 0);  // This should be enforced with static_assert where Variable's are defined, see MessageDescription::newVariable()
        // Limited to Arithmetic types
        // Compound types would allow host pointers inside structs to be passed
        static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
            "Only arithmetic types can be used");
        memcpy(default_value, _default_value.data(), sizeof(T) * N);
    }
    template<typename T>
    explicit Variable(const unsigned int &N, const std::vector<T> &_default_value)
        : type(typeid(T))
        , type_size(sizeof(T))
        , elements(N)
        , memory_vector(new MemoryVector<T>(N))
        , default_value(malloc(sizeof(T) * N)) {
        assert(N > 0);  // This should be enforced with static_assert where Variable's are defined, see MessageDescription::newVariable()
        assert(N  ==  _default_value.size());  // This should be enforced where variables are defined
        // Limited to Arithmetic types
        // Compound types would allow host pointers inside structs to be passed
        static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
            "Only arithmetic types can be used");
        memcpy(default_value, _default_value.data(), sizeof(T) * N);
    }
    /**
     * Destructor, frees memory
     */
    ~Variable() {
        if (default_value)
            free(default_value);
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
     * Default value for new agents
     */
    void * const default_value;
    /**
     * Copy constructor
     */
    Variable(const Variable &other)
        : type(other.type)
        , type_size(other.type_size)
        , elements(other.elements)
        , memory_vector(other.memory_vector->clone())
        , default_value(other.default_value ? malloc(type_size * elements) : nullptr) {
        if (default_value)
            memcpy(default_value, other.default_value, type_size * elements);
    }
};
/**
 * Map of name:variable definition
 * map<string, Variable>
 * map (rather than unordered_map) is used here intentionally, as device agent birth relies on iteration order not changing. 
 */
typedef std::map<std::string, Variable> VariableMap;

#endif  // INCLUDE_FLAMEGPU_MODEL_VARIABLE_H_

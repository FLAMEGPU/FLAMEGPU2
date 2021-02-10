#ifndef INCLUDE_FLAMEGPU_UTIL_ANY_H_
#define INCLUDE_FLAMEGPU_UTIL_ANY_H_

/**
 * Minimal std::any replacement, works pre c++17
 */
struct Any {
    /**
     * Constructor
     * @param _ptr Pointer to data to represent
     * @param _length Length of pointed to data
     * @param _type Identifier of type
     * @param _elements How many elements does the property have (1 if it's not an array)
     * @note Copies the data
     */
    Any(const void *_ptr, const size_t &_length, const std::type_index &_type, const unsigned int &_elements)
        : ptr(malloc(_length))
        , length(_length)
        , type(_type)
        , elements(_elements) {
        memcpy(ptr, _ptr, length);
    }
    template<typename T>
    explicit Any(const T&other)
        : ptr(malloc(sizeof(T)))
        , length(sizeof(T))
        , type(typeid(T))
        , elements(1) {
        memcpy(ptr, &other, sizeof(T));
    }
    /**
     * Copy constructor
     * @param _other Other Any to be cloned, makes a copy of the pointed to data
     */
    Any(const Any &_other)
        : ptr(malloc(_other.length))
        , length(_other.length)
        , type(_other.type)
        , elements(_other.elements) {
        memcpy(ptr, _other.ptr, length);
    }
    /*
     * Releases the allocated memory
     */
    ~Any() {
        free(ptr);
    }
    /**
     * Can't assign, members are const at creation
     */
    void operator=(const Any &_other) = delete;
    /**
     * Data represented by this object
     */
    void *const ptr;
    /**
     * Length of memory allocation pointed to by ptr
     */
    const size_t length;
    /**
     * Type index to unwrap the value
     */
    const std::type_index type;
    /**
     * Number of elements (1 if not array)
     */
    const unsigned int elements;
};

#endif  // INCLUDE_FLAMEGPU_UTIL_ANY_H_

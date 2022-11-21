#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY_MESSAGEARRAYDEVICE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY_MESSAGEARRAYDEVICE_CUH_

#include "flamegpu/runtime/messaging/MessageArray.h"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"

namespace flamegpu {

/**
 * This class is accessible via DeviceAPI.message_in if MessageArray is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading array messages
 */
class MessageArray::In {
    /**
     * Message has full access to In, they are treated as the same class so share everything
     * Reduces/memory data duplication
     */
    friend class Message;

 public:
    /**
    * Provides access to a specific message
    * Returned by In::at(size_type)
    * @see In::at(size_type)
    */
    class Message {
        /**
        * Paired In class which created the iterator
        */
        const MessageArray::In &_parent;
        /**
        * Position within the message list
        */
        size_type index;

     public:
        /**
        * Constructs a message and directly initialises all of it's member variables
        * index is always init to 0
        * @note See member variable documentation for their purposes
        */
        __device__ Message(const MessageArray::In &parent, const size_type _index) : _parent(parent), index(_index) {}
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        /**
         * A null message which always returns the message at index 0
         */
        __device__ Message(const MessageArray::In& parent) : _parent(parent), index(0) {}
#endif
        /**
        * Equality operator
        * Compares all internal member vars for equality
        * @note Does not compare _parent
        */
        __device__ bool operator==(const Message& rhs) const { return  this->index == rhs.index; }
        /**
        * Inequality operator
        * Returns inverse of equality operator
        * @see operator==(const Message&)
        */
        __device__ bool operator!=(const Message& rhs) const { return  this->index != rhs.index; }
        /**
        * Returns the index of the message within the full message list
        */
        __device__ size_type getIndex() const { return this->index; }
        /**
        * Returns the value for the current message attached to the named variable
        * @param variable_name Name of the variable
        * @tparam T type of the variable
        * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
        * @return The specified variable, else 0x0 if an error occurs
        */
        template<typename T, unsigned int N>
        __device__ T getVariable(const char(&variable_name)[N]) const;
        /**
         * Returns the specified variable array element from the current message attached to the named variable
         * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
         * @param index Index of the element within the variable array to return
         * @tparam T Type of the message variable being accessed
         * @tparam N The length of the array variable, as set within the model description hierarchy
         * @tparam M Length of variable_name, this should always be implicit if passing a string literal
         * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
         */
        template<typename T, flamegpu::size_type N, unsigned int M>
        __device__ T getVariable(const char(&variable_name)[M], unsigned int index) const;
    };
    /**
     * This class is created when a search origin is provided to MessageArray::In::operator()(size_type, size_type, size_type = 1)
     * It provides iterator access to a subset of the full message list, according to the provided search origin and radius
     *
     * @see MessageArray::In::wrap(size_type, size_type)
     */
    class WrapFilter {
        /**
         * Message has full access to WrapFilter, they are treated as the same class so share everything
         * Reduces/memory data duplication
         */
        friend class Message;

     public:
        /**
         * Provides access to a specific message
         * Returned by the iterator
         * @see In::WrapFilter::iterator
         */
        class Message {
            /**
             * Paired WrapFilter class which created the iterator
             */
            const WrapFilter&_parent;
            /**
             * Relative position within the Moore neighbourhood
             * This is initialised based on user provided radius
             */
            int relative_cell;
            /**
             * Index into memory of currently pointed message
             */
            size_type index_1d = 0;

         public:
            /**
             * Constructs a message and directly initialises all of it's member variables
             * @note See member variable documentation for their purposes
             */
            __device__ Message(const WrapFilter&parent, const int relative_x)
                : _parent(parent) {
                relative_cell = relative_x;
            }
            /**
             * Equality operator
             * Compares all internal member vars for equality
             * @note Does not compare _parent
             */
            __device__ bool operator==(const Message& rhs) const {
                return this->relative_cell == rhs.relative_cell;
                // && this->_parent.loc == rhs._parent.loc;
            }
            /**
             * Inequality operator
             * Returns inverse of equality operator
             * @see operator==(const Message&)
             */
            __device__ bool operator!=(const Message& rhs) const { return !(*this == rhs); }
            /**
             * Updates the message to return variables from the next cell in the Moore neighbourhood
             * @return Returns itself
             */
            __device__ inline Message& operator++();
            /**
             * Returns x array index of message
             */
            __device__ size_type getX() const {
                return (this->_parent.loc + relative_cell + this->_parent.length) % this->_parent.length;
            }
            /**
             * Returns the x array offset of message relative to the search origin
             * @note This value is unwrapped, so will always return a value within the search radius
             */
            __device__ size_type getOffsetX() const {
                return relative_cell;
            }
            /**
             * Returns the value for the current message attached to the named variable
             * @param variable_name Name of the variable
             * @tparam T type of the variable
             * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
             * @return The specified variable, else 0x0 if an error occurs
             */
            template<typename T, unsigned int N>
            __device__ T getVariable(const char(&variable_name)[N]) const;
            /**
             * Returns the specified variable array element from the current message attached to the named variable
             * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
             * @param index Index of the element within the variable array to return
             * @tparam T Type of the message variable being accessed
             * @tparam N The length of the array variable, as set within the model description hierarchy
             * @tparam M Length of variable_name, this should always be implicit if passing a string literal
             * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
             */
            template<typename T, flamegpu::size_type N, unsigned int M> __device__
            T getVariable(const char(&variable_name)[M], unsigned int index) const;
        };
        /**
         * Stock iterator for iterating MessageSpatial3D::In::WrapFilter::Message objects
         */
        class iterator {
            /**
             * The message returned to the user
             */
            Message _message;

         public:
            /**
             * Constructor
             * This iterator is constructed by MessageArray::In::WrapFilter::begin()(size_type, size_type)
             * @see MessageArray::In::wrap(size_type, size_type)
             */
            __device__ iterator(const WrapFilter&parent, const int relative_x)
                : _message(parent, relative_x) {
                // Increment to find first message
                ++_message;
            }
            /**
             * Moves to the next message
             * (Prefix increment operator)
             */
            __device__ iterator& operator++() { ++_message;  return *this; }
            /**
             * Moves to the next message
             * (Postfix increment operator, returns value prior to increment)
             */
            __device__ iterator operator++(int) {
                iterator temp = *this;
                ++*this;
                return temp;
            }
            /**
             * Equality operator
             * Compares message
             */
            __device__ bool operator==(const iterator& rhs) const { return  _message == rhs._message; }
            /**
             * Inequality operator
             * Compares message
             */
            __device__ bool operator!=(const iterator& rhs) const { return  _message != rhs._message; }
            /**
             * Dereferences the iterator to return the message object, for accessing variables
             */
            __device__ Message& operator*() { return _message; }
            /**
             * Dereferences the iterator to return the message object, for accessing variables
             */
            __device__ Message* operator->() { return &_message; }
        };
        /**
         * Constructor, takes the search parameters required
         * @param _length Pointer to message list length
         * @param x Search origin x coord
         * @param _radius Search radius
         */
        inline __device__ WrapFilter(const size_type _length, const size_type x, const size_type _radius);
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        /**
         * A null filter which always returns 0 messages
         */
        inline __device__ WrapFilter();
#endif
        /**
         * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
            if (!this->length)
                return iterator(*this, radius);
#endif
            // Bin before initial bin, as the constructor calls increment operator
            return iterator(*this, -static_cast<int>(radius) - 1);
        }
        /**
         * Returns an iterator to the position beyond the end of the message list subset
         * @note This iterator is the same for all message list subsets
         */
        inline __device__ iterator end(void) const {
            // Final bin, as the constructor calls increment operator
            return iterator(*this, radius);
        }

     private:
        /**
         * Search origin
         */
        size_type loc;
        /**
         * Search radius
         */
        const size_type radius;
        /**
         * Message list length
         */
        const size_type length;
    };
    /**
     * This class is created when a search origin is provided to MessageArray::In::operator()(size_type, size_type, size_type = 1)
     * It provides iterator access to a subset of the full message list, according to the provided search origin and radius
     * The radius does not wrap the message list bounds
     *
     * @see MessageArray::In::operator()(size_type, size_type)
     */
    class Filter {
        /**
         * Message has full access to Filter, they are treated as the same class so share everything
         * Reduces/memory data duplication
         */
        friend class Message;

     public:
        /**
         * Provides access to a specific message
         * Returned by the iterator
         * @see In::Filter::iterator
         */
        class Message {
            /**
             * Paired Filter class which created the iterator
             */
            const Filter& _parent;
            /**
             * Relative position within the Moore neighbourhood
             * This is initialised based on user provided radius
             */
            int relative_cell;
            /**
             * Index into memory of currently pointed message
             */
            size_type index_1d = 0;

         public:
            /**
             * Constructs a message and directly initialises all of it's member variables
             * @note See member variable documentation for their purposes
             */
            __device__ Message(const Filter& parent, const int relative_x)
                : _parent(parent) {
                relative_cell = relative_x;
            }
            /**
             * Equality operator
             * Compares all internal member vars for equality
             * @note Does not compare _parent
             */
            __device__ bool operator==(const Message& rhs) const {
                return this->relative_cell == rhs.relative_cell;
                    // && this->_parent.loc == rhs._parent.loc;
            }
            /**
             * Inequality operator
             * Returns inverse of equality operator
             * @see operator==(const Message&)
             */
            __device__ bool operator!=(const Message& rhs) const { return !(*this == rhs); }
            /**
             * Updates the message to return variables from the next cell in the Moore neighbourhood
             * @return Returns itself
             */
            __device__ inline Message& operator++();
            /**
             * Returns x array index of message
             */
            __device__ size_type getX() const {
                return this->_parent.loc + relative_cell;
            }
            /**
             * Returns the x array offset of message relative to the search origin
             */
            __device__ int getOffsetX() const {
                return this->relative_cell;
            }
            /**
             * Returns the value for the current message attached to the named variable
             * @param variable_name Name of the variable
             * @tparam T type of the variable
             * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
             * @return The specified variable, else 0x0 if an error occurs
             */
            template<typename T, unsigned int N>
            __device__ T getVariable(const char(&variable_name)[N]) const;
            /**
             * Returns the specified variable array element from the current message attached to the named variable
             * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
             * @param index Index of the element within the variable array to return
             * @tparam T Type of the message variable being accessed
             * @tparam N The length of the array variable, as set within the model description hierarchy
             * @tparam M Length of variable_name, this should always be implicit if passing a string literal
             * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
             */
            template<typename T, flamegpu::size_type N, unsigned int M>
            __device__ T getVariable(const char(&variable_name)[M], unsigned int index) const;
        };
        /**
         * Stock iterator for iterating MessageSpatial3D::In::Filter::Message objects
         */
        class iterator {
            /**
             * The message returned to the user
             */
            Message _message;

         public:
            /**
             * Constructor
             * This iterator is constructed by MessageArray::In::Filter::begin()(size_type, size_type)
             * @see MessageArray::In::Operator()(size_type, size_type)
             */
            __device__ iterator(const Filter& parent, const int relative_x)
                : _message(parent, relative_x) {
                // Increment to find first message
                ++_message;
            }
            /**
             * Moves to the next message
             * (Prefix increment operator)
             */
            __device__ iterator& operator++() { ++_message;  return *this; }
            /**
             * Moves to the next message
             * (Postfix increment operator, returns value prior to increment)
             */
            __device__ iterator operator++(int) {
                iterator temp = *this;
                ++* this;
                return temp;
            }
            /**
             * Equality operator
             * Compares message
             */
            __device__ bool operator==(const iterator& rhs) const { return  _message == rhs._message; }
            /**
             * Inequality operator
             * Compares message
             */
            __device__ bool operator!=(const iterator& rhs) const { return  _message != rhs._message; }
            /**
             * Dereferences the iterator to return the message object, for accessing variables
             */
            __device__ Message& operator*() { return _message; }
            /**
             * Dereferences the iterator to return the message object, for accessing variables
             */
            __device__ Message* operator->() { return &_message; }
        };
        /**
         * Constructor, takes the search parameters required
         * @param _length Pointer to message list length
         * @param x Search origin x coord
         * @param _radius Search radius
         */

        inline __device__ Filter(size_type _length, size_type x, size_type _radius);
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        /**
         * A null filter which always returns 0 messages
         */
        inline __device__ Filter();
#endif
        /**
         * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
            // Bin before initial bin, as the constructor calls increment operator
            return iterator(*this, min_cell - 1);
        }
        /**
         * Returns an iterator to the position beyond the end of the message list subset
         * @note This iterator is the same for all message list subsets
         */
        inline __device__ iterator end(void) const {
            // Final bin, as the constructor calls increment operator
            return iterator(*this, max_cell);
        }

     private:
        /**
         * Search origin
         */
        size_type loc;
        /**
         * Min offset to be accessed (inclusive)
         */
        int min_cell;
        /**
         * Max offset to be accessed (inclusive)
         */
        int max_cell;
        /**
         * Message list length
         */
        const size_type length;
    };
    /**
     * Constructer
     * Initialises member variables
     * @param metadata Reinterpreted as type MessageArray::MetaData to extract length
     */
    __device__ In(const void *metadata)
        : length(reinterpret_cast<const MetaData*>(metadata)->length)
    { }
    /**
     * Returns a Filter object which provides access to message iterator
     * for iterating a subset of messages including those within the radius of the search origin
     * this excludes the message at the search origin
     * The radius will wrap over environment bounds
     *
     * @param x Search origin x coord
     * @param radius Search radius
     * @note radius 1 is 2 cells
     * @note radius 2 is 4 cells
     * @note radius which produce a message read dimension (radius*2 + 1) greater than the length of the messagelist are unsupported
     * @note radius of 0 is unsupported
     * @note The location x must be within the bounds of the message list
     */
    inline __device__ WrapFilter wrap(const size_type x, const size_type radius = 1) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if (radius == 0) {
            DTHROW("Invalid radius %u for accessing array messagelist of length %u\n", radius, length);
            return WrapFilter();
        } else if ((radius * 2) + 1 > length) {
            unsigned int min_r = length % 2 == 0 ? length - 2 : length - 1;
            min_r /= 2;
            DTHROW("%u is not a valid radius for accessing Array message lists, as the diameter of messages accessed exceeds the message list length (%u)."
                " Maximum supported radius for this message list is %u.\n",
                radius, length, min_r);
            return WrapFilter();
        } else if (x >= length) {
            DTHROW("%u is not a valid position for iterating an Array message list of length %u, location must be within bounds.",
                x, length);
            return WrapFilter();
        }
#endif
        return WrapFilter(length, x, radius);
    }
    /**
     * Returns a Filter object which provides access to message iterator
     * for iterating a subset of messages including those within the radius of the search origin
     * this excludes the message at the search origin
     * The radius will not wrap over environment bounds
     *
     * @param x Search origin x coord
     * @param radius Search radius
     * @note radius 1 is 2 cells in 3
     * @note radius 2 is 4 cells in 5
     * @note radius of 0 is unsupported
     * @note The location x must be within the bounds of the message list
     */
    inline __device__ Filter operator() (const size_type x, const size_type radius = 1) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if (radius == 0) {
            DTHROW("Invalid radius %u for accessing array messagelist of length %u\n", radius, length);
            return Filter();
        } else if (x >= length) {
            DTHROW("%u is not a valid position for iterating an Array message list of length %u, location must be within bounds.",
                x, length);
            return Filter();
        }
#endif
        return Filter(length, x, radius);
    }
    /**
     * Returns the length of the message list.
     */
    __device__ size_type size(void) const {
        return length;
    }
    __device__ Message at(const size_type index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if (index >= length) {
            DTHROW("Index is out of bounds for Array messagelist (%u >= %u).\n", index, length);
            return Message(*this);
        }
#endif
        return Message(*this, index);
    }

 private:
    /**
     * Metadata struct for accessing messages
     */
    const size_type length;
};
/**
 * This class is accessible via DeviceAPI.message_out if MessageArray is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting array messages
 */
class MessageArray::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param _metadata Message specialisation specific metadata struct (of type MessageArray::MetaData)
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(const void *_metadata, unsigned int *scan_flag_messageOutput)
        : scan_flag(scan_flag_messageOutput)
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
#else
        , metadata(nullptr)
#endif
    { }
    /**
     * Sets the array index to store the message in
     */
    __device__ inline void setIndex(const size_type id) const;
    /**
     * Sets the specified variable for this agents message
     * @param variable_name Name of the variable
     * @param value The value to set the specified variable
     * @tparam T type of the variable
     * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
     * @return The specified variable, else 0x0 if an error occurs
     */
    template<typename T, unsigned int N>
    __device__ void setVariable(const char(&variable_name)[N], T value) const;
    /**
     * Sets an element of an array variable for this agents message
     * @param variable_name The name of the array variable
     * @param index The index to set within the array variable
     * @param value The value to set the element of the array element
     * @tparam T The type of the variable, as set within the model description hierarchy
     * @tparam N The length of the array variable, as set within the model description hierarchy
     * @tparam M variable_name length, this should be ignored as it is implicitly set
     * @throws exception::DeviceError If name is not a valid variable within the message (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N, unsigned int M>
    __device__ void setVariable(const char(&variable_name)[M], unsigned int index, T value) const;

 protected:
    /**
     * Scan flag array for optional message output
     */
    unsigned int *scan_flag;
    /**
     * Metadata struct for accessing messages
     */
    const MetaData * const metadata;
};

template<typename T, unsigned int N>
__device__ T MessageArray::In::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    return detail::curve::DeviceCurve::getMessageVariable<T>(variable_name, index);
}
template<typename T, flamegpu::size_type N, unsigned int M> __device__
T MessageArray::In::Message::getVariable(const char(&variable_name)[M], const unsigned int array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getMessageArrayVariable<T, N>(variable_name, index, array_index);
    return value;
}
template<typename T, unsigned int N>
__device__ T MessageArray::In::WrapFilter::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    return detail::curve::DeviceCurve::getMessageVariable<T>(variable_name, index_1d);
}
template<typename T, flamegpu::size_type N, unsigned int M> __device__
T MessageArray::In::WrapFilter::Message::getVariable(const char(&variable_name)[M], const unsigned int array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getMessageArrayVariable<T, N>(variable_name, index_1d, array_index);
    return value;
}
template<typename T, unsigned int N>
__device__ T MessageArray::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    return detail::curve::DeviceCurve::getMessageVariable<T>(variable_name, index_1d);
}
template<typename T, flamegpu::size_type N, unsigned int M> __device__
T MessageArray::In::Filter::Message::getVariable(const char(&variable_name)[M], const unsigned int array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getMessageArrayVariable<T, N>(variable_name, index_1d, array_index);
    return value;
}

template<typename T, unsigned int N>
__device__ void MessageArray::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    if (variable_name[0] == '_') {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in MessageArray::Out::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    detail::curve::DeviceCurve::setMessageVariable<T>(variable_name, value, index);

    // setIndex() sets the optional message scan flag
}
template<typename T, unsigned int N, unsigned int M>
__device__ void MessageArray::Out::setVariable(const char(&variable_name)[M], const unsigned int array_index, T value) const {
    if (variable_name[0] == '_') {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in MessageArray::Out::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    detail::curve::DeviceCurve::setMessageArrayVariable<T, N>(variable_name, value, index, array_index);

    // setIndex() sets the optional message scan flag
}

/**
* Sets the array index to store the message in
*/
__device__ void MessageArray::Out::setIndex(const size_type id) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (id >= metadata->length) {
        DTHROW("MessageArray index [%u] is out of bounds [%u]\n", id, metadata->length);
    }
#endif

    // set the variable using curve
    detail::curve::DeviceCurve::setMessageVariable<size_type>("___INDEX", id, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}
__device__ MessageArray::In::WrapFilter::WrapFilter(const size_type _length, const size_type x, const size_type _radius)
    : radius(_radius)
    , length(_length) {
    loc = x;
}
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
__device__ inline MessageArray::In::WrapFilter::WrapFilter()
    : radius(0)
    , length(0) {
    loc = 0;
}
#endif
__device__ MessageArray::In::WrapFilter::Message& MessageArray::In::WrapFilter::Message::operator++() {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (!_parent.length)
        return *this;
#endif
    relative_cell++;
    // Skip origin cell
    if (relative_cell == 0) {
        relative_cell++;
    }
    // Wrap over boundaries
    index_1d = (this->_parent.loc + relative_cell + this->_parent.length) % this->_parent.length;
    return *this;
}
__device__ MessageArray::In::Filter::Filter(const size_type _length, const size_type x, const size_type _radius)
    : length(_length) {
    loc = x;
    min_cell = static_cast<int>(x) - static_cast<int>(_radius) < 0 ? -static_cast<int>(x) : -static_cast<int>(_radius);
    max_cell = x + _radius >= _length ? static_cast<int>(_length) - 1 - static_cast<int>(x) : static_cast<int>(_radius);
}
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
__device__ inline MessageArray::In::Filter::Filter()
    : length(0) {
    loc = 0;
    min_cell = 1;
    max_cell = 0;
}
#endif
__device__ MessageArray::In::Filter::Message& MessageArray::In::Filter::Message::operator++() {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (!_parent.length)
        return *this;
#endif
    relative_cell++;
    // Skip origin cell
    if (relative_cell == 0) {
        relative_cell++;
    }
    // Solve to 1 dimensional bin index
    index_1d = this->_parent.loc + relative_cell;
    return *this;
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY_MESSAGEARRAYDEVICE_CUH_

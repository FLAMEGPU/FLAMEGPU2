#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCE_MESSAGEBRUTEFORCEDEVICE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCE_MESSAGEBRUTEFORCEDEVICE_CUH_

#include "flamegpu/defines.h"
#include "flamegpu/runtime/messaging/MessageNone.h"
#include "flamegpu/runtime/messaging/MessageBruteForce.h"
#ifndef __CUDACC_RTC__
#include "flamegpu/runtime/detail/curve/DeviceCurve.cuh"
#endif  // __CUDACC_RTC__

struct ModelData;

namespace flamegpu {

/**
 * This class is accessible via DeviceAPI.message_in if MessageBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading brute force messages
 */
class MessageBruteForce::In {
    /**
     * Message has full access to In, they are treated as the same class so share everything
     * Reduces/memory data duplication
     */

 public:
    class Message;      // Forward declare inner classes
    class iterator;     // Forward declare inner classes

    /**
     * Constructor
     * Initialises member variables
     * @param metadata Reinterpreted as type MessageBruteForce::MetaData to extract length
     */
    __device__ In(const void *metadata)
        : len(reinterpret_cast<const MetaData*>(metadata)->length)
    { }
    /**
     * Returns the number of elements in the message list.
     */
    __device__ size_type size(void) const {
        return len;
    }
    /**
      * Returns an iterator to the start of the message list
      */
    __device__ iterator begin(void) const {  // const
        return iterator(*this, 0);
    }
    /**
     * Returns an iterator to the position beyond the end of the message list
     */
    __device__ iterator end(void) const  {  // const
        // If there can be many begin, each with diff end, we need a middle layer to host the iterator/s
        return iterator(*this, len);
    }

    /**
     * Provides access to a specific message
     * Returned by the iterator
     * @see In::iterator
     */
    class Message {
         /**
          * Paired In class which created the iterator
          */
        const MessageBruteForce::In &_parent;
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
        __device__ Message(const MessageBruteForce::In &parent) : _parent(parent), index(0) {}
        /**
         * Alternate constructor, allows index to be manually set
         * @note I think this is unused
         */
        __device__ Message(const MessageBruteForce::In &parent, size_type index) : _parent(parent), index(index) {}
        /**
         * Equality operator
         * Compares all internal member vars for equality
         * @note Does not compare _parent
         */
        __host__ __device__ bool operator==(const Message& rhs) const { return  this->getIndex() == rhs.getIndex(); }
        /**
         * Inequality operator
         * Returns inverse of equality operator
         * @see operator==(const Message&)
         */
        __host__ __device__ bool operator!=(const Message& rhs) const { return  this->getIndex() != rhs.getIndex(); }
        /**
         * Updates the message to return variables from the next message in the message list
         * @return Returns itself
         */
        __host__ __device__ Message& operator++() { ++index;  return *this; }
        /**
         * Returns the index of the message within the full message list
         */
        __host__ __device__ size_type getIndex() const { return this->index; }
        /**
         * Returns the value for the current message attached to the named variable
         * @param variable_name Name of the variable
         * @tparam T type of the variable
         * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
         * @return The specified variable, else 0x0 if an error occurs
         * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
         */
        template<typename T, unsigned int N> __device__
        T getVariable(const char(&variable_name)[N]) const;
        /**
         * Returns the specified variable array element from the current message attached to the named variable
         * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
         * @param index Index of the element within the variable array to return
         * @tparam T Type of the message variable being accessed
         * @tparam N The length of the array variable, as set within the model description hierarchy
         * @tparam M Length of variable_name, this should always be implicit if passing a string literal
         * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
         */
        template<typename T, MessageNone::size_type N, unsigned int M> __device__
        T getVariable(const char(&variable_name)[M], const unsigned int &index) const;
    };

    /**
    * Stock iterator for iterating MessageBruteForce::In::Message objects
    */
    class iterator {
        /**
         * The message returned to the user
         */
         Message _message;

     public:
        /**
         * Constructor
         * This iterator is constructed by MessageBruteForce::begin()
         * @see MessageBruteForce::begin()
         */
        __device__ iterator(const In &parent, size_type index) : _message(parent, index) {}
        /**
         * Moves to the next message
         */
        __device__ iterator& operator++() { ++_message;  return *this; }
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
        __device__  Message& operator*() { return _message; }
    };

 private:
    /**
     * Total number of messages in the message list
     */
    size_type len;
};



/**
 * This class is accessible via DeviceAPI.message_out if MessageBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting brute force messages
 */
class MessageBruteForce::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(const void *, unsigned int *scan_flag_messageOutput)
        : scan_flag(scan_flag_messageOutput)
    { }
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
     * @throws exception::DeviceError If name is not a valid variable within the message (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N, unsigned int M>
    __device__ void setVariable(const char(&variable_name)[M], const unsigned int& index, T value) const;

 protected:
    /**
     * Scan flag array for optional message output
     */
    unsigned int *scan_flag;
};

template<typename T, unsigned int N>
__device__ T MessageBruteForce::In::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.len) {
        DTHROW("Brute force message index exceeds messagelist length, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
#ifdef USE_GLM
    T value = detail::curve::DeviceCurve::getMessageVariable<T>(variable_name, index);
#else
    T value = detail::curve::DeviceCurve::getMessageVariable_ldg<T>(variable_name, index);
#endif
    return value;
}
template<typename T, MessageNone::size_type N, unsigned int M> __device__
T MessageBruteForce::In::Message::getVariable(const char(&variable_name)[M], const unsigned int& array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.len) {
        DTHROW("Brute force message index exceeds messagelist length, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
#ifdef USE_GLM
    T value = detail::curve::DeviceCurve::getMessageArrayVariable<T, N>(variable_name, index, array_index);
#else
    T value = detail::curve::DeviceCurve::getMessageArrayVariable_ldg<T, N>(variable_name, index, array_index);
#endif
    return value;
}

template<typename T, unsigned int N>
__device__ void MessageBruteForce::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in MessageBruteForce::Out::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    detail::curve::DeviceCurve::setMessageVariable<T>(variable_name, value, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}
template<typename T, unsigned int N, unsigned int M>
__device__ void MessageBruteForce::Out::setVariable(const char(&variable_name)[M], const unsigned int& array_index, T value) const {
    if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in MessageBruteForce::Out::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    detail::curve::DeviceCurve::setMessageArrayVariable<T, N>(variable_name, value, index, array_index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCE_MESSAGEBRUTEFORCEDEVICE_CUH_

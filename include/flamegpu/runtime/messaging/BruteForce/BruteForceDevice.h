#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_BRUTEFORCEDEVICE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_BRUTEFORCEDEVICE_H_

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"

struct ModelData;

namespace flamegpu {

/**
 * This class is accessible via DeviceAPI.message_in if MsgBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading brute force messages
 */
class MsgBruteForce::In {
    /**
     * Message has full access to In, they are treated as the same class so share everything
     * Reduces/memory data duplication
     */

 public:
    class Message;      // Forward declare inner classes
    class iterator;     // Forward declare inner classes

    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to msg_hash to produce combined_hash
     * @param msg_hash Added to agentfn_hash to produce combined_hash
     * @param metadata Reinterpreted as type MsgBruteForce::MetaData to extract length
     */
    __device__ In(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash msg_hash, const void *metadata)
        : combined_hash(agentfn_hash + msg_hash)
        , len(reinterpret_cast<const MetaData*>(metadata)->length)
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
        const MsgBruteForce::In &_parent;
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
        __device__ Message(const MsgBruteForce::In &parent) : _parent(parent), index(0) {}
        /**
         * Alternate constructor, allows index to be manually set
         * @note I think this is unused
         */
        __device__ Message(const MsgBruteForce::In &parent, size_type index) : _parent(parent), index(index) {}
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
         */
        template<typename T, size_type N>
        __device__ T getVariable(const char(&variable_name)[N]) const;
    };

    /**
    * Stock iterator for iterating MsgBruteForce::In::Message objects
    */
    class iterator {
        /**
         * The message returned to the user
         */
         Message _message;

     public:
        /**
         * Constructor
         * This iterator is constructed by MsgBruteForce::begin()
         * @see MsgBruteForce::begin()
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
      * CURVE hash for accessing message data
      * agent function hash + message hash
      */
    detail::curve::Curve::NamespaceHash combined_hash;
    /**
     * Total number of messages in the message list
     */
    size_type len;
};



/**
 * This class is accessible via DeviceAPI.message_out if MsgBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting brute force messages
 */
class MsgBruteForce::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to msg_hash to produce combined_hash
     * @param msg_hash Added to agentfn_hash to produce combined_hash
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash msg_hash, const void *, unsigned int *scan_flag_messageOutput)
        : combined_hash(agentfn_hash + msg_hash)
        , scan_flag(scan_flag_messageOutput)
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

 protected:
    /**
     * CURVE hash for accessing message data
     * agentfn_hash + msg_hash
     */
    detail::curve::Curve::NamespaceHash combined_hash;
    /**
     * Scan flag array for optional message output
     */
    unsigned int *scan_flag;
};

template<typename T, unsigned int N>
__device__ T MsgBruteForce::In::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.len) {
        DTHROW("Brute force message index exceeds messagelist length, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    T value = detail::curve::Curve::getMessageVariable_ldg<T>(variable_name, this->_parent.combined_hash, index);
    return value;
}

template<typename T, unsigned int N>
__device__ void MsgBruteForce::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    if (variable_name[0] == '_') {
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    detail::curve::Curve::setMessageVariable<T>(variable_name, combined_hash, value, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_BRUTEFORCEDEVICE_H_

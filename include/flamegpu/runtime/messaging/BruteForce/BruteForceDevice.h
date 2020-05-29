#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCEDEVICE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCEDEVICE_H_

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"

struct ModelData;



/**
 * This class is accessible via FLAMEGPU_DEVICE_API.message_in if MsgBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
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
	__device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *metadata)
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
    class iterator{  // : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
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
	Curve::NamespaceHash combined_hash;
	/**
	 * Total number of messages in the message list
	 */
	size_type len;
};



/**
 * This class is accessible via FLAMEGPU_DEVICE_API.message_out if MsgBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting brute force messages
 */
class MsgBruteForce::Out {
 public:
	/**
	 * Constructer
	 * Initialises member variables
	 * @param agentfn_hash Added to msg_hash to produce combined_hash
	 * @param msg_hash Added to agentfn_hash to produce combined_hash
	 * @param _streamId Stream index, used for optional message output flag array
	 */
	__device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *, unsigned int _streamId)
		: combined_hash(agentfn_hash + msg_hash)
		, streamId(_streamId)
	{ }
	/**
	 * Sets the specified variable for this agents message
	 * @param variable_name Name of the variable
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
	Curve::NamespaceHash combined_hash;
	/**
	 * Stream index used for setting optional message output flag
	 */
	unsigned int streamId;
};

template<typename T, unsigned int N>
__device__ T MsgBruteForce::In::Message::getVariable(const char(&variable_name)[N]) const {
    // Ensure that the message is within bounds.
    if (index < this->_parent.len) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, index);
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}

/**
* \brief adds a message
* \param variable_name Name of message variable to set
* \param value Value to set it to
*/
template<typename T, unsigned int N>
__device__ void MsgBruteForce::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    if (variable_name[0] == '_') {
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    Curve::setVariable<T>(variable_name, combined_hash, value, index);

    // Set scan flag incase the message is optional
    flamegpu_internal::CUDAScanCompaction::ds_configs[flamegpu_internal::CUDAScanCompaction::MESSAGE_OUTPUT][streamId].scan_flag[index] = 1;
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCEDEVICE_H_

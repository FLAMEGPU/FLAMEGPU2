#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY_ARRAYDEVICE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY_ARRAYDEVICE_H_

#include "flamegpu/runtime/messaging/Array.h"
#include "flamegpu/runtime/messaging/BruteForce/BruteForceDevice.h"

class MsgArray::In {
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
        const MsgArray::In &_parent;
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
        __device__ Message(const MsgArray::In &parent, const size_type &_index) : _parent(parent), index(_index) {}
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
    };
    /**
     * This class is created when a search origin is provided to MsgArray::In::operator()(size_type, size_type, size_type = 1)
     * It provides iterator access to a subset of the full message list, according to the provided search origin and radius
     *
     * @see MsgArray::In::operator()(size_type, size_type)
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
            const Filter &_parent;
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
            __device__ Message(const Filter &parent, const int &relative_x)
                : _parent(parent) {
                relative_cell = relative_x;
            }
            /**
             * Equality operator
             * Compares all internal member vars for equality
             * @note Does not compare _parent
             */
            __device__ bool operator==(const Message& rhs) const {
                return this->index_1d == rhs.index_1d
                    && this->_parent.loc == rhs._parent.loc;
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
             * Returns the value for the current message attached to the named variable
             * @param variable_name Name of the variable
             * @tparam T type of the variable
             * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
             * @return The specified variable, else 0x0 if an error occurs
             */
            template<typename T, unsigned int N>
            __device__ T getVariable(const char(&variable_name)[N]) const;
        };
        /**
         * Stock iterator for iterating MsgSpatial3D::In::Filter::Message objects
         */
        class iterator {  // public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
            /**
             * The message returned to the user
             */
            Message _message;

         public:
            /**
             * Constructor
             * This iterator is constructed by MsgArray::In::Filter::begin()(size_type, size_type)
             * @see MsgArray::In::Operator()(size_type, size_type)
             */
            __device__ iterator(const Filter &parent, const int &relative_x)
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
         * Constructor, takes the search parameters requried
         * @param _length Pointer to message list length
         * @param _combined_hash agentfn+message hash for accessing message data
         * @param x Search origin x coord
         * @param _radius Search radius
         */
        __device__ inline Filter(const size_type &_length, const Curve::NamespaceHash &_combined_hash, const size_type &x, const size_type &_radius);
        /**
         * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
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
        /**
         * CURVE hash for accessing message data
         * agent function hash + message hash
         */
        Curve::NamespaceHash combined_hash;
    };
    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to msg_hash to produce combined_hash
     * @param msg_hash Added to agentfn_hash to produce combined_hash
     * @param metadata Reinterpreted as type MsgArray::MetaData to extract length
     */
    __device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *metadata)
        : combined_hash(agentfn_hash + msg_hash)
        , length(reinterpret_cast<const MetaData*>(metadata)->length)
    { }
    /**
     * Returns a Filter object which provides access to message iterator
     * for iterating a subset of messages including those within the radius of the search origin
     * this excludes the message at the search origin
     *
     * @param x Search origin x coord
     * @param radius Search radius
     * @note radius 1 is 2 cells
     * @note radius 2 is 4 cells
     * @note If radius is >= half of the array dimensions, cells will be doubly read
     * @note radius of 0 is unsupported
     */
    inline __device__ Filter operator() (const size_type &x, const size_type &radius = 1) const {
#ifndef NO_SEATBELTS
        if (radius == 0 || radius > length) {
            DTHROW("Invalid radius %llu for accessing array messaglist of length %u\n", radius, length);
        }
#endif
        return Filter(length, combined_hash, x, radius);
    }
    /**
     * Returns the length of the message list.
     */
    __device__ size_type size(void) const {
        return length;
    }
    __device__ Message at(const size_type &index) const {
#ifndef NO_SEATBELTS
        if (index >= length) {
            DTHROW("Index is out of bounds for Array messagelist (%u >= %u).\n", index, length);
        }
#endif
        return Message(*this, index);
    }

 private:
    /**
     * CURVE hash for accessing message data
     * agent function hash + message hash
     */
    Curve::NamespaceHash combined_hash;
    /**
     * Metadata struct for accessing messages
     */
    const size_type length;
};
/**
 * This class is accessible via FLAMEGPU_DEVICE_API.message_out if MsgArray is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting array messages
 */
class MsgArray::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to msg_hash to produce combined_hash
     * @param msg_hash Added to agentfn_hash to produce combined_hash
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *_metadata, unsigned int *scan_flag_messageOutput)
        : combined_hash(agentfn_hash + msg_hash)
        , scan_flag(scan_flag_messageOutput)
#ifndef NO_SEATBELTS
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
#else
        , metadata(nullptr)
#endif
    { }
    /**
     * Sets the array index to store the message in
     */
    __device__ inline void setIndex(const size_type &id) const;
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
     * Scan flag array for optional message output
     */
    unsigned int *scan_flag;
    /**
     * Metadata struct for accessing messages
     */
    const MetaData * const metadata;
};

template<typename T, unsigned int N>
__device__ T MsgArray::In::Message::getVariable(const char(&variable_name)[N]) const {
#ifndef NO_SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    return Curve::getMessageVariable<T>(variable_name, this->_parent.combined_hash, index);
}
template<typename T, unsigned int N>
__device__ T MsgArray::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
#ifndef NO_SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.length) {
        DTHROW("Invalid Array message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    return Curve::getMessageVariable<T>(variable_name, this->_parent.combined_hash, index_1d);
}

template<typename T, unsigned int N>
__device__ void MsgArray::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    if (variable_name[0] == '_') {
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    Curve::setMessageVariable<T>(variable_name, combined_hash, value, index);

    // setIndex() sets the optional msg scan flag
}

/**
* Sets the array index to store the message in
*/
__device__ void MsgArray::Out::setIndex(const size_type &id) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

#ifndef NO_SEATBELTS
    if (id >= metadata->length) {
        DTHROW("MsgArray index [%u] is out of bounds [%u]\n", id, metadata->length);
    }
#endif

    // set the variable using curve
    Curve::setMessageVariable<size_type>("___INDEX", combined_hash, id, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}
__device__ MsgArray::In::Filter::Filter(const size_type &_length, const Curve::NamespaceHash &_combined_hash, const size_type &x, const size_type &_radius)
    : radius(_radius)
    , length(_length)
    , combined_hash(_combined_hash) {
    loc = x;
}
__device__ MsgArray::In::Filter::Message& MsgArray::In::Filter::Message::operator++() {
    relative_cell++;
    // Skip origin cell
    if (relative_cell == 0) {
        relative_cell++;
    }
    // Wrap over boundaries
    index_1d = (this->_parent.loc + relative_cell + this->_parent.length) % this->_parent.length;
    return *this;
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY_ARRAYDEVICE_H_

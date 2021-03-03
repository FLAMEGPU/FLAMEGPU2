#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_BUCKETDEVICE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_BUCKETDEVICE_H_

#include "flamegpu/runtime/messaging/Bucket.h"
#include "flamegpu/runtime/messaging/BruteForce/BruteForceDevice.h"

/**
 * This class is accessible via DeviceAPI.message_in if MsgBucket is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading bucket
 */
class MsgBucket::In {
 public:
    /**
    * This class is created when a search origin is provided to MsgBucket::In::operator()(IntT)
    * It provides iterator access to the subset of messages found within the specified bucket
    *
    * @see MsgBucket::In::operator()(IntT)
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
            * This is the index of the currently accessed message, relative to the full message list
            */
            unsigned int cell_index;

         public:
            /**
            * Constructs a message and directly initialises all of it's member variables
            * @note See member variable documentation for their purposes
            */
            __device__ Message(const Filter &parent, const unsigned int &_cell_index)
                : _parent(parent)
                , cell_index(_cell_index) { }
            /**
            * Equality operator
            * Compares all internal member vars for equality
            * @note Does not compare _parent
            */
            __device__ bool operator==(const Message& rhs) const {
                return this->cell_index == rhs.cell_index;
            }
            /**
            * Inequality operator
            * Returns inverse of equality operator
            * @see operator==(const Message&)
            */
            __device__ bool operator!=(const Message& rhs) const { return !(*this == rhs); }
            /**
            * Updates the message to return variables from the next message in the message list
            * @return Returns itself
            */
            __device__ Message& operator++() { ++cell_index; return *this; }
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
        * Stock iterator for iterating MsgBucket::In::Filter::Message objects
        */
        class iterator {  // class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
                          /**
                          * The message returned to the user
                          */
            Message _message;

         public:
            /**
            * Constructor
            * This iterator is constructed by MsgBucket::In::Filter::begin()(IntT)
            * @see MsgBucket::In::Operator()(IntT)
            */
            __device__ iterator(const Filter &parent, const unsigned int &cell_index)
                : _message(parent, cell_index) {
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
        * Begin key and end key specify the [begin, end) contiguous range of bucket. (inclusive begin, exclusive end)
        * @param _metadata Pointer to message list metadata
        * @param combined_hash agentfn+message hash for accessing message data
        * @param beginKey Inclusive first bucket of range to access
        * @param endKey Exclusive final bucket of range to access, this is the final bucket + 1
        */
        inline __device__ Filter(const MetaData *_metadata, const Curve::NamespaceHash &combined_hash, const IntT &beginKey, const IntT &endKey);
        /**
        * Returns an iterator to the start of the message list subset about the search origin
        */
        inline __device__ iterator begin(void) const {
            // Bin before initial bin, as the constructor calls increment operator
            return iterator(*this, bucket_begin - 1);
        }
        /**
        * Returns an iterator to the position beyond the end of the message list subset
        * @note This iterator is the same for all message list subsets
        */
        inline __device__ iterator end(void) const {
            // Final bin, as the constructor calls increment operator
            return iterator(*this, bucket_end - 1);
        }
        /**
        * Returns the number of messages in the filtered bucket
        */
        inline __device__ unsigned int size(void) const {
            return bucket_end - bucket_begin;
        }

     private:
        /**
        * Search bucket bounds
        */
        IntT bucket_begin, bucket_end;
        /**
        * Pointer to message list metadata, e.g. environment bounds, search radius, PBM location
        */
        const MetaData *metadata;
        /**
        * CURVE hash for accessing message data
        * agent function hash + message hash
        */
        Curve::NamespaceHash combined_hash;
    };
    /**
    * Constructor
    * Initialises member variables
    * @param agentfn_hash Added to msg_hash to produce combined_hash
    * @param msg_hash Added to agentfn_hash to produce combined_hash
    * @param _metadata Reinterpreted as type MsgBucket::MetaData
    */
    __device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *_metadata)
        : combined_hash(agentfn_hash + msg_hash)
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
    { }
    /**
    * Returns a Filter object which provides access to message iterator
    * for iterating a subset of messages stored within the specified bucket
    *
    * @param key The bucket to access
    */
    inline __device__ Filter operator() (const IntT &key) const {
#if !defined(SEATBELTS) || SEATBELTS
        {
            if (key < metadata->min) {
                DTHROW("Bucket messaging iterator key %d is lower than minimum key (%d).\n", key, metadata->min);
            } else if (key >= metadata->max) {
                DTHROW("Bucket messaging iterator key %d is higher than maximum key (%d).\n", key, metadata->max - 1);
            }
        }
#endif
        return Filter(metadata, combined_hash, key, key + 1);
    }
    /**
    * Returns a Filter object which provides access to message iterator
    * for iterating a subset of messages within the [begin, end) range of buckets specified.
    *
    * @param beginKey The first bin to access messages from
    * @param endKey The bin beyond the last bin to access messages from
    */
    inline __device__ Filter operator() (const IntT &beginKey, const IntT &endKey) const {
#if !defined(SEATBELTS) || SEATBELTS
        {
            if (beginKey < metadata->min) {
                DTHROW("Bucket messaging iterator begin key %d is lower than minimum key (%d).\n", beginKey, metadata->min);
            } else if (endKey > metadata->max) {
                DTHROW("Bucket messaging iterator end key %d is higher than maximum key + 1 (%d).\n", endKey, metadata->max);
            } else if (endKey <= beginKey) {
                DTHROW("Bucket messaging iterator begin key must be lower than end key (%d !< %d).\n", beginKey, endKey);
            }
        }
#endif
        return Filter(metadata, combined_hash, beginKey, endKey);
    }

 private:
    /**
    * CURVE hash for accessing message data
    * agentfn_hash + msg_hash
    */
    Curve::NamespaceHash combined_hash;
    /**
    * Device pointer to metadata required for accessing data structure
    * e.g. PBM, search origin, environment bounds
    */
    const MetaData *metadata;
};

/**
 * This class is accessible via DeviceAPI.message_out if MsgBucket is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting bucketed messages
 */
class MsgBucket::Out : public MsgBruteForce::Out {
 public:
    /**
    * Constructor
    * Initialises member variables
    * @param agentfn_hash Added to msg_hash to produce combined_hash
    * @param msg_hash Added to agentfn_hash to produce combined_hash
    * @param scan_flag_messageOutput Scan flag array for optional message output
    */
    __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *_metadata, unsigned int *scan_flag_messageOutput)
        : MsgBruteForce::Out(agentfn_hash, msg_hash, nullptr, scan_flag_messageOutput)
#if !defined(SEATBELTS) || SEATBELTS
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
#else
        , metadata(nullptr)
#endif
    { }
    /**
    * Sets the location for this agents message
    * @param key Key of the bucket to store the message
    * @note Convenience wrapper for setVariable()
    */
    inline __device__ void setKey(const IntT &key) const;
    /**
    * Metadata struct for accessing messages
    */
    const MetaData * const metadata;
};

__device__ MsgBucket::In::Filter::Filter(const MetaData* _metadata, const Curve::NamespaceHash &_combined_hash, const IntT& beginKey, const IntT& endKey)
    : bucket_begin(0)
    , bucket_end(0)
    , metadata(_metadata)
    , combined_hash(_combined_hash) {
    // If key is in bounds
    if (beginKey >= metadata->min && endKey < metadata->max && beginKey <= endKey) {
        bucket_begin = metadata->PBM[beginKey - metadata->min];
        bucket_end = metadata->PBM[endKey - metadata->min];
    }
}

__device__ void MsgBucket::Out::setKey(const IntT &key) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

#if !defined(SEATBELTS) || SEATBELTS
    if (key < metadata->min || key >= metadata->max) {
        DTHROW("MsgArray key %u is out of range [%d, %d).\n", key, metadata->min, metadata->max);
    }
#endif
    // set the variables using curve
    Curve::setMessageVariable<IntT>("_key", combined_hash, key, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}

template<typename T, unsigned int N>
__device__ T MsgBucket::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (cell_index >= _parent.bucket_end) {
        DTHROW("Bucket message index exceeds bin length, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    T value = Curve::getMessageVariable<T>(variable_name, this->_parent.combined_hash, cell_index);
    return value;
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_BUCKETDEVICE_H_

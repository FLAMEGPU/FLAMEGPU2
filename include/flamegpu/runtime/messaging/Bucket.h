#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_H_

#ifndef __CUDACC_RTC__
#include <memory>
#include <string>

#include "flamegpu/runtime/cuRVE/curve.h"
#endif  // __CUDACC_RTC__

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"

typedef int IntT;

/**
 * Bucket messaging functionality
 *
 * User specifies an integer upper and lower bound, these form a set of consecutive indices which act as keys to buckets.
 * Each bucket may contain 0 to many messages, however an index is generated such that empty bins still consume a small amount of space.
 * As such, this is similar to a multi-map, however the key space must be a set of consecutive integers.
 *
 * By using your own hash function you can convert non-integer keys to suitable integer keys.
 */
class MsgBucket {
    /**
     * Common size type
     */
    typedef MsgNone::size_type size_type;

 public:
    class Message;      // Forward declare inner classes
    class iterator;     // Forward declare inner classes
    struct Data;        // Forward declare inner classes
    class Description;  // Forward declare inner classes

    /**
     * MetaData required by bucket messaging during message reads
     */
    struct MetaData {
        /**
         * The inclusive minimum environment bound
         */
        IntT min;
        /**
         * The exclusive maximum environment bound
         */
        IntT max;
        /**
         * Pointer to the partition boundary matrix in device memory
         * The PBM is never stored on the host
         */
        unsigned int *PBM;
    };

    /**
     * This class is accessible via FLAMEGPU_DEVICE_API.message_in if MsgBucket is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for reading bucket
     */
    class In {
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
            __device__ Filter(const MetaData *_metadata, const Curve::NamespaceHash &combined_hash, const IntT &beginKey, const IntT &endKey);
            /**
             * Returns an iterator to the start of the message list subset about the search origin
             */
            inline __device__ iterator begin(void) const {
                // Bin before initial bin, as the constructor calls increment operator
                return iterator(*this, bucket_begin-1);
            }
            /**
             * Returns an iterator to the position beyond the end of the message list subset
             * @note This iterator is the same for all message list subsets
             */
            inline __device__ iterator end(void) const {
                // Final bin, as the constructor calls increment operator
                return iterator(*this, bucket_end-1);
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
     * This class is accessible via FLAMEGPU_DEVICE_API.message_out if MsgBucket is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for outputting bucketed messages
     */
    class Out : public MsgBruteForce::Out {
     public:
        /**
         * Constructor
         * Initialises member variables
         * @param agentfn_hash Added to msg_hash to produce combined_hash
         * @param msg_hash Added to agentfn_hash to produce combined_hash
         * @param scan_flag_messageOutput Scan flag array for optional message output
         */
        __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *, unsigned int *scan_flag_messageOutput)
            : MsgBruteForce::Out(agentfn_hash, msg_hash, nullptr, scan_flag_messageOutput)
        { }
        /**
         * Sets the location for this agents message
         * @param key Key of the bucket to store the message
         * @note Convenience wrapper for setVariable()
         */
        __device__ void setKey(const IntT &key) const;
    };
#ifndef __CUDACC_RTC__
    /**
     * CUDA host side handler of bucket messages
     * Allocates memory for and constructs PBM
     */
    class CUDAModelHandler : public MsgSpecialisationHandler {
     public:
        /**
         * Constructor
         * 
         * Initialises metadata, decides PBM size etc
         * 
         * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
         */
         explicit CUDAModelHandler(CUDAMessage &a);
        /**
         * Destructor
         * Frees all allocated memory
         */
         ~CUDAModelHandler() override;
        /**
         * Allocates memory for the constructed index.
         * Sets data asthough message list is empty
         * @param scatter Scatter instance and scan arrays to be used (CUDAAgentModel::singletons->scatter)
         * @param streamId Index of stream specific structures used
         */
        void init(CUDAScatter &scatter, const unsigned int &streamId) override;
        /**
         * Reconstructs the partition boundary matrix
         * This should be called before reading newly output messages
         * @param scatter Scatter instance and scan arrays to be used (CUDAAgentModel::singletons->scatter)
         * @param streamId Index of stream specific structures used
         */
        void buildIndex(CUDAScatter &scatter, const unsigned int &streamId) override;
        /**
         * Allocates memory for the constructed index.
         * The memory allocation is checked by build index.
         */
        void allocateMetaDataDevicePtr() override;
        /**
         * Releases memory for the constructed index.
         */
        void freeMetaDataDevicePtr() override;
        /**
         * Returns a pointer to the metadata struct, this is required for reading the message data
         */
        const void *getMetaDataDevicePtr() const override { return d_data; }

     private:
        /**
         * Resizes the cub temp memory
         * Currently assumed that bounds of environment/rad never change
         * So this is only called from the constructor.
         * If it were called elsewhere, it would need to be changed to resize d_histogram too
         */
        void resizeCubTemp();
        /**
         * Resizes the key value store, this scales with agent count
         * @param newSize The new number of agents to represent
         * @note This only scales upwards, it will never reduce the size
         */
        void resizeKeysVals(const unsigned int &newSize);
        /**
         * upperBound-lowerBound
         */
        unsigned int bucketCount;
        /**
         * Size of currently allocated temp storage memory for cub
         */
        size_t d_CUB_temp_storage_bytes = 0;
        /**
         * Pointer to currently allocated temp storage memory for cub
         */
        unsigned int *d_CUB_temp_storage = nullptr;
        /**
         * Pointer to array used for histogram
         */
        unsigned int *d_histogram = nullptr;
        /**
         * Arrays used to store indices when sorting messages
         */
        unsigned int *d_keys = nullptr, *d_vals = nullptr;
        /**
         * Size currently allocated to d_keys, d_vals arrays
         */
        size_t d_keys_vals_storage_bytes = 0;
        /**
         * Host copy of metadata struct
         */
        MetaData hd_data;
        /**
         * Pointer to device copy of metadata struct
         */
        MetaData *d_data = nullptr;
        /**
         * Owning CUDAMessage, provides access to message storage etc
         */
        CUDAMessage &sim_message;
    };

    /**
     * Internal data representation of Bucket messages within model description hierarchy
     * @see Description
     */
    struct Data : public MsgBruteForce::Data {
        friend class ModelDescription;
        friend struct ModelData;
        /**
         * Initially set to 0
         * Min must be set to the first valid key
         */
        IntT lowerBound;
        /**
         * Initially set to std::numeric_limits<IntT>::max(), which acts as flag to say it has not been set
         * Max must be set to the last valid key
         */
        IntT upperBound;
        virtual ~Data() = default;

        std::unique_ptr<MsgSpecialisationHandler> getSpecialisationHander(CUDAMessage &owner) const override;

        /**
         * Used internally to validate that the corresponding Msg type is attached via the agent function shim.
         * @return The std::type_index of the Msg type which must be used.
         */
        std::type_index getType() const override;

     protected:
         Data *clone(const std::shared_ptr<const ModelData> &newParent) override;
        /**
         * Copy constructor
         * This is unsafe, should only be used internally, use clone() instead
         */
         Data(const std::shared_ptr<const ModelData> &, const Data &other);
        /**
         * Normal constructor, only to be called by ModelDescription
         */
         Data(const std::shared_ptr<const ModelData> &, const std::string &message_name);
    };

    /**
     * User accessible interface to Bucket messages within mode description hierarchy
     * @see Data
     */
    class Description : public MsgBruteForce::Description {
        /**
         * Data store class for this description, constructs instances of this class
         */
        friend struct Data;

     protected:
        /**
         * Constructors
         */
         Description(const std::shared_ptr<const ModelData> &_model, Data *const data);
        /**
         * Default copy constructor, not implemented
         */
         Description(const Description &other_message) = delete;
        /**
         * Default move constructor, not implemented
         */
         Description(Description &&other_message) noexcept = delete;
        /**
         * Default copy assignment, not implemented
         */
         Description& operator=(const Description &other_message) = delete;
        /**
         * Default move assignment, not implemented
         */
         Description& operator=(Description &&other_message) noexcept = delete;

     public:
        /**
         * Set the (inclusive) minimum bound, this is the first valid key
         */
        void setLowerBound(const IntT &key);
        /**
         * Set the (inclusive) maximum bound, this is the last valid key
         */
        void setUpperBound(const IntT &key);
        void setBounds(const IntT &min, const IntT &max);
        /**
         * Return the currently set (inclusive) lower bound, this is the first valid key
         */
        IntT getLowerBound() const;
        /**
         * Return the currently set (inclusive) upper bound, this is the last valid key
         */
        IntT getUpperBound() const;
    };
#endif  // __CUDACC_RTC__
};

template<typename T, unsigned int N>
__device__ T MsgBucket::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
    //// Ensure that the message is within bounds.
    if (cell_index < _parent.bucket_end) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, cell_index);
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_H_

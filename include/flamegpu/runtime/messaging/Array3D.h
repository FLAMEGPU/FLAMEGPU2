#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY3D_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY3D_H_

#ifndef __CUDACC_RTC__
#include <string>
#include <memory>
#include <array>

#include "flamegpu/model/Variable.h"
#endif  // __CUDACC_RTC__
#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Array2D.h"

/**
 * Array messaging functionality
 *
 * Like an array, each message is assigned an index within a known range
 * Only one message may exist at each index
 * Agent functions can access individual messages by requesting them with their index
 * 
 * Algorithm:
 * Every agent outputs a message to the array based on their thread index
 * They also set the __index variable with the intended output bin
 * When buildIndex() is called, messages are sorted and errors (multiple messages per bin) are detected
 */
class MsgArray3D {
 public:
    /**
     * Common size type
     */
    typedef MsgNone::size_type size_type;

    class Message;      // Forward declare inner classes
    class iterator;     // Forward declare inner classes
    struct Data;        // Forward declare inner classes
    class Description;  // Forward declare inner classes
    /**
     * MetaData required by brute force during message reads
     */
    struct MetaData {
        /**
         * Dimensions of array
         */
        size_type dimensions[3];
        /**
         * Total number of elements
         */
        size_type length;
    };
    /**
     * This class is accessible via FLAMEGPU_DEVICE_API.message_in if MsgArray3D is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for reading array messages
     */
    class In {
        /**
         * Message has full access to In, they are treated as the same class so share everything
         * Reduces/memory data duplication
         */
        friend class MsgArray3D::Message;

     public:
        /**
         * This class is created when a search origin is provided to MsgArray2D::In::operator()(size_type, size_type, size_type = 1)
         * It provides iterator access to a subset of the full message list, according to the provided search origin and radius
         * 
         * @see MsgArray2D::In::operator()(size_type, size_type, size_type)
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
                int relative_cell[3];
                /**
                 * Index into memory of currently pointed message
                 */
                size_type index_1d = 0;

             public:
                /**
                 * Constructs a message and directly initialises all of it's member variables
                 * @note See member variable documentation for their purposes
                 */
                __device__ Message(const Filter &parent, const int &relative_x, const int &relative_y, const int &relative_z)
                    : _parent(parent) {
                    relative_cell[0] = relative_x;
                    relative_cell[1] = relative_y;
                    relative_cell[2] = relative_z;
                }
                /**
                 * Equality operator
                 * Compares all internal member vars for equality
                 * @note Does not compare _parent
                 */
                __device__ bool operator==(const Message& rhs) const {
                    return this->index_1d == rhs.index_1d
                        && this->_parent.loc[0] == rhs._parent.loc[0]
                        && this->_parent.loc[1] == rhs._parent.loc[1]
                        && this->_parent.loc[2] == rhs._parent.loc[2];
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
                __device__ Message& operator++();
                /**
                 * Returns x array index of message
                 */
                __device__ size_type getX() const {
                    return (this->_parent.loc[0] + relative_cell[0] + this->_parent.metadata->dimensions[0]) % this->_parent.metadata->dimensions[0];
                }
                /**
                 * Returns y array index of message
                 */
                __device__ size_type getY() const {
                    return (this->_parent.loc[1] + relative_cell[1] + this->_parent.metadata->dimensions[1]) % this->_parent.metadata->dimensions[1];
                }
                /**
                 * Returns z array index of message
                 */
                __device__ size_type getZ() const {
                    return (this->_parent.loc[2] + relative_cell[2] + this->_parent.metadata->dimensions[2]) % this->_parent.metadata->dimensions[2];
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
                 * This iterator is constructed by MsgArray3D::In::Filter::begin()(size_type, size_type, size_type, size_type)
                 * @see MsgArray3D::In::Operator()(size_type, size_type, size_type, size_type)
                 */
                __device__ iterator(const Filter &parent, const int &relative_x, const int &relative_y, const int &relative_z)
                    : _message(parent, relative_x, relative_y, relative_z) {
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
             * @param _metadata Pointer to message list metadata
             * @param _combined_hash agentfn+message hash for accessing message data
             * @param x Search origin x coord
             * @param y Search origin y coord
             * @param z Search origin z coord
             * @param _radius Search radius
             */
            __device__ Filter(const MetaData *_metadata, const Curve::NamespaceHash &_combined_hash, const size_type &x, const size_type &y, const size_type &z, const size_type &_radius);
            /**
             * Returns an iterator to the start of the message list subset about the search origin
             */
            inline __device__ iterator begin(void) const {
                // Bin before initial bin, as the constructor calls increment operator
                return iterator(*this, -static_cast<int>(radius), -static_cast<int>(radius), -static_cast<int>(radius)-1);
            }
            /**
             * Returns an iterator to the position beyond the end of the message list subset
             * @note This iterator is the same for all message list subsets
             */
            inline __device__ iterator end(void) const {
                // Final bin, as the constructor calls increment operator
                return iterator(*this, radius, radius, radius);
            }

         private:
            /**
             * Search origin
             */
            size_type loc[3];
            /**
             * Search radius
             */
            const size_type radius;
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
         * Constructer
         * Initialises member variables
         * @param agentfn_hash Added to msg_hash to produce combined_hash
         * @param msg_hash Added to agentfn_hash to produce combined_hash
         * @param _metadata Reinterpreted as type MsgArray3D::MetaData
         */
        __device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *_metadata)
            : combined_hash(agentfn_hash + msg_hash)
            , metadata(reinterpret_cast<const MetaData*>(_metadata))
        { }
        /**
         * Returns a Filter object which provides access to message iterator
         * for iterating a subset of messages including those within the radius of the search origin
         * this excludes the message at the search origin
         *
         * @param x Search origin x coord
         * @param y Search origin y coord
         * @param z Search origin y coord
         * @param radius Search radius
         * @note radius 1 is 26 cells in 3x3x3
         * @note radius 2 is 124 cells in 5x5x5
         * @note If radius is >= half of the array dimensions, cells will be doubly read
         * @note radius of 0 is unsupported
         */
        inline __device__ Filter operator() (const size_type &x, const size_type &y, const size_type &z, const size_type &radius = 1) const {
            assert(radius > 0);  // Radius of 0 is bad
            return Filter(metadata, combined_hash, x, y, z, radius);
        }
        /**
         * Returns the x dimension size of the message list
         */
        __device__ size_type getDimX() const {
            return metadata->dimensions[0];
        }
        /**
         * Returns the y dimension size of the message list
         */
        __device__ size_type getDimY() const {
            return metadata->dimensions[1];
        }
        /**
         * Returns the z dimension size of the message list
         */
        __device__ size_type getDimZ() const {
            return metadata->dimensions[2];
        }
        /**
         * Returns the length of the message list.
         * xDim x yDim x zDim
         */
        __device__ size_type size(void) const {
            return metadata->length;
        }
        __device__ Message at(const size_type &x, const size_type &y, const size_type &z) const {
            const size_type index_1d =
                z * metadata->dimensions[0] * metadata->dimensions[1]  +
                y * metadata->dimensions[0]  +
                x;
            return Message(*this, index_1d);
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
        const MetaData * const metadata;
    };
    /**
     * Provides access to a specific message
     * Returned by In::at(size_type)
     * @see In::at(size_type)
     */
    class Message {
         /**
          * Paired In class which created the iterator
          */
        const MsgArray3D::In &_parent;
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
        __device__ Message(const MsgArray3D::In &parent, const size_type &_index) : _parent(parent), index(_index) {}
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
        template<typename T, size_type N>
        __device__ T getVariable(const char(&variable_name)[N]) const;
    };
    /**
     * This class is accessible via FLAMEGPU_DEVICE_API.message_out if MsgArray3D is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for outputting array messages
     */
    class Out {
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
            , metadata(reinterpret_cast<const MetaData*>(_metadata))
        { }
        /**
         * Sets the array index to store the message in
         */
        __device__ void setIndex(const size_type &x, const size_type &y, const size_type &z) const;
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
#ifndef __CUDACC_RTC__
    /**
     * Blank handler, brute force requires no index or special allocations
     * Only stores the length on device
     */
    class CUDAModelHandler : public MsgSpecialisationHandler {
     public:
        /**
         * Constructor
         * Allocates memory on device for message list length
         * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
         */
         explicit CUDAModelHandler(CUDAMessage &a);
        /** 
         * Destructor.
         * Should free any local host memory (device memory cannot be freed in destructors)
         */
        ~CUDAModelHandler() { }
        /**
         * Sort messages according to index
         * Detect and report any duplicate indicies/gaps
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
        const void *getMetaDataDevicePtr() const override { return d_metadata; }

     private:
        /**
         * Host copy of metadata struct (message list length)
         */
        MetaData hd_metadata;
        /**
         * Pointer to device copy of metadata struct (message list length)
         */
        MetaData *d_metadata;
        /**
         * Owning CUDAMessage, provides access to message storage etc
         */
        CUDAMessage &sim_message;
        /**
         * Buffer used by buildIndex if array length > agent count
         */
        unsigned int *d_write_flag;
        /**
         * Allocated length of d_write_flag (in number of uint, not bytes)
         */
        size_type d_write_flag_len;
    };
    /**
     * Internal data representation of Array messages within model description hierarchy
     * @see Description
     */
    struct Data : public MsgBruteForce::Data {
        friend class ModelDescription;
        friend struct ModelData;
        std::array<size_type, 3> dimensions;
        virtual ~Data() = default;

        std::unique_ptr<MsgSpecialisationHandler> getSpecialisationHander(CUDAMessage &owner) const override;

        /**
         * Used internally to validate that the corresponding Msg type is attached via the agent function shim.
         * @return The std::type_index of the Msg type which must be used.
         */
        std::type_index getType() const override;

     protected:
         Data *clone(const std::shared_ptr<const ModelData> & newParent) override;
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
     * User accessible interface to Array messages within mode description hierarchy
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
         Description(const std::shared_ptr<const ModelData> & _model, Data *const data);
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
        void setDimensions(const size_type &len_x, const size_type &len_y, const size_type &len_Z);
        void setDimensions(const std::array<size_type, 3> &dims);

        std::array<size_type, 3> getDimensions() const;
        size_type getDimX() const;
        size_type getDimY() const;
        size_type getDimZ() const;
    };
#endif  // __CUDACC_RTC__
};
template<typename T, unsigned int N>
__device__ T MsgArray3D::Message::getVariable(const char(&variable_name)[N]) const {
    // Ensure that the message is within bounds.
    if (index < this->_parent.metadata->length) {
        // get the value from curve using the stored hashes and message index.
        return Curve::getVariable<T>(variable_name, this->_parent.combined_hash, index);
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}
template<typename T, unsigned int N>
__device__ T MsgArray3D::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
    // Ensure that the message is within bounds.
    if (index_1d < this->_parent.metadata->length) {
        // get the value from curve using the stored hashes and message index.
        return Curve::getVariable<T>(variable_name, this->_parent.combined_hash, index_1d);
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}

template<typename T, unsigned int N>
__device__ void MsgArray3D::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    if (variable_name[0] == '_') {
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    Curve::setVariable<T>(variable_name, combined_hash, value, index);

    // setIndex() sets the optional msg scan flag
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY3D_H_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCESORTED_MESSAGEBRUTEFORCESORTEDDEVICE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCESORTED_MESSAGEBRUTEFORCESORTEDDEVICE_CUH_

#include "flamegpu/runtime/messaging/MessageBruteForceSorted.h"
#include "flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DDevice.cuh"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"

namespace flamegpu {

/**
 * This class is accessible via DeviceAPI.message_in if MessageBruteForceSorted is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading spatially partitioned messages
 */
class MessageBruteForceSorted::In {
 public:
    class Message;      // Forward declare inner classes
    class iterator;     // Forward declare inner classe

    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to message_hash to produce combined_hash
     * @param message_hash Added to agentfn_hash to produce combined_hash
     * @param _metadata Reinterpreted as type MessageBruteForceSorted::MetaData
     */
    __device__ In(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash message_hash, const void *_metadata)
        : combined_hash(agentfn_hash + message_hash)
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
    { }
/**
     * Returns the number of elements in the message list.
     */
     __device__ size_type size(void) const {
        return metadata->length;
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
        return iterator(*this, metadata->length);
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
        const MessageBruteForceSorted::In &_parent;
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
        __device__ Message(const MessageBruteForceSorted::In &parent) : _parent(parent), index(0) {}
        /**
         * Alternate constructor, allows index to be manually set
         * @note I think this is unused
         */
        __device__ Message(const MessageBruteForceSorted::In &parent, size_type index) : _parent(parent), index(index) {}
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
    * Stock iterator for iterating MessageBruteForceSorted::In::Message objects
    */
    class iterator {
        /**
         * The message returned to the user
         */
         Message _message;

     public:
        /**
         * Constructor
         * This iterator is constructed by MessageBruteForceSorted::begin()
         * @see MessageBruteForceSorted::begin()
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
     * agentfn_hash + message_hash
     */
    detail::curve::Curve::NamespaceHash combined_hash;
    /**
     * Device pointer to metadata required for accessing data structure
     * e.g. PBM, search origin, environment bounds
     */
    const MetaData *metadata;
};

/**
 * This class is accessible via DeviceAPI.message_out if MessageBruteForceSorted is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting spatially partitioned messages
 */
class MessageBruteForceSorted::Out : public MessageSpatial3D::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to message_hash to produce combined_hash
     * @param message_hash Added to agentfn_hash to produce combined_hash
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash message_hash, const void *, unsigned int *scan_flag_messageOutput)
        : MessageSpatial3D::Out(agentfn_hash, message_hash, nullptr, scan_flag_messageOutput)
    { }
    /**
     * Sets the location for this agents message
     * @param x Message x coord
     * @param y Message y coord
     * @param z Message z coord
     * @note Convenience wrapper for setVariable()
     */
    __device__ void setLocation(const float &x, const float &y, const float &z) const;
};

__device__ __forceinline__ MessageBruteForceSorted::GridPos3D getGridPosition3D(const MessageBruteForceSorted::MetaData *md, float x, float y, float z) {
    // Clamp each grid coord to 0<=x<dim
    int gridPos[3] = {
        static_cast<int>(floorf(((x-md->min[0]) / md->environmentWidth[0])*md->gridDim[0])),
        static_cast<int>(floorf(((y-md->min[1]) / md->environmentWidth[1])*md->gridDim[1])),
        static_cast<int>(floorf(((z-md->min[2]) / md->environmentWidth[2])*md->gridDim[2]))
    };
    MessageBruteForceSorted::GridPos3D rtn = {
        gridPos[0] < 0 ? 0 : (gridPos[0] >= static_cast<int>(md->gridDim[0]) ? static_cast<int>(md->gridDim[0]) - 1 : gridPos[0]),
        gridPos[1] < 0 ? 0 : (gridPos[1] >= static_cast<int>(md->gridDim[1]) ? static_cast<int>(md->gridDim[1]) - 1 : gridPos[1]),
        gridPos[2] < 0 ? 0 : (gridPos[2] >= static_cast<int>(md->gridDim[2]) ? static_cast<int>(md->gridDim[2]) - 1 : gridPos[2])
    };
    return rtn;
}
__device__ __forceinline__ unsigned int getHash3D(const MessageBruteForceSorted::MetaData *md, const MessageBruteForceSorted::GridPos3D &xyz) {
    // Bound gridPos to gridDimensions
    unsigned int gridPos[3] = {
        (unsigned int)(xyz.x < 0 ? 0 : (xyz.x >= static_cast<int>(md->gridDim[0]) - 1 ? static_cast<int>(md->gridDim[0]) - 1 : xyz.x)),  // Only x should ever be out of bounds here
        (unsigned int) xyz.y,  // xyz.y < 0 ? 0 : (xyz.y >= md->gridDim[1] - 1 ? md->gridDim[1] - 1 : xyz.y),
        (unsigned int) xyz.z,  // xyz.z < 0 ? 0 : (xyz.z >= md->gridDim[2] - 1 ? md->gridDim[2] - 1 : xyz.z)
    };
    // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
    return (unsigned int)(
        (gridPos[2] * md->gridDim[0] * md->gridDim[1]) +   // z
        (gridPos[1] * md->gridDim[0]) +                    // y
        gridPos[0]);                                      // x
}

__device__ inline void MessageBruteForceSorted::Out::setLocation(const float &x, const float &y, const float &z) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variables using curve
    detail::curve::Curve::setMessageVariable<float>("x", combined_hash, x, index);
    detail::curve::Curve::setMessageVariable<float>("y", combined_hash, y, index);
    detail::curve::Curve::setMessageVariable<float>("z", combined_hash, z, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}

template<typename T, unsigned int N>
__device__ T MessageBruteForceSorted::In::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.metadata->length) {
        DTHROW("Brute force sorted message index exceeds messagelist length, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    T value = detail::curve::Curve::getMessageVariable_ldg<T>(variable_name, this->_parent.combined_hash, index);
    return value;
}
template<typename T, MessageNone::size_type N, unsigned int M> __device__
T MessageBruteForceSorted::In::Message::getVariable(const char(&variable_name)[M], const unsigned int& array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.metadata->length) {
        DTHROW("Brute force sorted message index exceeds messagelist length, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    T value = detail::curve::Curve::getMessageArrayVariable_ldg<T, N>(variable_name, this->_parent.combined_hash, index, array_index);
    return value;
}
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCESORTED_MESSAGEBRUTEFORCESORTEDDEVICE_CUH_

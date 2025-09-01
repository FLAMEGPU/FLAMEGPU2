#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL3D_MESSAGESPATIAL3DDEVICE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL3D_MESSAGESPATIAL3DDEVICE_CUH_

#include "flamegpu/runtime/messaging/MessageSpatial3D.h"
#include "flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DDevice.cuh"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"

namespace flamegpu {

/**
 * This class is accessible via DeviceAPI.message_in if MessageSpatial3D is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading spatially partitioned messages
 */
class MessageSpatial3D::In {
 public:
    /**
     * This class is created when a search origin is provided to MessageSpatial3D::In::operator()(float, float, float)
     * It provides iterator access to a subset of the full message list, according to the provided search origin
     * 
     * @see MessageSpatial3D::In::operator()(float, float, float)
     */
    class Filter {
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
             * Relative strip within the Moore neighbourhood
             * Strips run along the x axis
             * relative_cell[0] corresponds to y offset
             * relative_cell[1] corresponds to z offset
             */
            int relative_cell[2] = { -2, 1 };
            /**
             * This is the index after the final message, relative to the full message list, in the current bin
             */
            int cell_index_max = 0;
            /**
             * This is the index of the currently accessed message, relative to the full message list
             */
            int cell_index = 0;
            /**
             * Utility function for deciding next strip to access
             */
            __device__ void nextStrip() {
                if (relative_cell[1] >= 1) {
                    relative_cell[1] = -1;
                    relative_cell[0]++;
                } else {
                    relative_cell[1]++;
                }
            }

         public:
            /**
             * Constructs a message and directly initialises all of it's member variables
             * @note See member variable documentation for their purposes
             */
            __device__ Message(const Filter &parent, const int relative_cell_y, const int relative_cell_z, const int _cell_index_max, const int _cell_index)
                : _parent(parent)
                , cell_index_max(_cell_index_max)
                , cell_index(_cell_index) {
                relative_cell[0] = relative_cell_y;
                relative_cell[1] = relative_cell_z;
            }
            /**
             * False minimal constructor used by iterator::end()
             */
            __device__ Message(const Filter &parent)
                : _parent(parent) { }
            /**
             * Equality operator
             * Compares all internal member vars for equality
             * @note Does not compare _parent
             */
            __device__ bool operator==(const Message &rhs) const {
                return this->relative_cell[0] == rhs.relative_cell[0]
                    && this->relative_cell[1] == rhs.relative_cell[1]
                    && this->cell_index_max == rhs.cell_index_max
                    && this->cell_index == rhs.cell_index;
            }
            /**
             * This should only be called to compare against end()
             * It has been modified to check for end of iteration with minimal instructions
             * Therefore it does not even perform the equality operation
             * @note Use operator==() if proper equality is required
             */
            __device__ bool operator!=(const Message&) const {
                // The incoming Message& is end(), so we don't care about that
                // We only care that the host object has reached end
                // When the strip number equals 2, it has exceeded the [-1, 1] range
                return !(this->relative_cell[0] >= 2);
            }
            /**
             * Updates the message to return variables from the next message in the message list
             * @return Returns itself
             */
            __device__ Message& operator++();
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
             * This iterator is constructed by MessageSpatial3D::In::Filter::begin()(float, float, float)
             * @see MessageSpatial3D::In::Operator()(float, float, float)
             */
            __device__ iterator(const Filter &parent, const int relative_cell_y, const int relative_cell_z, const int _cell_index_max, const int _cell_index)
                : _message(parent, relative_cell_y, relative_cell_z, _cell_index_max, _cell_index) {
                // Increment to find first message
                ++_message;
            }
            /**
             * False constructor
             * Only used by Filter::end(), creates a null objct
             */
            __device__ iterator(const Filter &parent)
                : _message(parent) { }
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
         * @param x Search origin x coord
         * @param y Search origin y coord
         * @param z search origin z coord
         */
        __device__ Filter(const MetaData *_metadata, float x, float y, float z);
        /**
         * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
            // Bin before initial bin, as the constructor calls increment operator
            return iterator(*this, -2, 1, 1, 0);
        }
        /**
         * Returns an iterator to the position beyond the end of the message list subset
         * @note This iterator is the same for all message list subsets
         */
        inline __device__ iterator end(void) const {
            // Empty init, because this object is never used
            // iterator equality doesn't actually check the end object
            return iterator(*this);
        }

     private:
        /**
         * Search origin
         */
        float loc[3];
        /**
         * Search origin's grid cell
         */
        GridPos3D cell;
        /**
         * Pointer to message list metadata, e.g. environment bounds, search radius, PBM location
         */
        const MetaData *metadata;
    };
    /**
     * This class is created when a search origin is provided to MessageSpatial3D::In::wrap()(float, float, float)
     * It provides iterator access to a subset of the full message list, according to the provided search origin
     * 
     * @see MessageSpatial3D::In::wrap()(float, float, float)
     */
    class WrapFilter {
     public:
        /**
         * Provides access to a specific message
         * Returned by the iterator
         * @see In::WrapFilter::iterator
         */
        class Message {
            /**
             * Paired Filter class which created the iterator
             */
            const WrapFilter&_parent;
            /**
             * Relative strip within the Moore neighbourhood
             * Strips run along the x axis
             * relative_cell[0] corresponds to x offset
             * relative_cell[1] corresponds to y offset
             * relative_cell[2] corresponds to z offset
             */
            int relative_cell[3] = { -2, 1, -1 };
            /**
             * This is the index after the final message, relative to the full message list, in the current bin
             */
            int cell_index_max = 0;
            /**
             * This is the index of the currently accessed message, relative to the full message list
             */
            int cell_index = 0;
            /**
             * Utility function for deciding next strip to access
             */
            __device__ void nextCell() {
                if (relative_cell[2] >= 1) {
                    relative_cell[2] = -1;
                    if (relative_cell[1] >= 1) {
                        relative_cell[1] = -1;
                        ++relative_cell[0];
                    } else {
                        ++relative_cell[1];
                    }
                } else {
                    ++relative_cell[2];
                }
            }

         public:
            /**
             * Constructs a message and directly initialises all of it's member variables
             * @note See member variable documentation for their purposes
             */
            __device__ Message(const WrapFilter& parent, const int relative_cell_x, const int relative_cell_y, const int relative_cell_z, const int _cell_index_max, const int _cell_index)
                : _parent(parent)
                , cell_index_max(_cell_index_max)
                , cell_index(_cell_index) {
                relative_cell[0] = relative_cell_x;
                relative_cell[1] = relative_cell_y;
                relative_cell[2] = relative_cell_z;
            }
            /**
             * False minimal constructor used by iterator::end()
             */
            __device__ Message(const WrapFilter& parent)
                : _parent(parent) { }
            /**
             * Equality operator
             * Compares all internal member vars for equality
             * @note Does not compare _parent
             */
            __device__ bool operator==(const Message &rhs) const {
                return this->relative_cell[0] == rhs.relative_cell[0]
                    && this->relative_cell[1] == rhs.relative_cell[1]
                    && this->relative_cell[2] == rhs.relative_cell[2]
                    && this->cell_index_max == rhs.cell_index_max
                    && this->cell_index == rhs.cell_index;
            }
            /**
             * This should only be called to compare against end()
             * It has been modified to check for end of iteration with minimal instructions
             * Therefore it does not even perform the equality operation
             * @note Use operator==() if proper equality is required
             */
            __device__ bool operator!=(const Message&) const {
                // The incoming Message& is end(), so we don't care about that
                // We only care that the host object has reached end
                // When the strip number equals 2, it has exceeded the [1, 1, 1] range
                return !(this->relative_cell[0] >= 2);
            }
            /**
             * Updates the message to return variables from the next message in the message list
             * @return Returns itself
             */
            __device__ Message& operator++();
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
            /**
             * Returns the virtual x variable of the message, relative to the search origin
             * This is the closest x coordinate the message would have, relative to the observer's x coordinate in a wrapped environment
             */
            __device__ float getVirtualX() const {
                return getVirtualX(_parent.loc[0]);
            }
            /**
             * Returns the virtual y variable of the message, relative to the search origin
             * This is the closest y coordinate the message would have, relative to the observer's y coordinate in a wrapped environment
             */
            __device__ float getVirtualY() const {
                return getVirtualY(_parent.loc[1]);
            }
            /**
             * Returns the virtual z variable of the message, relative to the search origin
             * This is the closest z coordinate the message would have, relative to the observer's z coordinate in a wrapped environment
             */
            __device__ float getVirtualZ() const {
                return getVirtualZ(_parent.loc[2]);
            }
            /**
             * Returns the virtual x variable of the message
             * This is the closest x coordinate the message would have, relative to the observer's x coordinate in a wrapped environment
             * @param x1 The x coordinate of the observer
             */
            __device__ float getVirtualX(const float x1) const {
                const float x2 = getVariable<float>("x");
                const float x21 = x2 - x1;
                return abs(x21) > _parent.metadata->environmentWidth[0] / 2.0f ? x2 - (x21 / abs(x21) * _parent.metadata->environmentWidth[0]) : x2;
            }
            /**
             * Returns the virtual y variable of the message
             * This is the closest y coordinate the message would have, relative to the observer's y coordinate in a wrapped environment
             * @param y1 The y coordinate of the observer
             */
            __device__ float getVirtualY(const float y1) const {
                const float y2 = getVariable<float>("y");
                const float y21 = y2 - y1;
                return abs(y21) > _parent.metadata->environmentWidth[1] / 2.0f ? y2 - (y21 / abs(y21) * _parent.metadata->environmentWidth[1]) : y2;
            }
            /**
             * Returns the virtual z variable of the message
             * This is the closest z coordinate the message would have, relative to the observer's z coordinate in a wrapped environment
             * @param z1 The z coordinate of the observer
             */
            __device__ float getVirtualZ(const float z1) const {
                const float z2 = getVariable<float>("z");
                const float z21 = z2 - z1;
                return abs(z21) > _parent.metadata->environmentWidth[2] / 2.0f ? z2 - (z21 / abs(z21) * _parent.metadata->environmentWidth[2]) : z2;
            }
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
             * This iterator is constructed by MessageSpatial3D::In::WrapFilter::begin()(float, float, float)
             * @see MessageSpatial3D::In::wrap()(float, float, float)
             */
            __device__ iterator(const WrapFilter& parent, const int relative_cell_x, const int relative_cell_y, const int relative_cell_z, const int _cell_index_max, const int _cell_index)
                : _message(parent, relative_cell_x, relative_cell_y, relative_cell_z, _cell_index_max, _cell_index) {
                // Increment to find first message
                ++_message;
            }
            /**
             * False constructor
             * Only used by Filter::end(), creates a null objct
             */
            __device__ iterator(const WrapFilter& parent)
                : _message(parent) { }
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
         * @param x Search origin x coord
         * @param y Search origin y coord
         * @param z search origin z coord
         */
        __device__ WrapFilter(const MetaData *_metadata, float x, float y, float z);
        /**
         * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
            // Bin before initial bin, as the constructor calls increment operator
            return iterator(*this, -2, 1, 1, 1, 0);
        }
        /**
         * Returns an iterator to the position beyond the end of the message list subset
         * @note This iterator is the same for all message list subsets
         */
        inline __device__ iterator end(void) const {
            // Empty init, because this object is never used
            // iterator equality doesn't actually check the end object
            return iterator(*this);
        }

     private:
        /**
         * Search origin
         */
        float loc[3];
        /**
         * Search origin's grid cell
         */
        GridPos3D cell;
        /**
         * Pointer to message list metadata, e.g. environment bounds, search radius, PBM location
         */
        const MetaData *metadata;
    };

    /**
     * Constructor
     * Initialises member variables
     * @param _metadata Reinterpreted as type MessageSpatial3D::MetaData
     */
    __device__ In(const void *_metadata)
        : metadata(reinterpret_cast<const MetaData*>(_metadata))
    { }
    /**
     * Returns a Filter object which provides access to message iterator
     * for iterating a subset of messages including those within the radius of the search origin
     * 
     * @param x Search origin x coord
     * @param y Search origin y coord
     * @param z Search origin z coord
     */
    inline __device__ Filter operator() (const float x, const float y, const float z) const {
        return Filter(metadata, x, y, z);
    }
    /**
     * Returns a WrapFilter object which provides access to message iterator
     * for iterating a subset of messages including those within the radius of the search origin
     *
     * @param x Search origin x coord
     * @param y Search origin y coord
     * @param z Search origin z coord
     *
     * @note Unlike the regular iterator, this iterator will not return messages outside of the search radius. The wrapped distance can be returned via WrapFilter::Message::distance()
     */
    inline __device__ WrapFilter wrap(const float x, const float y, const float z) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if (x > metadata->max[0] ||
            y > metadata->max[1] ||
            z > metadata->max[2] ||
            x < metadata->min[0] ||
            y < metadata->min[1] ||
            z < metadata->min[2]) {
            DTHROW("Location (%f, %f, %f) exceeds environment bounds (%g, %g, %g):(%g, %g, %g),"
                " this is unsupported for the wrapped iterator, MessageSpatial3D::In::wrap().\n", x, y, z,
                metadata->min[0], metadata->min[1], metadata->min[2],
                metadata->max[0], metadata->max[1], metadata->max[2]);
            // Return iterator at min corner of env, this should be safe
            return WrapFilter(metadata, metadata->min[0], metadata->min[1], metadata->min[2]);
        }
        if (!metadata->wrapCompatible) {
            DTHROW("Spatial messaging radius (%g) is not a factor of environment dimensions (%g, %g, %g),"
                " this is unsupported for the wrapped iterator, MessageSpatial3D::In::wrap().\n", metadata->radius,
                metadata->environmentWidth[0],
                metadata->environmentWidth[1],
                metadata->environmentWidth[2]);
        }
#endif
        return WrapFilter(metadata, x, y, z);
    }

    /**
     * Returns the search radius of the message list defined in the model description
     */
    __forceinline__ __device__ float radius() const {
        return metadata->radius;
    }

 private:
    /**
     * Device pointer to metadata required for accessing data structure
     * e.g. PBM, search origin, environment bounds
     */
    const MetaData *metadata;
};

/**
 * This class is accessible via DeviceAPI.message_out if MessageSpatial3D is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting spatially partitioned messages
 */
class MessageSpatial3D::Out : public MessageBruteForce::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(const void *, unsigned int *scan_flag_messageOutput)
        : MessageBruteForce::Out(nullptr, scan_flag_messageOutput)
    { }
    /**
     * Sets the location for this agents message
     * @param x Message x coord
     * @param y Message y coord
     * @param z Message z coord
     * @note Convenience wrapper for setVariable()
     */
    __device__ void setLocation(float x, float y, float z) const;
};

template<typename T, unsigned int N>
__device__ T MessageSpatial3D::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (relative_cell[0] >= 2) {
        DTHROW("MessageSpatial3D in invalid bin, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getMessageVariable<T>(variable_name, cell_index);
    return value;
}
template<typename T, flamegpu::size_type N, unsigned int M> __device__
T MessageSpatial3D::In::Filter::Message::getVariable(const char(&variable_name)[M], const unsigned int array_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (relative_cell[0] >= 2) {
        DTHROW("MessageSpatial3D in invalid bin, unable to get variable '%s'.\n", variable_name);
        return {};
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getMessageArrayVariable<T, N>(variable_name, cell_index, array_index);
    return value;
}
template<typename T, unsigned int N>
__device__ T MessageSpatial3D::In::WrapFilter::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (relative_cell[0] >= 2) {
        DTHROW("MessageSpatial3D in invalid bin, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getMessageVariable<T>(variable_name, cell_index);
    return value;
}
template<typename T, flamegpu::size_type N, unsigned int M> __device__
T MessageSpatial3D::In::WrapFilter::Message::getVariable(const char(&variable_name)[M], const unsigned int array_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Ensure that the message is within bounds.
    if (relative_cell[0] >= 2) {
        DTHROW("MessageSpatial3D in invalid bin, unable to get variable '%s'.\n", variable_name);
        return {};
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getMessageArrayVariable<T, N>(variable_name, cell_index, array_index);
    return value;
}


__device__ __forceinline__ MessageSpatial3D::GridPos3D getGridPosition3D(const MessageSpatial3D::MetaData *md, float x, float y, float z) {
    // Clamp each grid coord to 0<=x<dim
    int gridPos[3] = {
        static_cast<int>(floorf((x-md->min[0]) / md->radius)),
        static_cast<int>(floorf((y-md->min[1]) / md->radius)),
        static_cast<int>(floorf((z-md->min[2]) / md->radius))
    };
    MessageSpatial3D::GridPos3D rtn = {
        gridPos[0] < 0 ? 0 : (gridPos[0] >= static_cast<int>(md->gridDim[0]) ? static_cast<int>(md->gridDim[0]) - 1 : gridPos[0]),
        gridPos[1] < 0 ? 0 : (gridPos[1] >= static_cast<int>(md->gridDim[1]) ? static_cast<int>(md->gridDim[1]) - 1 : gridPos[1]),
        gridPos[2] < 0 ? 0 : (gridPos[2] >= static_cast<int>(md->gridDim[2]) ? static_cast<int>(md->gridDim[2]) - 1 : gridPos[2])
    };
    return rtn;
}
__device__ __forceinline__ unsigned int getHash3D(const MessageSpatial3D::MetaData *md, const MessageSpatial3D::GridPos3D &xyz) {
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

__device__ inline void MessageSpatial3D::Out::setLocation(const float x, const float y, const float z) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variables using curve
    detail::curve::DeviceCurve::setMessageVariable<float>("x", x, index);
    detail::curve::DeviceCurve::setMessageVariable<float>("y", y, index);
    detail::curve::DeviceCurve::setMessageVariable<float>("z", z, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}

__device__ inline MessageSpatial3D::In::Filter::Filter(const MetaData* _metadata, const float x, const float y, const float z)
    : metadata(_metadata) {
    loc[0] = x;
    loc[1] = y;
    loc[2] = z;
    cell = getGridPosition3D(_metadata, x, y, z);
}
__device__ inline MessageSpatial3D::In::Filter::Message& MessageSpatial3D::In::Filter::Message::operator++() {
    cell_index++;
    bool move_strip = cell_index >= cell_index_max;
    while (move_strip) {
        nextStrip();
        cell_index = 0;
        cell_index_max = 1;
        if (relative_cell[0] < 2) {
            // Calculate the strips start and end hash
            int absolute_cell[2] = { _parent.cell.y + relative_cell[0], _parent.cell.z + relative_cell[1] };
            // Skip the strip if it is completely out of bounds
            if (absolute_cell[0] >= 0 && absolute_cell[1] >= 0 && absolute_cell[0] < static_cast<int>(_parent.metadata->gridDim[1]) && absolute_cell[1] < static_cast<int>(_parent.metadata->gridDim[2])) {
                unsigned int start_hash = getHash3D(_parent.metadata, { _parent.cell.x - 1, absolute_cell[0], absolute_cell[1] });
                unsigned int end_hash = getHash3D(_parent.metadata, { _parent.cell.x + 1, absolute_cell[0], absolute_cell[1] });
                // Lookup start and end indicies from PBM
                cell_index = _parent.metadata->PBM[start_hash];
                cell_index_max = _parent.metadata->PBM[end_hash + 1];
            } else {
                // Goto next strip
                // Don't update move_strip
                continue;
            }
        }
        move_strip = cell_index >= cell_index_max;
    }
    return *this;
}
__device__ inline MessageSpatial3D::In::WrapFilter::WrapFilter(const MetaData* _metadata, const float x, const float y, const float z)
    : metadata(_metadata) {
    loc[0] = x;
    loc[1] = y;
    loc[2] = z;
    cell = getGridPosition3D(_metadata, x, y, z);
}
__device__ inline MessageSpatial3D::In::WrapFilter::Message& MessageSpatial3D::In::WrapFilter::Message::operator++() {
    cell_index++;
    bool move_strip = cell_index >= cell_index_max;
    while (move_strip) {
        nextCell();
        cell_index = 0;
        cell_index_max = 1;
        if (relative_cell[0] < 2) {
            // Calculate the strips start and end hash
            int absolute_cell_x = (_parent.cell.x + relative_cell[0] + static_cast<int>(_parent.metadata->gridDim[0])) % _parent.metadata->gridDim[0];
            int absolute_cell_y = (_parent.cell.y + relative_cell[1] + static_cast<int>(_parent.metadata->gridDim[1])) % _parent.metadata->gridDim[1];
            int absolute_cell_z = (_parent.cell.z + relative_cell[2] + static_cast<int>(_parent.metadata->gridDim[2])) % _parent.metadata->gridDim[2];
            // Skip the strip if it is completely out of bounds
            unsigned int start_hash = getHash3D(_parent.metadata, { absolute_cell_x, absolute_cell_y, absolute_cell_z });
            unsigned int end_hash = getHash3D(_parent.metadata, { absolute_cell_x, absolute_cell_y, absolute_cell_z });
            // Lookup start and end indicies from PBM
            cell_index = _parent.metadata->PBM[start_hash];
            cell_index_max = _parent.metadata->PBM[end_hash + 1];
        }
        move_strip = cell_index >= cell_index_max;
    }
    return *this;
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL3D_MESSAGESPATIAL3DDEVICE_CUH_

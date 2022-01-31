#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY3D_MESSAGEARRAY3DDEVICE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY3D_MESSAGEARRAY3DDEVICE_CUH_


#include "flamegpu/runtime/messaging/MessageArray3D.h"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"


namespace flamegpu {

/**
 * This class is accessible via DeviceAPI.message_in if MessageArray3D is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading three dimensional array messages
 */
class MessageArray3D::In {
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
        const MessageArray3D::In &_parent;
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
        __device__ Message(const MessageArray3D::In &parent, const size_type &_index) : _parent(parent), index(_index) {}
#if !defined(SEATBELTS) || SEATBELTS
        /**
         * A null message which always returns the message at index 0
         */
        __device__ Message(const MessageArray3D::In& parent) : _parent(parent), index(0) {}
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
        template<typename T, size_type N>
        __device__ T getVariable(const char(&variable_name)[N]) const;
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
        template<typename T, MessageNone::size_type N, unsigned int M>
        __device__ T getVariable(const char(&variable_name)[M], const unsigned int &index) const;
    };
    /**
     * This class is created when a search origin is provided to MessageArray2D::In::operator()(size_type, size_type, size_type = 1)
     * It provides iterator access to a subset of the full message list, according to the provided search origin and radius
     * The radius wraps the message list bounds
     * 
     * @see MessageArray2D::In::wrap()(size_type, size_type, size_type)
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
            __device__ Message(const WrapFilter&parent, const int &relative_x, const int &relative_y, const int &relative_z)
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
                return this->relative_cell[0] == rhs.relative_cell[0]
                    && this->relative_cell[1] == rhs.relative_cell[1]
                    && this->relative_cell[2] == rhs.relative_cell[2];
                    // && this->_parent.loc[0] == rhs._parent.loc[0]
                    // && this->_parent.loc[1] == rhs._parent.loc[1]
                    // && this->_parent.loc[2] == rhs._parent.loc[2];
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
            inline __device__ Message& operator++();
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
             * Returns the x array offset of message relative to the search origin
             * @note This value is unwrapped, so will always return a value within the search radius
             */
            __device__ size_type getOffsetX() const {
                return relative_cell[0];
            }
            /**
             * Returns the y array offset of message relative to the search origin
             * @note This value is unwrapped, so will always return a value within the search radius
             */
            __device__ size_type getOffsetY() const {
                return relative_cell[1];
            }
            /**
             * Returns the z array offset of message relative to the search origin
             * @note This value is unwrapped, so will always return a value within the search radius
             */
            __device__ size_type getOffsetZ() const {
                return relative_cell[2];
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
             * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
             */
            template<typename T, MessageNone::size_type N, unsigned int M>
            __device__ T getVariable(const char(&variable_name)[M], const unsigned int& index) const;
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
             * This iterator is constructed by MessageArray3D::In::WrapFilter::begin()(size_type, size_type, size_type, size_type)
             * @see MessageArray3D::In::wrap(size_type, size_type, size_type, size_type)
             */
            __device__ iterator(const WrapFilter&parent, const int &relative_x, const int &relative_y, const int &relative_z)
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
         * Constructor, takes the search parameters required
         * @param _metadata Pointer to message list metadata
         * @param _combined_hash agentfn+message hash for accessing message data
         * @param x Search origin x coord
         * @param y Search origin y coord
         * @param z Search origin z coord
         * @param _radius Search radius
         */
        inline __device__ WrapFilter(const MetaData *_metadata, const detail::curve::Curve::NamespaceHash &_combined_hash, const size_type &x, const size_type &y, const size_type &z, const size_type &_radius);
#if !defined(SEATBELTS) || SEATBELTS
        /**
         * A null filter which always returns 0 messages
         */
        inline __device__ WrapFilter();
#endif
        /**
         * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
#if !defined(SEATBELTS) || SEATBELTS
            if (!this->metadata)
                return iterator(*this, radius, radius, radius);
#endif
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
        detail::curve::Curve::NamespaceHash combined_hash;
    };
    /**
     * This class is created when a search origin is provided to MessageArray2D::In::operator()(size_type, size_type, size_type = 1)
     * It provides iterator access to a subset of the full message list, according to the provided search origin and radius
     * The radius does not wrap the message list bounds
     *
     * @see MessageArray2D::In::operator()(size_type, size_type, size_type)
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
             * Current offset
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
            __device__ Message(const Filter& parent, const int& relative_x, const int& relative_y, const int& relative_z)
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
                return this->relative_cell[0] == rhs.relative_cell[0]
                    && this->relative_cell[1] == rhs.relative_cell[1]
                    && this->relative_cell[2] == rhs.relative_cell[2];
                    // && this->_parent.loc[0] == rhs._parent.loc[0]
                    // && this->_parent.loc[1] == rhs._parent.loc[1]
                    // && this->_parent.loc[2] == rhs._parent.loc[2];
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
            inline __device__ Message& operator++();
            /**
             * Returns x array index of message
             */
            __device__ size_type getX() const {
                return this->_parent.loc[0] + relative_cell[0];
            }
            /**
             * Returns y array index of message
             */
            __device__ size_type getY() const {
                return this->_parent.loc[1] + relative_cell[1];
            }
            /**
             * Returns z array index of message
             */
            __device__ size_type getZ() const {
                return this->_parent.loc[2] + relative_cell[2];
            }
            /**
             * Returns the x array offset of message relative to the search origin
             */
            __device__ int getOffsetX() const {
                return this->relative_cell[0];
            }
            /**
             * Returns the y array offset of message relative to the search origin
             */
            __device__ int getOffsetY() const {
                return this->relative_cell[1];
            }
            /**
             * Returns the z array offset of message relative to the search origin
             */
            __device__ int getOffsetZ() const {
                return this->relative_cell[2];
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
             * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If T is not the type of variable 'name' within the message (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
             */
            template<typename T, MessageNone::size_type N, unsigned int M>
            __device__ T getVariable(const char(&variable_name)[M], const unsigned int& index) const;
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
             * This iterator is constructed by MessageArray3D::In::Filter::begin()(size_type, size_type, size_type, size_type)
             * @see MessageArray3D::In::Operator()(size_type, size_type, size_type, size_type)
             */
            __device__ iterator(const Filter& parent, const int& relative_x, const int& relative_y, const int& relative_z)
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
         * @param _metadata Pointer to message list metadata
         * @param _combined_hash agentfn+message hash for accessing message data
         * @param x Search origin x coord
         * @param y Search origin y coord
         * @param z Search origin z coord
         * @param _radius Search radius
         */
        inline __device__ Filter(const MetaData* _metadata, const detail::curve::Curve::NamespaceHash& _combined_hash, const size_type& x, const size_type& y, const size_type& z, const size_type& _radius);
#if !defined(SEATBELTS) || SEATBELTS
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
            return iterator(*this,  min_cell[0], min_cell[1], min_cell[2] - 1);
        }
        /**
         * Returns an iterator to the position beyond the end of the message list subset
         * @note This iterator is the same for all message list subsets
         */
        inline __device__ iterator end(void) const {
            // Final bin, as the constructor calls increment operator
            return iterator(*this, max_cell[0], max_cell[1], max_cell[2]);
        }

     private:
        /**
         * Search origin
         */
        size_type loc[3];
        /**
         * Min offset to be accessed (inclusive)
         */
        int min_cell[3];
        /**
         * Max offset to be accessed (inclusive)
         */
        int max_cell[3];
        /**
         * Pointer to message list metadata, e.g. environment bounds, search radius, PBM location
         */
        const MetaData* metadata;
        /**
         * CURVE hash for accessing message data
         * agent function hash + message hash
         */
        detail::curve::Curve::NamespaceHash combined_hash;
    };
    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to message_hash to produce combined_hash
     * @param message_hash Added to agentfn_hash to produce combined_hash
     * @param _metadata Reinterpreted as type MessageArray3D::MetaData
     */
    __device__ In(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash message_hash, const void *_metadata)
        : combined_hash(agentfn_hash + message_hash)
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
    { }
    /**
     * Returns a WrapFilter object which provides access to message iterator
     * for iterating a subset of messages including those within the radius of the search origin
     * this excludes the message at the search origin
     * The radius will wrap over environment bounds
     *
     * @param x Search origin x coord
     * @param y Search origin y coord
     * @param z Search origin y coord
     * @param radius Search radius
     * @note radius 1 is 26 cells in 3x3x3
     * @note radius 2 is 124 cells in 5x5x5
     * @note radius which produce a message read dimension (radius*2 + 1) greater than one of the message list dimensions are unsupported
     * @note radius of 0 is unsupported
     * @note The location [x, y, z] must be within the bounds of the message list
     */
    inline __device__ WrapFilter wrap(const size_type &x, const size_type &y, const size_type &z, const size_type &radius = 1) const {
#if !defined(SEATBELTS) || SEATBELTS
        if (radius == 0) {
            DTHROW("%u is not a valid radius for accessing Array3D message lists.\n", radius);
            return WrapFilter();
        } else if ((radius * 2) + 1 > metadata->dimensions[0] ||
                   (radius * 2) + 1 > metadata->dimensions[1] ||
                   (radius * 2) + 1 > metadata->dimensions[2]) {
            unsigned int min_r = metadata->dimensions[0] < metadata->dimensions[1] ? metadata->dimensions[0] : metadata->dimensions[1];
            min_r = min_r < metadata->dimensions[2] ? min_r : metadata->dimensions[2];
            min_r = min_r % 2 == 0 ? min_r - 2: min_r - 1;
            min_r /= 2;
            DTHROW("%u is not a valid radius for accessing Array3D message lists, as the diameter of messages accessed exceeds one or more of the message list dimensions (%u, %u, %u)."
            " Maximum supported radius for this message list is %u.\n",
            radius, metadata->dimensions[0], metadata->dimensions[1], metadata->dimensions[2], min_r);
            return WrapFilter();
        } else if (x >= metadata->dimensions[0] ||
                   y >= metadata->dimensions[1] ||
                   z >= metadata->dimensions[2]) {
            DTHROW("(%u, %u, %u) is not a valid position for iterating an Array3D message list of dimensions (%u, %u, %u), location must be within bounds.",
                x, y, z, metadata->dimensions[0], metadata->dimensions[1], metadata->dimensions[2]);
            return WrapFilter();
        }
#endif
        return WrapFilter(metadata, combined_hash, x, y, z, radius);
    }
    /**
     * Returns a Filter object which provides access to message iterator
     * for iterating a subset of messages including those within the radius of the search origin
     * this excludes the message at the search origin
     * The radius will not wrap over environment bounds
     *
     * @param x Search origin x coord
     * @param y Search origin y coord
     * @param z Search origin y coord
     * @param radius Search radius
     * @note radius 1 is 26 cells in 3x3x3
     * @note radius 2 is 124 cells in 5x5x5
     * @note radius of 0 is unsupported
     * @note The location [x, y, z] must be within the bounds of the message list
     */
    inline __device__ Filter operator()(const size_type& x, const size_type& y, const size_type& z, const size_type& radius = 1) const {
#if !defined(SEATBELTS) || SEATBELTS
        if (radius == 0) {
            DTHROW("%u is not a valid radius for accessing Array3D message lists.\n", radius);
            return Filter();
        } else if (x >= metadata->dimensions[0] ||
                   y >= metadata->dimensions[1] ||
                   z >= metadata->dimensions[2]) {
            DTHROW("(%u, %u, %u) is not a valid position for iterating an Array3D message list of dimensions (%u, %u, %u), location must be within bounds.",
                x, y, z, metadata->dimensions[0], metadata->dimensions[1], metadata->dimensions[2]);
            return Filter();
        }
#endif
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
#if !defined(SEATBELTS) || SEATBELTS
        if (x >= metadata->dimensions[0] || y >= metadata->dimensions[1] || z >= metadata->dimensions[2]) {
            DTHROW("Index is out of bounds for Array3D messagelist ([%u, %u, %u] >= [%u, %u, %u]).\n", x, y, z, metadata->dimensions[0], metadata->dimensions[1], metadata->dimensions[2]);
            return Message(*this);
        }
#endif
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
    detail::curve::Curve::NamespaceHash combined_hash;
    /**
     * Metadata struct for accessing messages
     */
    const MetaData * const metadata;
};

/**
 * This class is accessible via DeviceAPI.message_out if MessageArray3D is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting array messages
 */
class MessageArray3D::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to message_hash to produce combined_hash
     * @param message_hash Added to agentfn_hash to produce combined_hash
     * @param _metadata Message specialisation specific metadata struct (of type MessageArray3D::MetaData)
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash message_hash, const void *_metadata, unsigned int *scan_flag_messageOutput)
        : combined_hash(agentfn_hash + message_hash)
        , scan_flag(scan_flag_messageOutput)
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
    { }
    /**
     * Sets the array index to store the message in
     */
    inline __device__ void setIndex(const size_type &x, const size_type &y, const size_type &z) const;
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
     * CURVE hash for accessing message data
     * agentfn_hash + message_hash
     */
    detail::curve::Curve::NamespaceHash combined_hash;
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
__device__ T MessageArray3D::In::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.metadata->length) {
        DTHROW("Invalid Array3D message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    return detail::curve::Curve::getMessageVariable<T>(variable_name, this->_parent.combined_hash, index);
}
template<typename T, MessageNone::size_type N, unsigned int M> __device__
T MessageArray3D::In::Message::getVariable(const char(&variable_name)[M], const unsigned int& array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index >= this->_parent.metadata->length) {
        DTHROW("Invalid Array3D message, unable to get variable '%s'.\n", variable_name);
        return {};
    }
#endif
    // get the value from curve using the stored hashes and message index.
    T value = detail::curve::Curve::getMessageArrayVariable<T, N>(variable_name, this->_parent.combined_hash, index, array_index);
    return value;
}
template<typename T, unsigned int N>
__device__ T MessageArray3D::In::WrapFilter::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.metadata->length) {
        DTHROW("Invalid Array3D message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    return detail::curve::Curve::getMessageVariable<T>(variable_name, this->_parent.combined_hash, index_1d);
}
template<typename T, MessageNone::size_type N, unsigned int M> __device__
T MessageArray3D::In::WrapFilter::Message::getVariable(const char(&variable_name)[M], const unsigned int& array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.metadata->length) {
        DTHROW("Invalid Array3D message, unable to get variable '%s'.\n", variable_name);
        return {};
    }
#endif
    // get the value from curve using the stored hashes and message index.
    T value = detail::curve::Curve::getMessageArrayVariable<T, N>(variable_name, this->_parent.combined_hash, index, array_index);
    return value;
}
template<typename T, unsigned int N>
__device__ T MessageArray3D::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.metadata->length) {
        DTHROW("Invalid Array3D message, unable to get variable '%s'.\n", variable_name);
        return static_cast<T>(0);
    }
#endif
    // get the value from curve using the stored hashes and message index.
    return detail::curve::Curve::getMessageVariable<T>(variable_name, this->_parent.combined_hash, index_1d);
}
template<typename T, MessageNone::size_type N, unsigned int M> __device__
T MessageArray3D::In::Filter::Message::getVariable(const char(&variable_name)[M], const unsigned int& array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
#if !defined(SEATBELTS) || SEATBELTS
    // Ensure that the message is within bounds.
    if (index_1d >= this->_parent.metadata->length) {
        DTHROW("Invalid Array3D message, unable to get variable '%s'.\n", variable_name);
        return {};
    }
#endif
    // get the value from curve using the stored hashes and message index.
    T value = detail::curve::Curve::getMessageArrayVariable<T, N>(variable_name, this->_parent.combined_hash, index, array_index);
    return value;
}

template<typename T, unsigned int N>
__device__ void MessageArray3D::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in MessageArray3D::Out::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    detail::curve::Curve::setMessageVariable<T>(variable_name, combined_hash, value, index);

    // setIndex() sets the optional message scan flag
}
template<typename T, unsigned int N, unsigned int M>
__device__ void MessageArray3D::Out::setVariable(const char(&variable_name)[M], const unsigned int& array_index, T value) const {
    if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in MessageArray3D::Out::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    detail::curve::Curve::setMessageArrayVariable<T, N>(variable_name, combined_hash, value, index, array_index);

    // setIndex() sets the optional message scan flag
}

/**
* Sets the array index to store the message in
*/
__device__ inline void MessageArray3D::Out::setIndex(const size_type &x, const size_type &y, const size_type &z) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    size_type index_1d =
        z * metadata->dimensions[0] * metadata->dimensions[1] +
        y * metadata->dimensions[0] +
        x;
#if !defined(SEATBELTS) || SEATBELTS
    if (x >= metadata->dimensions[0] ||
        y >= metadata->dimensions[1] ||
        z >= metadata->dimensions[2]) {
        DTHROW("MessageArray3D index [%u, %u, %u] is out of bounds [%u, %u, %u]\n", x, y, z, metadata->dimensions[0], metadata->dimensions[1], metadata->dimensions[2]);
        return;  // Fail silently
    }
#endif

    // set the variable using curve
    detail::curve::Curve::setMessageVariable<size_type>("___INDEX", combined_hash, index_1d, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}
__device__ inline MessageArray3D::In::WrapFilter::WrapFilter(const MetaData *_metadata, const detail::curve::Curve::NamespaceHash &_combined_hash, const size_type &x, const size_type &y, const size_type &z, const size_type &_radius)
    : radius(_radius)
    , metadata(_metadata)
    , combined_hash(_combined_hash) {
    loc[0] = x;
    loc[1] = y;
    loc[2] = z;
}
#if !defined(SEATBELTS) || SEATBELTS
__device__ inline MessageArray3D::In::WrapFilter::WrapFilter()
    : radius(0)
    , metadata(nullptr)
    , combined_hash(0) {
    loc[0] = 0;
    loc[1] = 0;
    loc[2] = 0;
}
#endif
__device__ inline MessageArray3D::In::WrapFilter::Message& MessageArray3D::In::WrapFilter::Message::operator++() {
#if !defined(SEATBELTS) || SEATBELTS
    if (!_parent.metadata)
        return *this;
#endif
    if (relative_cell[2] >= static_cast<int>(_parent.radius)) {
        relative_cell[2] = -static_cast<int>(_parent.radius);
        if (relative_cell[1] >= static_cast<int>(_parent.radius)) {
            relative_cell[1] = -static_cast<int>(_parent.radius);
            relative_cell[0]++;
        } else {
            relative_cell[1]++;
        }
    } else {
        relative_cell[2]++;
    }
    // Skip origin cell
    if (relative_cell[0] == 0 && relative_cell[1] == 0 && relative_cell[2] == 0) {
        relative_cell[2]++;
    }
    // Wrap over boundaries
    const unsigned int their_x = (this->_parent.loc[0] + relative_cell[0] + this->_parent.metadata->dimensions[0]) % this->_parent.metadata->dimensions[0];
    const unsigned int their_y = (this->_parent.loc[1] + relative_cell[1] + this->_parent.metadata->dimensions[1]) % this->_parent.metadata->dimensions[1];
    const unsigned int their_z = (this->_parent.loc[2] + relative_cell[2] + this->_parent.metadata->dimensions[2]) % this->_parent.metadata->dimensions[2];
    // Solve to 1 dimensional bin index
    index_1d = their_z * this->_parent.metadata->dimensions[0] * this->_parent.metadata->dimensions[1] +
               their_y * this->_parent.metadata->dimensions[0] +
               their_x;
    return *this;
}
__device__ inline MessageArray3D::In::Filter::Filter(const MetaData* _metadata, const detail::curve::Curve::NamespaceHash& _combined_hash, const size_type& x, const size_type& y, const size_type& z, const size_type& _radius)
    : metadata(_metadata)
    , combined_hash(_combined_hash) {
    loc[0] = x;
    loc[1] = y;
    loc[2] = z;
    min_cell[0] = static_cast<int>(x) - static_cast<int>(_radius) < 0 ? -static_cast<int>(x) : - static_cast<int>(_radius);
    min_cell[1] = static_cast<int>(y) - static_cast<int>(_radius) < 0 ? -static_cast<int>(y) : - static_cast<int>(_radius);
    min_cell[2] = static_cast<int>(z) - static_cast<int>(_radius) < 0 ? -static_cast<int>(z) : - static_cast<int>(_radius);
    max_cell[0] = x + _radius >= _metadata->dimensions[0] ? static_cast<int>(_metadata->dimensions[0]) - 1 - static_cast<int>(x) : static_cast<int>(_radius);
    max_cell[1] = y + _radius >= _metadata->dimensions[1] ? static_cast<int>(_metadata->dimensions[1]) - 1 - static_cast<int>(y) : static_cast<int>(_radius);
    max_cell[2] = z + _radius >= _metadata->dimensions[2] ? static_cast<int>(_metadata->dimensions[2]) - 1 - static_cast<int>(z) : static_cast<int>(_radius);
}
#if !defined(SEATBELTS) || SEATBELTS
__device__ inline MessageArray3D::In::Filter::Filter()
    : metadata(nullptr)
    , combined_hash(0) {
    loc[0] = 0;
    loc[1] = 0;
    loc[2] = 0;
    min_cell[0] = 0;
    min_cell[1] = 0;
    min_cell[2] = 1;
    max_cell[0] = 0;
    max_cell[1] = 0;
    max_cell[2] = 0;
}
#endif
__device__ inline MessageArray3D::In::Filter::Message& MessageArray3D::In::Filter::Message::operator++() {
#if !defined(SEATBELTS) || SEATBELTS
    if (!_parent.metadata)
        return *this;
#endif
    if (relative_cell[2] >= _parent.max_cell[2]) {
        relative_cell[2] = _parent.min_cell[2];
        if (relative_cell[1] >= _parent.max_cell[1]) {
            relative_cell[1] = _parent.min_cell[1];
            relative_cell[0]++;
        } else {
            relative_cell[1]++;
        }
    } else {
        relative_cell[2]++;
    }
    // Skip origin cell
    if (relative_cell[0] == 0 && relative_cell[1] == 0 && relative_cell[2] == 0) {
        if (relative_cell[2] >= _parent.max_cell[2]) {
            relative_cell[2] = _parent.min_cell[2];
            if (relative_cell[1] >= _parent.max_cell[1]) {
                relative_cell[1] = _parent.min_cell[1];
                relative_cell[0]++;
            } else {
                relative_cell[1]++;
            }
        } else {
            relative_cell[2]++;
        }
    }
    // Solve to 1 dimensional bin index
    index_1d = (this->_parent.loc[2] + relative_cell[2]) * this->_parent.metadata->dimensions[0] * this->_parent.metadata->dimensions[1] +
               (this->_parent.loc[1] + relative_cell[1]) * this->_parent.metadata->dimensions[0] +
               (this->_parent.loc[0] + relative_cell[0]);
    return *this;
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEARRAY3D_MESSAGEARRAY3DDEVICE_CUH_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL2D_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL2D_H_

#include <memory>
#include <string>

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"

#include "flamegpu/runtime/cuRVE/curve.h"
/**
 * 2D Continuous spatial messaging functionality
 *
 * User specifies the environment bounds and search radius
 * When accessing messages, a search origin is specified
 * A subset of messages, including those within radius of the search origin are returned
 * The user must distance check that they fall within the search radius manually
 * Unlike FLAMEGPU1, these spatial messages do not wrap over environment bounds.
 */
class MsgSpatial2D {
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
     * Basic class to group 3 dimensional bin coordinates
     * Would use glm::ivec3, but project does not currently have glm
     */
    struct GridPos2D {
        int x, y;
    };

    /**
     * MetaData required by spatial partitioning during message reads
     */
    struct MetaData {
        /**
         * Minimum environment bounds
         */
        float min[2];
        /**
         * Maximum environment bounds
         */
        float max[2];
        /**
         * Search radius (also used as subdividision bin width)
         */
        float radius;
        /**
         * Pointer to the partition boundary matrix in device memory
         * The PBM is never stored on the host
         */
        unsigned int *PBM;
        /**
         * The number of subdividision bins in each dimensions
         */
        unsigned int gridDim[2];
        /**
         * max-min
         */
        float environmentWidth[3];
    };

    /**
     * This class is accessible via FLAMEGPU_DEVICE_API.message_in if MsgSpatial3D is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for reading spatially partitioned messages
     */
    class In {
     public:
        /**
         * This class is created when a search origin is provided to MsgSpatial2D::In::operator()(float, float)
         * It provides iterator access to a subset of the full message list, according to the provided search origin
         * 
         * @see MsgSpatial2D::In::operator()(float, float)
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
                 * Relative strip within the Moore neighbourhood
                 * Strips run along the x axis
                 * relative_cell corresponds to y offset
                 */
                int relative_cell = { -2 };
                /**
                 * This is the index after the final message, relative to the full message list, in the current bin
                 */
                int cell_index_max = 0;
                /**
                 * This is the index of the currently accessed message, relative to the full message list
                 */
                int cell_index = 0;

             public:
                /**
                 * Constructs a message and directly initialises all of it's member variables
                 * @note See member variable documentation for their purposes
                 */
                __device__ Message(const Filter &parent, const int &relative_cell_y, const int &_cell_index_max, const int &_cell_index)
                    : _parent(parent)
                    , cell_index_max(_cell_index_max)
                    , cell_index(_cell_index) {
                    relative_cell = relative_cell_y;
                }
                /**
                 * Equality operator
                 * Compares all internal member vars for equality
                 * @note Does not compare _parent
                 */
                __device__ bool operator==(const Message& rhs) const {
                    return this->relative_cell == rhs.relative_cell
                        && this->cell_index_max == rhs.cell_index_max
                        && this->cell_index == rhs.cell_index;
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
                __device__ Message& operator++();
                /**
                 * Utility function for deciding next strip to access
                 */
                __device__ void nextStrip() {
                    relative_cell++;
                }
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
             * Stock iterator for iterating MsgSpatial3D::In::Filter::Message objects
             */
            class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
                /**
                 * The message returned to the user
                 */
                Message _message;

             public:
                /**
                 * Constructor
                 * This iterator is constructed by MsgSpatial2D::In::Filter::begin()(float, float)
                 * @see MsgSpatial2D::In::Operator()(float, float)
                 */
                __device__ iterator(const Filter &parent, const int &relative_cell_y, const int &_cell_index_max, const int &_cell_index)
                    : _message(parent, relative_cell_y, _cell_index_max, _cell_index) {
                    // Increment to find first message
                    ++_message;
                }
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
                __device__ Message& operator*() { return _message; }
            };
            /**
             * Constructor, takes the search parameters requried
             * @param _metadata Pointer to message list metadata
             * @param combined_hash agentfn+message hash for accessing message data
             * @param x Search origin x coord
             * @param y Search origin y coord
             */
            __device__ Filter(const MetaData *_metadata, const Curve::NamespaceHash &combined_hash, const float &x, const float &y);
            /**
             * Returns an iterator to the start of the message list subset about the search origin
             */
            inline __device__ iterator begin(void) const {
                // Bin before initial bin, as the constructor calls increment operator
                return iterator(*this, -2, 1, 0);
            }
            /**
             * Returns an iterator to the position beyond the end of the message list subset
             * @note This iterator is the same for all message list subsets
             */
            inline __device__ iterator end(void) const {
                // Final bin, as the constructor calls increment operator
                return iterator(*this, 1, 1, 0);
            }

         private:
            /**
             * Search origin
             */
            float loc[2];
            /**
             * Search origin's grid cell
             */
            GridPos2D cell;
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
         * @param _metadata Reinterpreted as type MsgSpatial3D::MetaData
         */
        __device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *_metadata)
            : combined_hash(agentfn_hash + msg_hash)
            , metadata(reinterpret_cast<const MetaData*>(_metadata))
        { }
        /**
         * Returns a Filter object which provides access to message iterator
         * for iterating a subset of messages including those within the radius of the search origin
         *
         * @param x Search origin x coord
         * @param y Search origin y coord
         */
         inline __device__ Filter operator() (const float &x, const float &y) const {
             return Filter(metadata, combined_hash, x, y);
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
     * This class is accessible via FLAMEGPU_DEVICE_API.message_out if MsgSpatial3D is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for outputting spatially partitioned messages
     */
    class Out : public MsgBruteForce::Out {
     public:
        /**
         * Constructer
         * Initialises member variables
         * @param agentfn_hash Added to msg_hash to produce combined_hash
         * @param msg_hash Added to agentfn_hash to produce combined_hash
         * @param _streamId Stream index, used for optional message output flag array
         */
        __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, unsigned int _streamId)
            : MsgBruteForce::Out(agentfn_hash, msg_hash, _streamId)
        { }
        /**
         * Sets the location for this agents message
         * @param x Message x coord
         * @param y Message y coord
         * @note Convenience wrapper for setVariable()
         */
        __device__ void setLocation(const float &x, const float &y) const;
    };

    /**
     * CUDA host side handler of spatial messages
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
         * Frees all alocated memory
         */
         ~CUDAModelHandler() override;
        /**
         * Reconstructs the partition boundary matrix
         * This should be called before reading newly output messages
         */
        void buildIndex() override;
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
         * Number of bins, arrays are +1 this length
         */
        unsigned int binCount = 0;
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
     * Internal data representation of Spatial2D messages within model description hierarchy
     * @see Description
     */
    struct Data : public MsgBruteForce::Data {
        friend class ModelDescription;
        friend struct ModelData;
        float radius;
        float minX;
        float minY;
        float maxX;
        float maxY;
        virtual ~Data() = default;

        std::unique_ptr<MsgSpecialisationHandler> getSpecialisationHander(CUDAMessage &owner) const override;

        /**
         * Used internally to validate that the corresponding Msg type is attached via the agent function shim.
         * @return The std::type_index of the Msg type which must be used.
         */
        std::type_index getType() const override;

     protected:
         Data *clone(ModelData *const newParent) override;
        /**
         * Copy constructor
         * This is unsafe, should only be used internally, use clone() instead
         */
         Data(ModelData *const, const Data &other);
        /**
         * Normal constructor, only to be called by ModelDescription
         */
         Data(ModelData *const, const std::string &message_name);
    };

    /**
     * User accessible interface to Spatial2D messages within mode description hierarchy
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
         Description(ModelData *const _model, Data *const data);
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
        void setRadius(const float &r);
        void setMinX(const float &x);
        void setMinY(const float &y);
        void setMin(const float &x, const float &y);
        void setMaxX(const float &x);
        void setMaxY(const float &y);
        void setMax(const float &x, const float &y);

        float &Radius();
        float &MinX();
        float &MinY();
        float &MaxX();
        float &MaxY();

        float getRadius() const;
        float getMinX() const;
        float getMinY() const;
        float getMaxX() const;
        float getMaxY() const;
    };
};

#ifdef __CUDACC__
template<typename T, unsigned int N>
__device__ T MsgSpatial2D::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
    //// Ensure that the message is within bounds.
    if (relative_cell < 2) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, cell_index);
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}
#endif  // __CUDACC__

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL2D_H_

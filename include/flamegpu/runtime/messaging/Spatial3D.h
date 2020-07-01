#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3D_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3D_H_

#ifndef __CUDACC_RTC__
#include <memory>
#include <string>

#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/util/nvtx.h"
#endif  // __CUDACC_RTC__

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial2D.h"

/**
 * 3D Continuous spatial messaging functionality
 *
 * User specifies the environment bounds and search radius
 * When accessing messages, a search origin is specified
 * A subset of messages, including those within radius of the search origin are returned
 * The user must distance check that they fall within the search radius manually
 * Unlike FLAMEGPU1, these spatial messages do not wrap over environment bounds.
 */
class MsgSpatial3D {
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
    struct GridPos3D {
        int x, y, z;
    };
    /**
     * MetaData required by spatial partitioning during message reads
     */
    struct MetaData {
        /**
         * Minimum environment bounds
         */
        float min[3];
        /**
         * Maximum environment bounds
         */
        float max[3];
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
        unsigned int gridDim[3];
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
         * This class is created when a search origin is provided to MsgSpatial3D::In::operator()(float, float, float)
         * It provides iterator access to a subset of the full message list, according to the provided search origin
         * 
         * @see MsgSpatial3D::In::operator()(float, float, float)
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

             public:
                /**
                 * Constructs a message and directly initialises all of it's member variables
                 * @note See member variable documentation for their purposes
                 */
                __device__ Message(const Filter &parent, const int &relative_cell_y, const int &relative_cell_z, const int &_cell_index_max, const int &_cell_index)
                    : _parent(parent)
                    , cell_index_max(_cell_index_max)
                    , cell_index(_cell_index) {
                    relative_cell[0] = relative_cell_y;
                    relative_cell[1] = relative_cell_z;
                }
                /**
                 * Equality operator
                 * Compares all internal member vars for equality
                 * @note Does not compare _parent
                 */
                __device__ bool operator==(const Message& rhs) const {
                    return this->relative_cell[0] == rhs.relative_cell[0]
                        && this->relative_cell[1] == rhs.relative_cell[1]
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
                    if (relative_cell[1] >= 1) {
                        relative_cell[1] = -1;
                        relative_cell[0]++;
                    } else {
                        relative_cell[1]++;
                    }
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
            class iterator {  // class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
                /**
                 * The message returned to the user
                 */
                Message _message;

             public:
                /**
                 * Constructor
                 * This iterator is constructed by MsgSpatial3D::In::Filter::begin()(float, float, float)
                 * @see MsgSpatial3D::In::Operator()(float, float, float)
                 */
                __device__ iterator(const Filter &parent, const int &relative_cell_y, const int &relative_cell_z, const int &_cell_index_max, const int &_cell_index)
                    : _message(parent, relative_cell_y, relative_cell_z, _cell_index_max, _cell_index) {
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
             * @param combined_hash agentfn+message hash for accessing message data
             * @param x Search origin x coord
             * @param y Search origin y coord
             * @param z search origin z coord
             */
            __device__ Filter(const MetaData *_metadata, const Curve::NamespaceHash &combined_hash, const float &x, const float &y, const float &z);
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
                // Final bin, as the constructor calls increment operator
                return iterator(*this, 1, 1, 1, 0);
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
         * @param z Search origin z coord
         */
        inline __device__ Filter operator() (const float &x, const float &y, const float &z) const {
            return Filter(metadata, combined_hash, x, y, z);
        }

        /**
         * Returns the search radius of the message list defined in the model description
         */
        __forceinline__ __device__ float radius() const {
            return metadata->radius;
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
         * @param scan_flag_messageOutput Scan flag array for optional message output
         */
        __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *, unsigned int *scan_flag_messageOutput)
            : MsgBruteForce::Out(agentfn_hash, msg_hash, nullptr, scan_flag_messageOutput)
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
#ifndef __CUDACC_RTC__
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
        explicit CUDAModelHandler(CUDAMessage &a)
         : MsgSpecialisationHandler()
         , sim_message(a) {
            NVTX_RANGE("Spatial3D::CUDAModelHandler");
            const Data &d = (const Data &)a.getMessageDescription();
            hd_data.radius = d.radius;
            hd_data.min[0] = d.minX;
            hd_data.min[1] = d.minY;
            hd_data.min[2] = d.minZ;
            hd_data.max[0] = d.maxX;
            hd_data.max[1] = d.maxY;
            hd_data.max[2] = d.maxZ;
            binCount = 1;
            for (unsigned int axis = 0; axis < 3; ++axis) {
                hd_data.environmentWidth[axis] = hd_data.max[axis] - hd_data.min[axis];
                hd_data.gridDim[axis] = static_cast<unsigned int>(ceil(hd_data.environmentWidth[axis] / hd_data.radius));
                binCount *= hd_data.gridDim[axis];
            }
            // Device allocation occurs in allocateMetaDataDevicePtr rather than the constructor.
        }
        /**
         * Destructor
         * Frees all alocated memory
         */
        ~CUDAModelHandler() override { }
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
     * Internal data representation of Spatial3D messages within model description hierarchy
     * @see Description
     */
    struct Data : public MsgSpatial2D::Data {
        friend class ModelDescription;
        friend struct ModelData;
        float minZ;
        float maxZ;
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
     * User accessible interface to Spatial3D messages within mode description hierarchy
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
        void setRadius(const float &r);
        void setMinX(const float &x);
        void setMinY(const float &y);
        void setMinZ(const float &z);
        void setMin(const float &x, const float &y, const float &z);
        void setMaxX(const float &x);
        void setMaxY(const float &y);
        void setMaxZ(const float &z);
        void setMax(const float &x, const float &y, const float &z);

        float getRadius() const;
        float getMinX() const;
        float getMinY() const;
        float getMinZ() const;
        float getMaxX() const;
        float getMaxY() const;
        float getMaxZ() const;
    };
#endif  // __CUDACC_RTC__
};


template<typename T, unsigned int N>
__device__ T MsgSpatial3D::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
    //// Ensure that the message is within bounds.
    if (relative_cell[0] < 2) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, cell_index);
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3D_H_

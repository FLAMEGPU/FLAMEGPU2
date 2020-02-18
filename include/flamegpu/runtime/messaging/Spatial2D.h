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
        // TODO
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
         // inline __device__ Filter operator() (const float &x, const float &y, const float &z) const {
         //    return Filter(metadata, combined_hash, x, y, z);
         // }
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
    class Out {
     public:
        /**
         * Constructer
         * Initialises member variables
         * @param agentfn_hash Added to msg_hash to produce combined_hash
         * @param msg_hash Added to agentfn_hash to produce combined_hash
         * @param _streamId Stream index, used for optional message output flag array
         */
        __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, unsigned int _streamId)
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
        /**
         * Sets the location for this agents message
         * @param x Message x coord
         * @param y Message y coord
         * @note Convenience wrapper for setVariable()
         */
        __device__ void setLocation(const float &x, const float &y) const;

     private:
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
// template<typename T, unsigned int N>
// __device__ T MsgSpatial2D::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
//    //// Ensure that the message is within bounds.
//    if (relative_cell < 2) {
//        // get the value from curve using the stored hashes and message index.
//        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, cell_index);
//        return value;
//    } else {
//        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
//        return static_cast<T>(0);
//    }
// }

/**
 * \brief adds a message
 * \param variable_name Name of message variable to set
 * \param value Value to set it to
 */
template<typename T, unsigned int N>
__device__ void MsgSpatial2D::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variable using curve
    Curve::setVariable<T>(variable_name, combined_hash, value, index);

    // Don't bother, handled by setLocation
    // Set scan flag incase the message is optional
    flamegpu_internal::CUDAScanCompaction::ds_message_configs[streamId].scan_flag[index] = 1;
}
#endif  // __CUDACC__

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL2D_H_

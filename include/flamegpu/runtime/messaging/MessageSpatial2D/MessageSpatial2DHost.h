#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL2D_MESSAGESPATIAL2DHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL2D_MESSAGESPATIAL2DHOST_H_

#include <memory>
#include <string>

#include "flamegpu/model/Variable.h"
#include "flamegpu/runtime/messaging/MessageSpatial2D.h"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceHost.h"

namespace flamegpu {


/**
 * CUDA host side handler of spatial messages
 * Allocates memory for and constructs PBM
 */
class MessageSpatial2D::CUDAModelHandler : public MessageSpecialisationHandler {
 public:
    /**
     * Constructor
     * 
     * Initialises metadata, decides PBM size etc
     * 
     * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
     */
     explicit CUDAModelHandler(detail::CUDAMessage &a);
    /**
     * Destructor
     * Frees all alocated memory
     */
     ~CUDAModelHandler() override;
    /**
     * Allocates memory for the constructed index.
     * Sets data asthough message list is empty
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId Index of stream specific structures used
     * @param stream The CUDAStream to use for CUDA operations
     */
    void init(detail::CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) override;
    /**
     * Reconstructs the partition boundary matrix
     * This should be called before reading newly output messages
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream The CUDAStream to use for CUDA operations
     */
    void buildIndex(detail::CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) override;
    /**
     * Allocates memory for the constructed index.
     * The memory allocation is checked by build index.
     */
    void allocateMetaDataDevicePtr(cudaStream_t stream) override;
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
    void resizeCubTemp(cudaStream_t stream);
    /**
     * Resizes the key value store, this scales with agent count
     * @param newSize The new number of agents to represent
     * @note This only scales upwards, it will never reduce the size
     */
    void resizeKeysVals(unsigned int newSize);
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
    detail::CUDAMessage &sim_message;
};

/**
 * Internal data representation of Spatial2D messages within model description hierarchy
 * @see Description
 */
struct MessageSpatial2D::Data : public MessageBruteForce::Data {
    friend class ModelDescription;
    friend struct ModelData;
    float radius;
    float minX;
    float minY;
    float maxX;
    float maxY;
    virtual ~Data() = default;

    std::unique_ptr<MessageSpecialisationHandler> getSpecialisationHander(detail::CUDAMessage &owner) const override;

    /**
     * Used internally to validate that the corresponding Message type is attached via the agent function shim.
     * @return The std::type_index of the Message type which must be used.
     */
    std::type_index getType() const override;
    /**
     * Return the sorting type for this message type
     */
    flamegpu::MessageSortingType getSortingType() const override;

 protected:
    Data *clone(const std::shared_ptr<const ModelData> &newParent) override;
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     */
    Data(std::shared_ptr<const ModelData> model, const Data &other);
    /**
     * Normal constructor, only to be called by ModelDescription
     */
    Data(std::shared_ptr<const ModelData> model, const std::string &message_name);
};

class MessageSpatial2D::CDescription : public MessageBruteForce::CDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct Data;

 public:
    /**
     * Constructor, creates an interface to the MessageData
     * @param data Data store of this message's data
     */
    explicit CDescription(std::shared_ptr<Data> data);
    explicit CDescription(std::shared_ptr<const Data> data);
    /**
     * Copy constructor
     * Creates a new interface to the same MessageData/ModelData
     */
    CDescription(const CDescription& other_agent) = default;
    CDescription(CDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same MessageData/ModelData
     */
    CDescription& operator=(const CDescription& other_agent) = default;
    CDescription& operator=(CDescription&& other_agent) = default;
    /**
     * Equality operator, checks whether message Description hierarchies are functionally the same
     * @param rhs right hand side
     * @returns True when messages are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CDescription& rhs) const;
    /**
     * Equality operator, checks whether message Description hierarchies are functionally different
     * @param rhs right hand side
     * @returns True when messages are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CDescription& rhs) const;

    float getRadius() const;
    float getMinX() const;
    float getMinY() const;
    float getMaxX() const;
    float getMaxY() const;

 protected:
    void setRadius(float r);
    void setMinX(float x);
    void setMinY(float y);
    void setMin(float x, float y);
    void setMaxX(float x);
    void setMaxY(float y);
    void setMax(float x, float y);
};
/**
 * User accessible interface to Spatial2D messages within mode description hierarchy
 * @see Data
 */
class MessageSpatial2D::Description : public CDescription {
 public:
    /**
     * Constructor, creates an interface to the MessageData
     * @param data Data store of this agent's data
     */
    explicit Description(std::shared_ptr<Data> data);
    /**
     * Copy constructor
     * Creates a new interface to the same MessageData/ModelData
     */
    Description(const Description& other_message) = default;
    Description(Description && other_message) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same MessageData/ModelData
     */
    Description& operator=(const Description & other_message) = default;
    Description& operator=(Description && other_message) = default;

    using MessageBruteForce::CDescription::setPersistent;
    using MessageBruteForce::CDescription::newVariable;
#ifdef SWIG
    using MessageBruteForce::CDescription::newVariableArray;
#endif

    using MessageSpatial2D::CDescription::setRadius;
    using MessageSpatial2D::CDescription::setMinX;
    using MessageSpatial2D::CDescription::setMinY;
    using MessageSpatial2D::CDescription::setMin;
    using MessageSpatial2D::CDescription::setMaxX;
    using MessageSpatial2D::CDescription::setMaxY;
    using MessageSpatial2D::CDescription::setMax;
};

}  // namespace flamegpu


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL2D_MESSAGESPATIAL2DHOST_H_

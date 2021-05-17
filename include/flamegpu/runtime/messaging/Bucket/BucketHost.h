#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_BUCKETHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_BUCKETHOST_H_

#include <string>
#include <memory>

#include "flamegpu/model/Variable.h"
#include "flamegpu/runtime/messaging/Bucket.h"
#include "flamegpu/runtime/messaging/BruteForce/BruteForceHost.h"

namespace flamegpu {

/**
* CUDA host side handler of bucket messages
* Allocates memory for and constructs PBM
*/
class MsgBucket::CUDAModelHandler : public MsgSpecialisationHandler {
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
    * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
    * @param streamId Index of stream specific structures used
    */
    void init(CUDAScatter &scatter, const unsigned int &streamId) override;
    /**
     * Reconstructs the partition boundary matrix
     * This should be called before reading newly output messages
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void buildIndex(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) override;
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
struct MsgBucket::Data : public MsgBruteForce::Data {
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
class MsgBucket::Description : public MsgBruteForce::Description {
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

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BUCKET_BUCKETHOST_H_

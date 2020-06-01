#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY2D_ARRAY2DHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY2D_ARRAY2DHOST_H_


#include <string>
#include <memory>
#include <array>

#include "flamegpu/model/Variable.h"
#include "flamegpu/runtime/messaging/Array2D.h"
#include "flamegpu/runtime/messaging/BruteForce/BruteForceHost.h"


/**
 * Blank handler, brute force requires no index or special allocations
 * Only stores the length on device
 */
class MsgArray2D::CUDAModelHandler : public MsgSpecialisationHandler {
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
struct MsgArray2D::Data : public MsgBruteForce::Data {
    friend class ModelDescription;
    friend struct ModelData;
    std::array<size_type, 2> dimensions;
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
 * User accessible interface to Array messages within mode description hierarchy
 * @see Data
 */
class MsgArray2D::Description : public MsgBruteForce::Description {
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
     void setDimensions(const size_type &len_x, const size_type &len_y);
     void setDimensions(const std::array<size_type, 2> &dims);

    std::array<size_type, 2> getDimensions() const;
    size_type getDimX() const;
    size_type getDimY() const;
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_ARRAY2D_ARRAY2DHOST_H_

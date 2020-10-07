#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEHOST_H_

#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/runtime/messaging/None.h"

class CUDAMessage;
class CUDAScatter;
/**
 * Interface for message specialisation
 * A derived implementation of this is required for each combination of message type (e.g. MsgBruteForce) and simulation type (e.g. CUDASimulation)
 * @note It is recommended that derived classes require an object that provides access to the model specialisation's representation of messages (e.g. CUDAMessage)
 * @note: this is slightly CUDA aware. Future abstraction DevicePtr should be in a CUDANone message or similar.
 */
class MsgSpecialisationHandler {
 public:
    MsgSpecialisationHandler() { }
    /**
     * Destructor, should free any allocated memory in derived classes
     */
    virtual ~MsgSpecialisationHandler() { }
    /**
     * Allocate and fill metadata, as though message list was empty
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId Index of stream specific structures used
     */
    virtual void init(CUDAScatter &scatter, const unsigned int &streamId) = 0;
    /**
     * Constructs an index for the message data structure (e.g. Partition boundary matrix for spatial message types)
     * This is called the first time messages are read, after new messages have been output
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId Index of stream specific structures used
     */
    virtual void buildIndex(CUDAScatter &, const unsigned int &) { }
    /**
     * Allocates memory for the constructed index.
     * The memory allocation is checked by build index.
     */
    virtual void allocateMetaDataDevicePtr() { }
    /**
     * Releases memory for the constructed index.
     */
    virtual void freeMetaDataDevicePtr() { }
    /**
     * Returns a pointer to metadata for message access during agent functions
     * (For CUDASimulation this is a device pointer)
     * @note: this is slightly CUDA aware. Future abstraction this should be base CUDANone or similar.
     */
    virtual const void *getMetaDataDevicePtr() const { return nullptr; }
};

/**
 * Provides specialisation behaviour for messages between agent functions
 * e.g. allocates/initialises additional data structure memory, sorts messages and builds an index
 * Created and owned by CUDAMessage
 */
class MsgNone::CUDAModelHandler : public MsgSpecialisationHandler {
 public:
    /**
     * Constructor
     */
    explicit CUDAModelHandler(CUDAMessage &a)
        : MsgSpecialisationHandler()
        , sim_message(a)
    { }
    /**
     * Owning CUDAMessage
     */
    CUDAMessage &sim_message;
};


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_NONEHOST_H_

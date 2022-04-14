#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPECIALISATIONHANDLER_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPECIALISATIONHANDLER_H_

#include "flamegpu/runtime/detail/curve/curve.cuh"

namespace flamegpu {

class CUDAScatter;
/**
 * Interface for message specialisation
 * A derived implementation of this is required for each combination of message type (e.g. MessageBruteForce) and simulation type (e.g. CUDASimulation)
 * @note It is recommended that derived classes require an object that provides access to the model specialisation's representation of messages (e.g. CUDAMessage)
 * @note: this is slightly CUDA aware. Future abstraction DevicePtr should be in a CUDANone message or similar.
 */
class MessageSpecialisationHandler {
 public:
    MessageSpecialisationHandler() { }
    /**
     * Destructor, should free any allocated memory in derived classes
     */
    virtual ~MessageSpecialisationHandler() { }
    /**
     * Allocate and fill metadata, as though message list was empty
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId Index of stream specific structures used
     * @param stream The CUDAStream to use for CUDA operations
     */
    virtual void init(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) = 0;
    /**
     * Constructs an index for the message data structure (e.g. Partition boundary matrix for spatial message types)
     * This is called the first time messages are read, after new messages have been output
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream The CUDAStream to use for CUDA operations
     */
    virtual void buildIndex(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) { }
    /**
     * Allocates memory for the constructed index.
     * The memory allocation is checked by build index.
     */
    virtual void allocateMetaDataDevicePtr(cudaStream_t stream) { }
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

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPECIALISATIONHANDLER_H_

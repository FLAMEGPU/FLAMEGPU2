#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_H_

#include "flamegpu/runtime/cuRVE/curve.h"

/**
 * Forward declaration, used for CUDASpecialisationHandler
 */
class CUDAMessage;
/**
 * Interface for message specialisation
 * A derived implementation of this is required for each combination of message type (e.g. MsgBruteForce) and simulation type (e.g. CUDAAgentModel)
 * @note It is recommended that derrived classes require an object that provides access to the model specialisation's representation of messages (e.g. CUDAMessage)
 */
class MsgSpecialisationHandler {
 public:
    MsgSpecialisationHandler() { }
    /**
     * Destructor, should free any allocated memory in derived classes
     */
    virtual ~MsgSpecialisationHandler() { }
    /**
     * Constructs an index for the message data structure (e.g. Partition boundary matrix for spatial message types)
     * This is called the first time messages are read, after new messages have been output
     */
    virtual void buildIndex() { }
    /**
     * Returns a pointer to metadata for message access during agent functions
     * (For CUDAAgentModel this is a device pointer)
     */
    virtual const void *getMetaDataDevicePtr() const { return nullptr; }
};

/**
 * This empty class is used when messaging is not enabled for an agent function
 * It also provides the best overview of the required components of a new messsaging type
 */
class MsgNone {
 public:
    /**
     * Common size type
     */
    typedef unsigned int size_type;
    /**
     * Provides message input functionality during agent functions
     * Constructed and owned by FLAMEGPU_DEVICE_API
     */
    class In {
     public:
        /**
         * Constructor
         * Requires CURVE hashes for agent function and message name to retrieve variable memory locations
         * Takes a device pointer to a struct for metadata related to accessing the messages (e.g. an index data structure)
         */
        __device__ In(Curve::NamespaceHash /*agent fn hash*/, Curve::NamespaceHash /*message name hash*/, const void * /*metadata*/) {
        }
    };
    /**
     * Provides message output functionality during agent functions
     * Constructed and owned by FLAMEGPU_DEVICE_API
     */
    class Out {
     public:
        /**
         * Constructor
         * Requires CURVE hashes for agent function and message name to retrieve variable memory locations
         * Takes a device pointer to a struct for metadata related to accessing the messages (e.g. an index data structure)
         */
        __device__ Out(Curve::NamespaceHash /*agent fn hash*/, Curve::NamespaceHash /*message name hash*/, unsigned int /*streamid*/){
        }
    };
    /**
     * Provides specialisation behaviour for messages between agent functions
     * e.g. allocates/initialises additional data structure memory, sorts messages and builds an index
     * Created and owned by CUDAMessage
     */
    class CUDAModelHandler : public MsgSpecialisationHandler {
     public:
        /**
         * Constructur
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
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_NONE_H_

#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

/**
 * This file provides the root include for messaging
 * The bulk of message specialisation is implemented within headers included at the bottom of this file
 */

/**
 * Forward declaration, used for CUDASpecialisationHandler
 */
class CUDAMessage;
/**
 * Interface for message specialisation
 * A derived implementation of this is required for each combination of message type (e.g. MsgBruteForce) and simulation type (e.g. CUDAAgentModel)
 * @tparam SimSpecialisationMsg The simulation type (e.g. CUDAAgentModel)
 */
template<typename SimSpecialisationMsg>
class MsgSpecialisationHandler {
 public:
    explicit MsgSpecialisationHandler(SimSpecialisationMsg &_sim_message)
        : sim_message(_sim_message)
    { }
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

 protected:
    /**
     * Provides access to the model specialisation's representation of messages (e.g. CUDAMessage)
     */
    SimSpecialisationMsg &sim_message;
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
     * @tparam SimSpecialisationMsg Always CUDAMessage
     */
    template<typename SimSpecialisationMsg>
    class CUDAModelHandler : public MsgSpecialisationHandler<SimSpecialisationMsg> {
     public:
        /**
         * Constructur
         */
        explicit CUDAModelHandler(CUDAMessage &a)
            : MsgSpecialisationHandler<SimSpecialisationMsg>(a)
        { }
    };
};

#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial3D.h"

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

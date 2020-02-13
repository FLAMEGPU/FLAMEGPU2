#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

class CUDAMessage;
/**
 * Interface for message specialisation
 * But we need a different specialisation for each Simulation specialisation
 */
template<typename SimSpecialisationMsg>
class MsgSpecialisationHandler {
 public:
    explicit MsgSpecialisationHandler(SimSpecialisationMsg &_sim_message)
        : sim_message(_sim_message)
    { }
    virtual ~MsgSpecialisationHandler() { }
    virtual void buildIndex() { }
    virtual const void *getMetaDataDevicePtr() const { return nullptr; }

 protected:
    SimSpecialisationMsg &sim_message;
};

/**
 * This empty class is used when messaging is not enabled for an agent function
 */
class MsgNone {
 public:
    class In {
     public:
        In() {}
        __device__ In(Curve::NamespaceHash /*agent fn hash*/, Curve::NamespaceHash /*message name hash*/, const void * /*metadata*/) {
        }
    };
    class Out {
     public:
        Out() {}
        __device__ Out(Curve::NamespaceHash /*agent fn hash*/, Curve::NamespaceHash /*message name hash*/, unsigned int /*streamid*/){
        }
    };
    template<typename SimSpecialisationMsg>
    class CUDAModelHandler : public MsgSpecialisationHandler<SimSpecialisationMsg> {
     public:
        explicit CUDAModelHandler(CUDAMessage &a)
            : MsgSpecialisationHandler(a)
        { }
    };
};

#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial3D.h"

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

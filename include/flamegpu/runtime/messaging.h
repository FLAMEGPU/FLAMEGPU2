#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

class CUDAMessage;
/**
 * Interface for message specialisation
 * But we need a different specialisation for each Simulation specialisation
 */
template<typename SimSpecialisationMsg>
class MsgSpecialisationHandler
{
public:
    MsgSpecialisationHandler(SimSpecialisationMsg &_sim_message)
        : sim_message(_sim_message)
    { }
    virtual ~MsgSpecialisationHandler() { }
    virtual void buildIndex() { }
protected:
    SimSpecialisationMsg &sim_message;
};

/**
 * This empty class is used when messaging is not enabled for an agent function
 */
class MsgNone
{
public:
    class In
    {
    public:
        In() {}
        __device__ In(Curve::NamespaceHash, Curve::NamespaceHash, unsigned int)
        {

        }
    };
    class Out
    {
    public:
        Out() {}
        __device__ Out(Curve::NamespaceHash, Curve::NamespaceHash, unsigned int)
        {

        }
    };
    template<typename SimSpecialisationMsg>
    class CUDAModelHandler : public MsgSpecialisationHandler<SimSpecialisationMsg>{
    public:
        CUDAModelHandler(CUDAMessage &a) 
            : MsgSpecialisationHandler(a)
        { }
    };
};

#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial3D.h"

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
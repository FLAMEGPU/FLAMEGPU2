#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial2D.h"

/**
 * This empty class is used when messaging is not enabled for an agent function
 */
class MsgNone
{
public:
    MsgNone() {}
    __device__ MsgNone(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, unsigned int len)
    {

    }
};
#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_